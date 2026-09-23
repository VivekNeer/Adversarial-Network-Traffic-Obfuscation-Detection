"""Lightweight, self-contained HTTP server for the ANTOD Web Dashboard.

Provides:
- Static file serving for the SPA interface (HTML/CSS/JS)
- Image and table serving from experiments/results/
- REST API endpoints for model metrics, comparison, predictions, and live inference
- Safe CLI execution runner with live output streaming
"""

from __future__ import annotations

import csv
import json
import logging
import mimetypes
import subprocess
import sys
import threading
import time
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("antod.gui")

WEB_DIR = Path(__file__).parent / "web"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

# In-memory cache for loaded model checkpoints and scalers
_MODEL_CACHE: dict[str, Any] = {}
_RUNNING_TASKS: dict[str, dict[str, Any]] = {}
_TASK_COUNTER = 0


def _load_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("failed to load json from %s: %s", path, exc)
        return None


def _load_text_safe(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("failed to read file %s: %s", path, exc)
        return None


def get_available_models() -> list[dict[str, Any]]:
    """Scan experiments/results/ for completed experiment folders."""
    models = []
    if not RESULTS_DIR.is_dir():
        return models

    for subdir in sorted(RESULTS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_file = subdir / "metrics.json"
        if not metrics_file.is_file():
            continue

        data = _load_json_safe(metrics_file)
        if not data:
            continue

        test_metrics = data.get("test", {})
        calib = data.get("calibration", {})
        br = data.get("precision_at_base_rate", {})

        models.append(
            {
                "id": subdir.name,
                "name": data.get("name", subdir.name),
                "model_type": data.get("model", subdir.name),
                "defense": data.get("defense", "none"),
                "accuracy": test_metrics.get("accuracy", 0.0),
                "macro_f1": test_metrics.get("macro_f1", 0.0),
                "obfuscated_recall": test_metrics.get("obfuscated_recall", 0.0),
                "malicious_recall": test_metrics.get("malicious_recall", 0.0),
                "false_positive_rate": test_metrics.get("false_positive_rate", 0.0),
                "roc_auc_macro": test_metrics.get("roc_auc_macro", 0.0),
                "ece_after": calib.get("ece_after", 0.0),
                "temperature": calib.get("temperature", 1.0),
                "precision_99": br.get("0.99", {}).get("precision", 0.0),
                "false_alerts_99": br.get("0.99", {}).get("false_alerts_per_10k", 0.0),
                "n_parameters": data.get("n_parameters", 0),
                "train_seconds": data.get("train_seconds", 0.0),
                "best_epoch": data.get("best_epoch", 0),
                "has_checkpoint": (subdir / "checkpoint.pt").is_file(),
                "has_attacks": (subdir / "attack_results.json").is_file(),
                "has_evaluation": (subdir / "evaluation.json").is_file(),
                "has_predictions": (subdir / "predictions.csv").is_file(),
            }
        )
    return models


def get_model_details(model_id: str) -> dict[str, Any] | None:
    folder = RESULTS_DIR / model_id
    if not folder.is_dir():
        return None

    metrics = _load_json_safe(folder / "metrics.json") or {}
    attacks = _load_json_safe(folder / "attack_results.json") or {}
    eval_data = _load_json_safe(folder / "evaluation.json") or {}
    config_text = _load_text_safe(folder / "config.yaml") or ""

    # Load tables
    tables = {}
    tables_dir = folder / "tables"
    if tables_dir.is_dir():
        for tfile in sorted(tables_dir.glob("*.md")):
            tables[tfile.stem] = tfile.read_text(encoding="utf-8")

    # List figures
    figures = []
    figures_dir = folder / "figures"
    if figures_dir.is_dir():
        for ffile in sorted(figures_dir.glob("*.png")):
            csv_counterpart = figures_dir / f"{ffile.stem}.csv"
            figures.append(
                {
                    "name": ffile.stem,
                    "url": f"/results/{model_id}/figures/{ffile.name}",
                    "has_csv": csv_counterpart.is_file(),
                    "csv_url": f"/results/{model_id}/figures/{csv_counterpart.name}"
                    if csv_counterpart.is_file()
                    else None,
                }
            )

    return {
        "id": model_id,
        "metrics": metrics,
        "attacks": attacks,
        "evaluation": eval_data,
        "config_yaml": config_text,
        "tables": tables,
        "figures": figures,
    }


def query_predictions(
    model_id: str,
    page: int = 1,
    limit: int = 50,
    filter_type: str = "all",
    profile: str = "all",
    search: str = "",
) -> dict[str, Any]:
    csv_path = RESULTS_DIR / model_id / "predictions.csv"
    if not csv_path.is_file():
        return {"rows": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}

    rows = []
    search_lower = search.lower().strip()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Filters
            is_correct = int(r.get("correct", 1)) == 1
            y_true = r.get("y_true", "")
            y_pred = r.get("y_pred", "")
            prof = r.get("profile", "")
            recipe = r.get("recipe", "")

            if profile != "all" and prof != profile:
                continue

            if filter_type == "errors" and is_correct:
                continue
            elif filter_type == "correct" and not is_correct:
                continue
            elif filter_type == "benign" and y_true != "benign":
                continue
            elif filter_type == "malicious_plain" and y_true != "malicious_plain":
                continue
            elif filter_type == "malicious_obfuscated" and y_true != "malicious_obfuscated":
                continue
            elif filter_type == "fp" and not (y_true == "benign" and y_pred != "benign"):
                continue
            elif filter_type == "fn" and not (y_true != "benign" and y_pred == "benign"):
                continue

            if search_lower:
                combined = f"{prof} {recipe} {y_true} {y_pred}".lower()
                if search_lower not in combined:
                    continue

            rows.append(
                {
                    "index": int(r.get("index", 0)),
                    "profile": prof,
                    "recipe": recipe,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "correct": is_correct,
                    "benign": float(r.get("benign", 0.0)),
                    "malicious_plain": float(r.get("malicious_plain", 0.0)),
                    "malicious_obfuscated": float(r.get("malicious_obfuscated", 0.0)),
                }
            )

    total = len(rows)
    total_pages = max(1, (total + limit - 1) // limit)
    page = max(1, min(page, total_pages))
    start = (page - 1) * limit
    end = start + limit
    page_rows = rows[start:end]

    return {
        "rows": page_rows,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }


def get_comparison_data() -> dict[str, Any]:
    """Gather metrics across all models and their baselines for side-by-side comparison."""
    models = get_available_models()
    comparison_rows = []
    feature_importances: dict[str, Any] = {}
    transfer_matrices: dict[str, Any] = {}

    for m in models:
        m_id = m["id"]
        details = get_model_details(m_id)
        if not details:
            continue
        comparison_rows.append(m)

        # Baseline entries
        metrics = details.get("metrics", {})
        baselines = metrics.get("baselines", {})
        for b_name, b_data in baselines.items():
            if not any(r["id"] == f"{m_id}_{b_name}" for r in comparison_rows):
                comparison_rows.append(
                    {
                        "id": f"{m_id}_{b_name}",
                        "name": f"{b_name} ({m_id})",
                        "model_type": b_name,
                        "defense": "none",
                        "accuracy": b_data.get("accuracy", 0.0),
                        "macro_f1": b_data.get("macro_f1", 0.0),
                        "obfuscated_recall": b_data.get("obfuscated_recall", 0.0),
                        "malicious_recall": b_data.get("malicious_recall", 0.0),
                        "false_positive_rate": b_data.get("false_positive_rate", 0.0),
                        "is_baseline": True,
                    }
                )

        fi = metrics.get("feature_importance", {})
        if fi:
            feature_importances[m_id] = fi

        eval_data = details.get("evaluation", {})
        tm = eval_data.get("transfer_matrix")
        if tm:
            transfer_matrices[m_id] = tm

    return {
        "models": comparison_rows,
        "feature_importances": feature_importances,
        "transfer_matrices": transfer_matrices,
    }


def do_live_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    """Run real-time inference on a synthesized or custom flow using model checkpoint."""
    import numpy as np

    from antod.data.obfuscation import (
        TRANSFORMS,
        Recipe,
        _sample_params,
    )
    from antod.data.profiles import BENIGN_PROFILES, MALICIOUS_PROFILES
    from antod.data.synth import BENIGN, LABEL_NAMES, MALICIOUS_OBFUSCATED, MALICIOUS_PLAIN
    from antod.inference import score_flow_windows
    from antod.train import load_checkpoint

    model_name = payload.get("model", "hybrid")
    checkpoint_path = RESULTS_DIR / model_name / "checkpoint.pt"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint for {model_name} not found at {checkpoint_path}")

    # Cache checkpoint
    if model_name not in _MODEL_CACHE:
        logger.info("loading checkpoint for %s into memory...", model_name)
        model, scaler, blob = load_checkpoint(checkpoint_path, device="cpu")
        _MODEL_CACHE[model_name] = (model, scaler, blob)
    else:
        model, scaler, blob = _MODEL_CACHE[model_name]

    profile_name = payload.get("profile", "video_streaming")
    recipe_name = payload.get("recipe", "random_padding+timing_jitter")
    seed = int(payload.get("seed", int(time.time()) % 10000))
    rng = np.random.default_rng(seed)

    # 1. Determine profile and base packets
    if profile_name in BENIGN_PROFILES:
        pkts = BENIGN_PROFILES[profile_name](rng)
        true_label = BENIGN
    elif profile_name in MALICIOUS_PROFILES:
        pkts = MALICIOUS_PROFILES[profile_name](rng)
        true_label = MALICIOUS_PLAIN
    else:
        # Default fallback
        profile_name = "web_browsing"
        pkts = BENIGN_PROFILES[profile_name](rng)
        true_label = BENIGN

    # 2. Apply obfuscation recipe if specified
    if recipe_name and recipe_name != "none":
        steps = []
        for step_name in recipe_name.split("+"):
            step_name = step_name.strip()
            if step_name in TRANSFORMS:
                steps.append((step_name, _sample_params(step_name, rng)))
        if steps:
            recipe = Recipe(steps)
            pkts = recipe.apply(pkts, rng)
            if true_label == MALICIOUS_PLAIN:
                true_label = MALICIOUS_OBFUSCATED
        else:
            recipe_name = "none"
    else:
        recipe_name = "none"

    try:
        # Score with model
        score = score_flow_windows(model, scaler, pkts, device="cpu")
        probs = [float(p) for p in score.proba]

        duration = (
            float(pkts[-1, 0] - pkts[0, 0])
            if pkts.shape[0] > 1
            else 0.0
        )
        total_bytes = int(np.abs(pkts[:, 1]).sum())
        avg_pkt = round(float(np.abs(pkts[:, 1]).mean()), 1)

        return {
            "predicted_class": score.label_name,
            "true_class": LABEL_NAMES[true_label],
            "probabilities": {
                "benign": round(probs[0], 4),
                "malicious_plain": round(probs[1], 4),
                "malicious_obfuscated": round(probs[2], 4),
            },
            "flow_summary": {
                "profile": profile_name,
                "recipe": recipe_name,
                "n_packets": int(pkts.shape[0]),
                "duration_seconds": round(duration, 4),
                "total_bytes": total_bytes,
                "avg_packet_size": avg_pkt,
            },
        }
    except Exception as exc:
        logger.exception("prediction error: %s", exc)
        return {"error": str(exc)}



class AntodDashboardHandler(SimpleHTTPRequestHandler):
    """Custom request handler that serves API endpoints and static assets."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. API: List available models
        if path == "/api/models":
            models = get_available_models()
            self._send_json(models)
            return

        # 2. API: Model Details
        if path.startswith("/api/model/"):
            model_id = path.replace("/api/model/", "").strip("/")
            details = get_model_details(model_id)
            if details:
                self._send_json(details)
            else:
                self._send_error(404, f"Model {model_id} not found")
            return

        # 3. API: Predictions Explorer
        if path.startswith("/api/predictions/"):
            model_id = path.replace("/api/predictions/", "").strip("/")
            page = int(query.get("page", ["1"])[0])
            limit = int(query.get("limit", ["50"])[0])
            filter_type = query.get("filter", ["all"])[0]
            profile = query.get("profile", ["all"])[0]
            search = query.get("search", [""])[0]

            data = query_predictions(model_id, page, limit, filter_type, profile, search)
            self._send_json(data)
            return

        # 4. API: Cross-Model Comparison
        if path == "/api/comparison":
            data = get_comparison_data()
            self._send_json(data)
            return

        # 5. API: Presets for live predictor
        if path == "/api/presets":
            from antod.data.obfuscation import ALL_RECIPES
            from antod.data.profiles import BENIGN_PROFILES, MALICIOUS_PROFILES

            self._send_json(
                {
                    "benign_profiles": [p.name for p in BENIGN_PROFILES],
                    "malicious_profiles": [p.name for p in MALICIOUS_PROFILES],
                    "recipes": ALL_RECIPES,
                }
            )
            return

        # 6. API: CLI Command task status
        if path == "/api/run/status":
            task_id = query.get("id", [""])[0]
            task = _RUNNING_TASKS.get(task_id)
            if not task:
                self._send_error(404, "Task not found")
                return
            self._send_json(task)
            return

        # 7. Experiment result static files (figures & tables)
        if path.startswith("/results/"):
            rel_path = path.replace("/results/", "", 1)
            file_path = RESULTS_DIR / rel_path
            if file_path.is_file():
                self._serve_file(file_path)
            else:
                self._send_error(404, f"Result file {rel_path} not found")
            return

        # 8. Web dashboard static files (index.html, style.css, app.js)
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8")) if body else {}

        # 1. Live prediction
        if path == "/api/predict":
            result = do_live_prediction(payload)
            self._send_json(result)
            return

        # 2. Run CLI command
        if path == "/api/run":
            cmd_args = payload.get("args", [])
            task_id = self._launch_command(cmd_args)
            self._send_json({"task_id": task_id, "status": "started"})
            return

        self._send_error(404, "POST endpoint not found")

    def _launch_command(self, args: list[str]) -> str:
        global _TASK_COUNTER
        _TASK_COUNTER += 1
        task_id = f"task_{_TASK_COUNTER}_{int(time.time())}"

        python_exe = sys.executable
        full_cmd = [python_exe, "-m", "antod.cli"] + args

        task_record = {
            "id": task_id,
            "cmd": " ".join(full_cmd),
            "status": "running",
            "output": "",
            "returncode": None,
            "start_time": time.time(),
        }
        _RUNNING_TASKS[task_id] = task_record

        def runner():
            try:
                proc = subprocess.Popen(
                    full_cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                for line in iter(proc.stdout.readline, ""):
                    task_record["output"] += line
                proc.stdout.close()
                proc.wait()
                task_record["status"] = "done" if proc.returncode == 0 else "failed"
                task_record["returncode"] = proc.returncode
            except Exception as e:
                task_record["status"] = "failed"
                task_record["output"] += f"\nError: {e}\n"

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return task_id

    def _serve_file(self, file_path: Path):
        content_type, _ = mimetypes.guess_type(str(file_path))
        content_type = content_type or "application/octet-stream"
        try:
            with file_path.open("rb") as f:
                data = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self._send_error(500, f"Error reading file: {e}")

    def _send_json(self, data: Any):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code: int, message: str):
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message, "code": code}).encode("utf-8"))


def start_server(port: int = 8000, host: str = "127.0.0.1", open_browser: bool = True):
    server = ThreadingHTTPServer((host, port), AntodDashboardHandler)
    url = f"http://{host}:{port}"
    print("\n=======================================================")
    print(f"  ANTOD Web Dashboard active at: {url}")
    print(f"  Pre-computed results loaded from: {RESULTS_DIR}")
    print("  Press Ctrl+C to stop the server.")
    print("=======================================================\n")

    if open_browser:
        import webbrowser

        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down ANTOD Web Dashboard...")
        server.server_close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ANTOD Web GUI Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    start_server(port=args.port, host=args.host, open_browser=not args.no_browser)
