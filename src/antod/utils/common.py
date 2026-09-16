"""Seeding, logging and small IO helpers shared across the pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
DATE_FORMAT = "%H:%M:%S"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every generator the pipeline touches.

    ``deterministic`` also pins cuDNN's algorithm choice. It costs some speed, but
    a robustness study whose numbers move between runs cannot support a claim
    about whether adversarial training helped.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "antod", level: int = logging.INFO) -> logging.Logger:
    """Console logger, configured once per name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA when present, otherwise CPU."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """Write JSON, creating parents and coercing numpy scalars."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, default=_json_default), encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def format_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Fixed-width text table, for console output and for pasting into the report."""
    if not rows:
        return "(no rows)"
    columns = columns or list(rows[0])
    widths = {c: max(len(c), *(len(_fmt(r.get(c, ""))) for r in rows)) for c in columns}

    head = "  ".join(c.ljust(widths[c]) for c in columns)
    rule = "  ".join("-" * widths[c] for c in columns)
    body = ["  ".join(_fmt(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows]
    return "\n".join([head, rule, *body])


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_table(rows: list[dict[str, Any]], path: str | Path, columns=None) -> Path:
    """Markdown table, so results land in the report without retyping."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("(no rows)\n", encoding="utf-8")
        return path

    columns = columns or list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
