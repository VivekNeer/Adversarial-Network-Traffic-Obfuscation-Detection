"""Train one config under several seeds and report mean +/- std of the test metrics.

A single seed cannot support "model A beats model B". This script re-runs a config
with the dataset seed, split seed and training seed all varied together, and
writes a table that reports each metric as mean +/- std.

    python scripts/seed_variance.py configs/cnn1d.yaml --seeds 1 2 3
    python scripts/seed_variance.py configs/hybrid.yaml --seeds 1 2 3 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antod.cli import cmd_train  # noqa: E402
from antod.config import load_config  # noqa: E402
from antod.utils.common import (  # noqa: E402
    format_table,
    get_logger,
    save_json,
    write_markdown_table,
)

logger = get_logger("seed_variance")
METRICS = ["accuracy", "macro_f1", "obfuscated_recall", "malicious_recall", "false_positive_rate"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-flows", type=int)
    args = parser.parse_args(argv)

    base = load_config(args.config)
    name = base.name
    out_root = ROOT / "experiments" / "results" / f"{name}_seeds"
    per_seed: list[dict] = []

    for seed in args.seeds:
        cfg = load_config(args.config)
        cfg.name = f"{name}_seed{seed}"
        cfg.seed = cfg.train.seed = cfg.split.seed = seed
        # a cached dataset has one fixed seed; regenerate so the data varies too
        cfg.dataset.source = "synthetic"
        cfg.dataset.seed = seed
        cfg.output.dir = str(out_root / f"seed{seed}")
        cfg.output.save_figures = False
        cfg.baselines = []
        if args.epochs:
            cfg.train.epochs = args.epochs
        if args.n_flows:
            cfg.dataset.n_flows = args.n_flows

        logger.info("=== %s seed %d ===", name, seed)
        if cmd_train(cfg) != 0:
            return 1
        test = json.loads((Path(cfg.output.dir) / "metrics.json").read_text())["test"]
        per_seed.append({"seed": seed, **{m: test[m] for m in METRICS}})

    summary = {"model": name, "n_seeds": len(args.seeds)}
    for m in METRICS:
        vals = np.array([r[m] for r in per_seed])
        summary[m] = f"{vals.mean():.4f} +/- {vals.std(ddof=1) if len(vals) > 1 else 0:.4f}"

    print("\n" + format_table(per_seed))
    print("\n" + format_table([summary]))
    write_markdown_table(per_seed, out_root / "per_seed.md")
    write_markdown_table([summary], out_root / "summary.md")
    save_json({"per_seed": per_seed, "summary": summary}, out_root / "seed_variance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
