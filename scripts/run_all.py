"""Reproduce every experiment in the report, in order.

Ordering matters. All four models are trained first, then attacked, then
evaluated -- because the transfer matrix in the evaluation step needs every
checkpoint to exist before any of them is asked to craft examples for the others.

Run from the repository root:

    python scripts/run_all.py
    python scripts/run_all.py --quick     # small dataset, few epochs, for a smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antod.cli import cmd_attack, cmd_evaluate, cmd_generate, cmd_train  # noqa: E402
from antod.config import load_config  # noqa: E402
from antod.utils.common import get_logger  # noqa: E402

logger = get_logger("run_all")

MODEL_CONFIGS = ["cnn1d", "mlp", "hybrid", "hybrid_advtrain"]


def _load(name: str, quick: bool):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    if quick:
        cfg.dataset.n_flows = 2000
        cfg.train.epochs = 4
        cfg.train.patience = 4
        cfg.output.dir = f"experiments/runs/quick_{name}"
        cfg.dataset.path = "data/processed/quick.npz"
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="tiny run, for checking wiring")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()

    if not args.skip_generate:
        data_cfg = _load("dataset", args.quick)
        data_cfg.dataset.source = "synthetic"
        logger.info("=== generating dataset ===")
        if cmd_generate(data_cfg) != 0:
            return 1

    for stage, step in (("training", cmd_train), ("attacking", cmd_attack), ("evaluating", cmd_evaluate)):
        for name in MODEL_CONFIGS:
            logger.info("=== %s %s ===", stage, name)
            cfg = _load(name, args.quick)
            if args.quick:
                cfg.dataset.source = "cached"
            if step(cfg) != 0:
                logger.error("%s failed during %s", name, stage)
                return 1

    logger.info("all experiments finished in %.1f minutes", (time.time() - started) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
