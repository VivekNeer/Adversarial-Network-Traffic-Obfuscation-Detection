"""Sweep the share of each batch replaced by adversarial examples.

adv_ratio trades clean accuracy for robustness. This script trains the defended
config at several ratios, attacks each result with the evaluation attack, and
tabulates clean accuracy, constrained-PGD accuracy at eps=0.1 and targeted
evasion rate so the trade-off can be read off directly.

    python scripts/adv_ratio_sweep.py --ratios 0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from antod.cli import cmd_attack, cmd_train  # noqa: E402
from antod.config import load_config  # noqa: E402
from antod.utils.common import format_table, get_logger, write_markdown_table  # noqa: E402

logger = get_logger("adv_ratio_sweep")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "hybrid_advtrain.yaml"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args(argv)

    out_root = ROOT / "experiments" / "results" / "adv_ratio_sweep"
    rows = []
    for ratio in args.ratios:
        cfg = load_config(args.config)
        cfg.name = f"adv_ratio_{ratio:g}"
        cfg.train.adv_ratio = ratio
        cfg.output.dir = str(out_root / f"ratio_{ratio:g}")
        cfg.output.save_figures = False
        cfg.baselines = []
        if args.epochs:
            cfg.train.epochs = args.epochs
        logger.info("=== adv_ratio %.2f ===", ratio)
        if cmd_train(cfg) != 0 or cmd_attack(cfg) != 0:
            return 1
        metrics = json.loads((Path(cfg.output.dir) / "metrics.json").read_text())
        attack = json.loads((Path(cfg.output.dir) / "attack_results.json").read_text())
        curve = attack["robustness_curves"]["domain-constrained"]
        evasion = attack["evasion_curves"]["targeted at benign"]
        i = curve["eps"].index(0.1)
        rows.append(
            {
                "adv_ratio": ratio,
                "clean_accuracy": metrics["test"]["accuracy"],
                "clean_fpr": metrics["test"]["false_positive_rate"],
                "pgd_eps0.1_accuracy": curve["value"][i],
                "targeted_evasion_eps0.1": evasion["value"][i],
            }
        )
    print("\n" + format_table(rows))
    write_markdown_table(rows, out_root / "adv_ratio_sweep.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
