"""Command-line entry point: ``antod generate | train | attack | evaluate | calibrate | all | predict``.

Every subcommand is driven by the same YAML file and writes into the same output
directory, and each one starts by re-resolving the dataset from the config rather
than trusting whatever is on disk. That is what makes a run reproducible from
``config.yaml`` alone: the resolved configuration is copied next to the results it
produced, so a number in the report can always be traced back to the settings that
generated it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

from antod.adversarial.attacks import AttackConfig
from antod.adversarial.defenses import adversarial_training
from antod.calibration import calibrate
from antod.config import ExperimentConfig, load_config, save_config
from antod.data.datasets import (
    FlowDataset,
    FlowTensorDataset,
    Splits,
    build_dataset,
    stratified_split,
)
from antod.evaluate import (
    attack_sweep,
    constrained_vs_unconstrained,
    evaluate_baselines,
    evaluate_clean,
    fit_baselines,
    per_profile_accuracy,
    per_technique_recall,
    robustness_curve,
    smoothing_sweep,
    transfer_matrix,
)
from antod.train import Trainer, load_checkpoint, save_checkpoint
from antod.utils.common import (
    format_table,
    get_logger,
    pick_device,
    save_json,
    set_seed,
    write_markdown_table,
)
from antod.utils.metrics import precision_at_base_rate
from antod.utils.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_ranked_bars,
    plot_robustness_curves,
    plot_training_curves,
)

logger = get_logger("antod")

DEFAULT_EPS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def _paths(cfg: ExperimentConfig) -> dict[str, Path]:
    root = cfg.output.path
    return {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "checkpoint": root / "checkpoint.pt",
        "metrics": root / "metrics.json",
        "config": root / "config.yaml",
    }


def resolve_dataset(cfg: ExperimentConfig) -> FlowDataset:
    """Produce the dataset the config asks for, generating or loading as needed."""
    path = Path(cfg.dataset.path)

    if cfg.dataset.source == "cached":
        logger.info("loading cached dataset from %s", path)
        return FlowDataset.load(path)

    if cfg.dataset.source == "real":
        from antod.data.real_loader import load_real_dataset

        logger.info("loading real capture from %s", path)
        flows = load_real_dataset(
            path,
            cfg=cfg.dataset.synth_config(),
            min_packets=cfg.dataset.min_packets,
            max_flows=cfg.dataset.max_flows,
        )
        return FlowDataset.from_flows(flows)

    logger.info("generating %d synthetic flows (seed %d)", cfg.dataset.n_flows, cfg.dataset.seed)
    return build_dataset(cfg.dataset.synth_config())


def resolve_splits(cfg: ExperimentConfig) -> Splits:
    ds = resolve_dataset(cfg)
    splits = stratified_split(
        ds,
        val_frac=cfg.split.val_frac,
        test_frac=cfg.split.test_frac,
        seed=cfg.split.seed,
    )
    logger.info("splits:\n%s", splits.summary())
    return splits


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #
def cmd_generate(cfg: ExperimentConfig) -> int:
    from antod.data.synth import generate_dataset, summarise

    set_seed(cfg.seed)
    if cfg.dataset.source == "synthetic":
        flows = generate_dataset(cfg.dataset.synth_config())
        print(summarise(flows))
        ds = FlowDataset.from_flows(flows)
    else:
        ds = resolve_dataset(cfg)

    path = Path(cfg.dataset.path)
    ds.save(path)
    logger.info("wrote %d flows to %s", len(ds), path)
    print(f"class counts: {ds.class_counts()}")
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(cfg: ExperimentConfig) -> int:
    set_seed(cfg.seed)
    paths = _paths(cfg)
    device = pick_device(cfg.train.device)
    splits = resolve_splits(cfg)

    if cfg.defense.method == "adversarial_training":
        logger.info("adversarial training with %s", cfg.defense.attack.label())
        result = adversarial_training(cfg.train, splits, cfg.defense)
    else:
        result = Trainer(cfg.train).fit(splits)

    test = evaluate_clean(result.model, splits.test, splits.scaler, device)
    logger.info("TEST  %s", test.summary())
    print(test.table())
    write_predictions(
        result.model, splits.test, splits.scaler, device, paths["root"] / "predictions.csv"
    )

    calibration = calibrate(
        result.model,
        tuple(t.to(device) for t in FlowTensorDataset(splits.val, splits.scaler).tensors()),
        tuple(t.to(device) for t in FlowTensorDataset(splits.test, splits.scaler).tensors()),
    )
    logger.info(
        "calibration: T=%.3f  ECE %.4f -> %.4f",
        calibration["temperature"],
        calibration["ece_before"],
        calibration["ece_after"],
    )
    base_rate = {f"{share:g}": precision_at_base_rate(test, share) for share in (0.9, 0.99, 0.999)}
    logger.info(
        "at 99%% benign traffic: precision %.3f, %.1f false alerts per 10k flows",
        base_rate["0.99"]["precision"],
        base_rate["0.99"]["false_alerts_per_10k"],
    )

    report: dict = {
        "name": cfg.name,
        "model": cfg.train.model,
        "defense": cfg.defense.method,
        "calibration": calibration,
        "precision_at_base_rate": base_rate,
        "best_epoch": result.best_epoch,
        "train_seconds": result.train_seconds,
        "n_parameters": result.model.n_parameters(),
        "val": result.best_val.to_dict(),
        "test": test.to_dict(),
    }

    if cfg.baselines:
        baselines = fit_baselines(splits, cfg.baselines, seed=cfg.seed)
        baseline_metrics = evaluate_baselines(baselines, splits)
        report["baselines"] = {k: v.to_dict() for k, v in baseline_metrics.items()}

        rows = [{"model": cfg.train.model, **_metric_row(test)}]
        rows += [{"model": name, **_metric_row(m)} for name, m in baseline_metrics.items()]
        print("\n" + format_table(rows))
        write_markdown_table(rows, paths["tables"] / "model_comparison.md")
        if cfg.output.save_figures:
            plot_model_comparison(
                rows,
                paths["figures"] / "model_comparison.png",
                title=f"{cfg.name}: deep models vs classical baselines",
            )

        importances = {
            name: model.top_features(15)
            for name, model in baselines.items()
            if model.top_features(1)
        }
        report["feature_importance"] = {k: dict(v) for k, v in importances.items()}
        if cfg.output.save_figures and "random_forest" in importances:
            plot_ranked_bars(
                dict(importances["random_forest"]),
                paths["figures"] / "feature_importance.png",
                title="Random forest: most informative flow statistics",
                xlabel="importance",
            )

    if cfg.output.save_figures:
        plot_training_curves(
            result.history,
            paths["figures"] / "training_curves.png",
            title=f"{cfg.name}: training history",
        )
        plot_confusion_matrix(
            test.confusion,
            paths["figures"] / "confusion_matrix.png",
            title=f"{cfg.name}: test-set confusion matrix",
        )

    if cfg.output.save_checkpoint:
        save_checkpoint(paths["checkpoint"], result, extra={"test": test.to_dict()})
        logger.info("checkpoint -> %s", paths["checkpoint"])

    save_json(report, paths["metrics"])
    save_config(cfg, paths["config"])
    logger.info("metrics -> %s", paths["metrics"])
    return 0


def _metric_row(metrics) -> dict:
    return {
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "obfuscated_recall": metrics.obfuscated_recall,
        "malicious_recall": metrics.malicious_recall,
        "false_positive_rate": metrics.false_positive_rate,
    }


# --------------------------------------------------------------------------- #
# attack
# --------------------------------------------------------------------------- #
def cmd_attack(cfg: ExperimentConfig) -> int:
    set_seed(cfg.seed)
    paths = _paths(cfg)
    device = pick_device(cfg.train.device)

    if not paths["checkpoint"].exists():
        logger.error("no checkpoint at %s -- run `antod train` first", paths["checkpoint"])
        return 1

    model, scaler, _ = load_checkpoint(paths["checkpoint"], device=cfg.train.device)
    splits = resolve_splits(cfg)
    test = splits.test.subsample(cfg.output.attack_flows, seed=cfg.seed)
    if cfg.output.attack_flows:
        logger.info("attacking a stratified subsample of %d test flows", len(test))

    configs = cfg.attacks or [
        AttackConfig(name="fgsm", surface="both", eps=0.1),
        AttackConfig(name="pgd", surface="both", eps=0.1, steps=10),
        AttackConfig(name="pgd", surface="both", eps=0.2, steps=20),
        AttackConfig(name="pgd", surface="both", eps=0.1, steps=10, targeted=True),
    ]
    outcomes = attack_sweep(model, test, scaler, configs, device)
    rows = [o.row() for o in outcomes]

    clean = evaluate_clean(model, test, scaler, device)
    rows.insert(
        0,
        {
            "attack": "clean",
            **_metric_row(clean),
            "evasion_rate": 0.0,
            "stats_linf": 0.0,
            "seq_linf": 0.0,
        },
    )
    print("\n" + format_table(rows))
    write_markdown_table(rows, paths["tables"] / "attack_sweep.md")

    surface = (
        "both" if (model.uses_seq and model.uses_stats) else ("seq" if model.uses_seq else "stats")
    )
    base = AttackConfig(name="pgd", surface=surface, eps=0.1, steps=10)

    curves = constrained_vs_unconstrained(model, test, scaler, base, DEFAULT_EPS, device)
    if cfg.output.save_figures:
        plot_robustness_curves(
            curves,
            paths["figures"] / "robustness_constrained.png",
            title="PGD: domain-constrained vs unconstrained perturbations",
        )

    evasion = {
        "untargeted": robustness_curve(
            model, test, scaler, base, DEFAULT_EPS, device, metric="evasion_rate"
        ),
        "targeted at benign": robustness_curve(
            model,
            test,
            scaler,
            replace(base, targeted=True),
            DEFAULT_EPS,
            device,
            metric="evasion_rate",
        ),
    }
    if cfg.output.save_figures:
        plot_robustness_curves(
            evasion,
            paths["figures"] / "evasion_rate.png",
            title="Share of malicious flows classified benign",
            ylabel="evasion rate",
        )

    smoothing = smoothing_sweep(
        model, test, scaler, [0.0, 0.02, 0.05, 0.1, 0.2], device, attack=base
    )
    print("\nrandomised smoothing\n" + format_table(smoothing))
    write_markdown_table(smoothing, paths["tables"] / "smoothing.md")

    # packet-space black-box attack on freshly generated malicious flows: the
    # strictly correct threat model, and comparable across all architectures
    from antod.adversarial.packet_attack import PacketAttackConfig, evaluate_packet_attack
    from antod.data.synth import generate_dataset

    probe_cfg = cfg.dataset.synth_config()
    probe_cfg.n_flows = 300
    probe_cfg.seed = cfg.seed + 1000
    probe_flows = generate_dataset(probe_cfg)
    packet_attack = evaluate_packet_attack(
        model, scaler, probe_flows, PacketAttackConfig(seed=cfg.seed), device
    )
    logger.info(
        "packet-space attack: evasion %.3f -> %.3f, mean overhead %.1f%%, %.0f queries",
        packet_attack["evasion_before"],
        packet_attack["evasion_after"],
        100 * packet_attack["mean_overhead"],
        packet_attack["mean_queries_to_evade"],
    )
    write_markdown_table([packet_attack], paths["tables"] / "packet_attack.md")

    save_json(
        {
            "clean": clean.to_dict(),
            "sweep": rows,
            "robustness_curves": {k: {"eps": v[0], "value": v[1]} for k, v in curves.items()},
            "evasion_curves": {k: {"eps": v[0], "value": v[1]} for k, v in evasion.items()},
            "smoothing": smoothing,
            "packet_attack": packet_attack,
        },
        paths["root"] / "attack_results.json",
    )
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def cmd_evaluate(cfg: ExperimentConfig) -> int:
    set_seed(cfg.seed)
    paths = _paths(cfg)
    device = pick_device(cfg.train.device)

    if not paths["checkpoint"].exists():
        logger.error("no checkpoint at %s -- run `antod train` first", paths["checkpoint"])
        return 1

    model, scaler, _ = load_checkpoint(paths["checkpoint"], device=cfg.train.device)
    splits = resolve_splits(cfg)

    technique = per_technique_recall(
        model, scaler, device, n_per_technique=400, synth=cfg.dataset.synth_config()
    )
    if cfg.output.save_figures:
        plot_ranked_bars(
            technique,
            paths["figures"] / "per_technique_recall.png",
            title="Recall by obfuscation technique",
            xlabel="recall on malicious_obfuscated",
            highlight_below=0.8,
        )
    write_markdown_table(
        [{"technique": k, "recall": v} for k, v in sorted(technique.items(), key=lambda kv: kv[1])],
        paths["tables"] / "per_technique_recall.md",
    )
    print(
        "\nper-technique recall\n"
        + format_table([{"technique": k, "recall": v} for k, v in technique.items()])
    )

    profile = per_profile_accuracy(model, splits.test, scaler, device)
    if cfg.output.save_figures:
        plot_ranked_bars(
            profile,
            paths["figures"] / "per_profile_accuracy.png",
            title="Accuracy by application profile",
            xlabel="accuracy",
        )
    write_markdown_table(
        [{"profile": k, "accuracy": v} for k, v in profile.items()],
        paths["tables"] / "per_profile_accuracy.md",
    )

    results: dict = {"per_technique_recall": technique, "per_profile_accuracy": profile}

    # transfer needs at least two trained checkpoints next to this run
    siblings = sorted(paths["root"].parent.glob("*/checkpoint.pt"))
    if len(siblings) >= 2:
        models = {}
        for ckpt in siblings:
            other, other_scaler, blob = load_checkpoint(ckpt, device=cfg.train.device)
            label = f"{blob['model_name']}"
            if blob.get("config", {}).get("adv_ratio") and "adv" in ckpt.parent.name:
                label += "+advtrain"
            models[f"{ckpt.parent.name}:{label}"] = other
        matrix = transfer_matrix(
            models,
            splits.test.subsample(cfg.output.transfer_flows, seed=cfg.seed),
            scaler,
            AttackConfig(name="pgd", surface="both", eps=0.1, steps=10),
            device,
        )
        print("\ntransfer accuracy (rows: crafted on, columns: evaluated on)")
        print(format_table(matrix))
        write_markdown_table(matrix, paths["tables"] / "transfer_matrix.md")
        results["transfer_matrix"] = matrix
    else:
        logger.info("only one checkpoint found; skipping the transfer matrix")

    save_json(results, paths["root"] / "evaluation.json")
    return 0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def write_predictions(model, ds: FlowDataset, scaler, device, path: Path) -> Path:
    """Dump one row per flow: provenance, true label, prediction and probabilities.

    Metrics say *how often* the model is wrong; this file says *which flows*, so a
    failure can be traced back to its profile and obfuscation recipe.
    """
    import csv

    from antod.data.synth import LABEL_NAMES

    seq, stats, y = FlowTensorDataset(ds, scaler).tensors()
    with torch.no_grad():
        proba = torch.softmax(model.eval()(seq.to(device), stats.to(device)), dim=1).cpu().numpy()
    pred = proba.argmax(axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "profile", "recipe", "y_true", "y_pred", "correct", *LABEL_NAMES])
        for i in range(len(ds)):
            w.writerow(
                [
                    i,
                    ds.profile[i],
                    ds.recipe[i],
                    LABEL_NAMES[int(y[i])],
                    LABEL_NAMES[int(pred[i])],
                    int(pred[i] == y[i]),
                    *[f"{p:.4f}" for p in proba[i]],
                ]
            )
    logger.info("per-flow predictions -> %s", path)
    return path


def cmd_predict(cfg: ExperimentConfig, csv_path: str | None, out: str | None) -> int:
    """Score a per-packet CSV with a trained checkpoint. No labels required."""
    from collections import Counter

    from antod.data.real_loader import load_packet_csv
    from antod.data.synth import LABEL_NAMES

    paths = _paths(cfg)
    if not paths["checkpoint"].exists():
        logger.error("no checkpoint at %s -- run `antod train` first", paths["checkpoint"])
        return 1
    if not csv_path:
        logger.error("predict needs --input <per-packet csv>")
        return 1

    device = pick_device(cfg.train.device)
    model, scaler, _ = load_checkpoint(paths["checkpoint"], device=cfg.train.device)
    flows = load_packet_csv(csv_path, min_packets=cfg.dataset.min_packets)
    ds = FlowDataset.from_flows(flows)

    target = Path(out) if out else paths["root"] / "predictions_external.csv"
    write_predictions(model, ds, scaler, device, target)

    seq, stats, _ = FlowTensorDataset(ds, scaler).tensors()
    pred = model.predict(seq.to(device), stats.to(device)).cpu().numpy()
    counts = Counter(LABEL_NAMES[int(p)] for p in pred)
    print(f"{len(ds)} flows scored -> {target}")
    for name in LABEL_NAMES:
        print(f"  {name:<22} {counts.get(name, 0):>6}")
    return 0


def cmd_calibrate(cfg: ExperimentConfig) -> int:
    """Fit temperature scaling on an existing checkpoint and record base-rate precision.

    Both are also computed by ``train``; this command exists so a checkpoint
    trained earlier can be calibrated without retraining, and so the numbers in
    ``metrics.json`` can be refreshed after the base-rate assumptions change.
    """
    from antod.utils.common import load_json

    paths = _paths(cfg)
    if not paths["checkpoint"].exists():
        logger.error("no checkpoint at %s -- run `antod train` first", paths["checkpoint"])
        return 1

    device = pick_device(cfg.train.device)
    model, scaler, _ = load_checkpoint(paths["checkpoint"], device=cfg.train.device)
    splits = resolve_splits(cfg)
    test = evaluate_clean(model, splits.test, scaler, device)

    calibration = calibrate(
        model,
        tuple(t.to(device) for t in FlowTensorDataset(splits.val, scaler).tensors()),
        tuple(t.to(device) for t in FlowTensorDataset(splits.test, scaler).tensors()),
    )
    base_rate = {f"{share:g}": precision_at_base_rate(test, share) for share in (0.9, 0.99, 0.999)}
    logger.info(
        "calibration: T=%.3f  ECE %.4f -> %.4f;  at 99%% benign: precision %.3f, %.1f false alerts / 10k",
        calibration["temperature"],
        calibration["ece_before"],
        calibration["ece_after"],
        base_rate["0.99"]["precision"],
        base_rate["0.99"]["false_alerts_per_10k"],
    )

    report = load_json(paths["metrics"]) if paths["metrics"].exists() else {"name": cfg.name}
    report["calibration"] = calibration
    report["precision_at_base_rate"] = base_rate
    save_json(report, paths["metrics"])
    write_predictions(model, splits.test, scaler, device, paths["root"] / "predictions.csv")
    return 0


def cmd_all(cfg: ExperimentConfig) -> int:
    for step in (cmd_train, cmd_attack, cmd_evaluate):
        code = step(cfg)
        if code != 0:
            return code
    return 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
COMMANDS = {
    "generate": cmd_generate,
    "train": cmd_train,
    "attack": cmd_attack,
    "evaluate": cmd_evaluate,
    "calibrate": cmd_calibrate,
    "all": cmd_all,
    "predict": cmd_predict,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="antod",
        description="Adversarial Network Traffic Obfuscation Detection",
    )
    parser.add_argument("command", choices=sorted(COMMANDS), help="what to run")
    parser.add_argument("--config", "-c", required=True, help="path to a YAML experiment config")
    parser.add_argument("--name", help="override the experiment name")
    parser.add_argument("--out", help="override the output directory")
    parser.add_argument("--epochs", type=int, help="override the epoch count")
    parser.add_argument("--model", help="override the model name")
    parser.add_argument("--n-flows", type=int, help="override the dataset size")
    parser.add_argument("--seed", type=int, help="override the seed")
    parser.add_argument("--device", help="cpu | cuda | auto")
    parser.add_argument("--input", help="predict: per-packet CSV to score")
    parser.add_argument("--predictions-out", help="predict: where to write the scored rows")
    return parser


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Command-line flags win over the file, so a sweep needs one config, not ten."""
    if args.name:
        cfg.name = args.name
    if args.out:
        cfg.output.dir = args.out
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.model:
        cfg.train.model = args.model
    if args.n_flows is not None:
        cfg.dataset.n_flows = args.n_flows
    if args.seed is not None:
        cfg.seed = args.seed
        cfg.train.seed = args.seed
        cfg.dataset.seed = args.seed
        cfg.split.seed = args.seed
    if args.device:
        cfg.train.device = args.device
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args)
    logger.info("experiment %r -> %s", cfg.name, cfg.output.dir)
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
    if args.command == "predict":
        return cmd_predict(cfg, args.input, args.predictions_out)
    return COMMANDS[args.command](cfg)


if __name__ == "__main__":
    sys.exit(main())
