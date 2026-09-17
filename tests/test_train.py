"""Tests for the trainer, checkpointing, config loading and metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from antod.adversarial.attacks import AttackConfig
from antod.adversarial.defenses import DefenseConfig, adversarial_training
from antod.config import ExperimentConfig, config_from_dict, load_config, save_config
from antod.data.datasets import build_dataset, make_loaders, stratified_split
from antod.data.synth import BENIGN, MALICIOUS_OBFUSCATED, MALICIOUS_PLAIN, SynthConfig
from antod.train import (
    TrainConfig,
    Trainer,
    evaluate_model,
    load_checkpoint,
    predict_loader,
    save_checkpoint,
)
from antod.utils.metrics import clean_vs_attacked, compute_metrics, recall_by_group


@pytest.fixture(scope="module")
def splits():
    return stratified_split(build_dataset(SynthConfig(n_flows=300, seed=31)), seed=2)


@pytest.fixture(scope="module")
def trained(splits):
    cfg = TrainConfig(model="mlp", epochs=3, patience=3, batch_size=64, device="cpu")
    return Trainer(cfg).fit(splits), splits


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class TestTrainer:
    def test_produces_history_and_a_best_epoch(self, trained) -> None:
        result, _ = trained
        assert len(result.history) == 3
        assert 1 <= result.best_epoch <= 3
        assert result.train_seconds > 0
        assert set(result.curves()) >= {"train_loss", "val_acc", "val_macro_f1"}

    def test_learns_something_above_chance(self, trained) -> None:
        result, splits = trained
        metrics = evaluate_model(
            result.model, make_loaders(splits, batch_size=64)["test"], torch.device("cpu")
        )
        assert metrics.accuracy > 0.4  # three classes, chance is 0.33

    def test_returns_the_best_weights_not_the_last(self, splits) -> None:
        """Early stopping is pointless if the final epoch's weights are kept."""
        cfg = TrainConfig(model="mlp", epochs=6, patience=6, batch_size=64, device="cpu")
        result = Trainer(cfg).fit(splits)
        best = max(result.history, key=lambda h: h["val_macro_f1"])
        val = evaluate_model(
            result.model, make_loaders(splits, batch_size=64)["val"], torch.device("cpu")
        )
        assert val.macro_f1 == pytest.approx(best["val_macro_f1"], abs=1e-4)

    def test_early_stopping_can_end_a_run_sooner(self, splits) -> None:
        cfg = TrainConfig(model="mlp", epochs=40, patience=1, batch_size=64, device="cpu", lr=1e-6)
        result = Trainer(cfg).fit(splits)
        assert len(result.history) < 40

    def test_training_is_reproducible_under_seed(self, splits) -> None:
        cfg = TrainConfig(model="mlp", epochs=2, batch_size=64, device="cpu", seed=123)
        a = Trainer(cfg).fit(splits)
        b = Trainer(cfg).fit(splits)
        assert a.history[-1]["train_loss"] == pytest.approx(b.history[-1]["train_loss"], abs=1e-6)

    @pytest.mark.parametrize("scheduler", ["cosine", "plateau", "none"])
    def test_every_scheduler_runs(self, splits, scheduler: str) -> None:
        cfg = TrainConfig(model="mlp", epochs=2, batch_size=64, device="cpu", scheduler=scheduler)
        assert len(Trainer(cfg).fit(splits).history) == 2

    def test_selection_metric_is_honoured(self, splits) -> None:
        cfg = TrainConfig(
            model="mlp",
            epochs=4,
            patience=4,
            batch_size=64,
            device="cpu",
            selection_metric="obfuscated_recall",
        )
        result = Trainer(cfg).fit(splits)
        best = max(h["val_obf_recall"] for h in result.history)
        assert result.best_val.obfuscated_recall == pytest.approx(best, abs=1e-4)

    @pytest.mark.parametrize(
        "kwargs",
        [{"scheduler": "magic"}, {"selection_metric": "auc"}, {"adv_ratio": 1.5}],
    )
    def test_bad_train_configs_are_rejected(self, kwargs) -> None:
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)

    def test_predict_loader_returns_aligned_arrays(self, trained) -> None:
        result, splits = trained
        y, pred, proba = predict_loader(
            result.model, make_loaders(splits, batch_size=32)["test"], torch.device("cpu")
        )
        assert y.shape == pred.shape == (len(splits.test),)
        assert proba.shape == (len(splits.test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


class TestAdversarialTraining:
    def test_runs_and_keeps_the_same_loop(self, splits) -> None:
        cfg = TrainConfig(model="hybrid", epochs=2, batch_size=64, device="cpu", adv_ratio=0.5)
        defense = DefenseConfig(attack=AttackConfig("pgd", "both", eps=0.05, steps=2))
        result = adversarial_training(cfg, splits, defense)
        assert len(result.history) == 2

    def test_defense_none_falls_back_to_plain_training(self, splits) -> None:
        cfg = TrainConfig(model="mlp", epochs=2, batch_size=64, device="cpu", seed=7)
        defended = adversarial_training(cfg, splits, DefenseConfig(method="none"))
        plain = Trainer(cfg).fit(splits)
        assert defended.history[-1]["train_loss"] == pytest.approx(
            plain.history[-1]["train_loss"], abs=1e-6
        )

    def test_adv_ratio_zero_leaves_the_batch_clean(self, splits) -> None:
        cfg = TrainConfig(model="mlp", epochs=2, batch_size=64, device="cpu", adv_ratio=0.0, seed=9)
        defended = adversarial_training(
            cfg, splits, DefenseConfig(attack=AttackConfig("pgd", "stats", eps=0.1, steps=2))
        )
        plain = Trainer(cfg).fit(splits)
        assert defended.history[-1]["train_loss"] == pytest.approx(
            plain.history[-1]["train_loss"], abs=1e-6
        )


class TestCheckpoints:
    def test_round_trip_preserves_predictions(self, trained, tmp_path) -> None:
        result, splits = trained
        path = save_checkpoint(tmp_path / "ckpt.pt", result)
        model, scaler, blob = load_checkpoint(path, device="cpu")

        seq = torch.from_numpy(splits.test.seq)
        stats = torch.from_numpy(scaler.transform(splits.test.stats))
        torch.testing.assert_close(
            model.predict(seq, stats),
            result.model.predict(seq, torch.from_numpy(result.scaler.transform(splits.test.stats))),
        )
        assert blob["model_name"] == "mlp"
        assert blob["best_epoch"] == result.best_epoch

    def test_scaler_travels_with_the_weights(self, trained, tmp_path) -> None:
        """A model applied with the wrong scaling produces plausible nonsense, not an error."""
        result, _ = trained
        _, scaler, _ = load_checkpoint(save_checkpoint(tmp_path / "c.pt", result), device="cpu")
        np.testing.assert_allclose(scaler.state_dict()["mean"], result.scaler.state_dict()["mean"])


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class TestConfig:
    def test_defaults_build(self) -> None:
        cfg = config_from_dict({})
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.train.model == "cnn1d"

    def test_nested_sections_are_built_as_dataclasses(self) -> None:
        """With postponed annotations, field.type is a string -- if the hints are not
        resolved, a nested mapping reaches the constructor as a raw dict."""
        cfg = config_from_dict({"dataset": {"n_flows": 99}, "train": {"model": "mlp", "epochs": 7}})
        assert cfg.dataset.n_flows == 99
        assert cfg.train.model == "mlp"
        assert isinstance(cfg.train, TrainConfig)
        assert cfg.train.epochs == 7

    def test_unknown_key_is_a_hard_error(self) -> None:
        """A silently ignored typo would run at the default and never be noticed."""
        with pytest.raises(ValueError, match="unknown option"):
            config_from_dict({"train": {"epocs": 200}})

    def test_unknown_top_level_key_is_a_hard_error(self) -> None:
        with pytest.raises(ValueError, match="unknown option"):
            config_from_dict({"modle": "cnn1d"})

    def test_attack_list_is_parsed(self) -> None:
        cfg = config_from_dict(
            {"attacks": [{"name": "fgsm", "eps": 0.05}, {"name": "pgd", "steps": 3}]}
        )
        assert [a.name for a in cfg.attacks] == ["fgsm", "pgd"]
        assert cfg.attacks[0].eps == 0.05

    def test_nested_defense_attack_is_parsed(self) -> None:
        cfg = config_from_dict(
            {
                "defense": {
                    "method": "adversarial_training",
                    "attack": {"name": "pgd", "eps": 0.2, "steps": 4},
                }
            }
        )
        assert cfg.defense.method == "adversarial_training"
        assert isinstance(cfg.defense.attack, AttackConfig)
        assert cfg.defense.attack.steps == 4

    def test_bad_attack_inside_defense_is_reported_with_its_path(self) -> None:
        with pytest.raises(ValueError, match="defense.attack"):
            config_from_dict({"defense": {"attack": {"nmae": "pgd"}}})

    def test_synth_config_is_derived_from_the_dataset_section(self) -> None:
        cfg = config_from_dict({"dataset": {"n_flows": 50, "benign_obfuscation_rate": 0.2}})
        synth = cfg.dataset.synth_config()
        assert synth.n_flows == 50
        assert synth.benign_obfuscation_rate == 0.2

    def test_bad_dataset_source_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset source"):
            config_from_dict({"dataset": {"source": "magic"}})

    def test_split_fractions_must_leave_a_training_set(self) -> None:
        with pytest.raises(ValueError, match="training split"):
            config_from_dict({"split": {"val_frac": 0.6, "test_frac": 0.5}})

    def test_yaml_round_trip(self, tmp_path) -> None:
        cfg = config_from_dict({"name": "x", "train": {"model": "hybrid", "epochs": 5}})
        path = save_config(cfg, tmp_path / "c.yaml")
        assert yaml.safe_load(path.read_text())["train"]["model"] == "hybrid"
        assert load_config(path).train.epochs == 5

    def test_missing_file_is_reported(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    @pytest.mark.parametrize("name", ["dataset", "cnn1d", "mlp", "hybrid", "hybrid_advtrain"])
    def test_shipped_configs_load(self, name: str) -> None:
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / f"{name}.yaml")
        assert cfg.name


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        m = compute_metrics(y, y)
        assert m.accuracy == 1.0
        assert m.macro_f1 == 1.0
        assert m.obfuscated_recall == 1.0
        assert m.false_positive_rate == 0.0

    def test_collapsing_obfuscated_into_plain_is_caught_by_macro_f1(self) -> None:
        """The failure the trainer selects against: accuracy barely moves, macro-F1 does."""
        y = np.array([BENIGN] * 10 + [MALICIOUS_PLAIN] * 10 + [MALICIOUS_OBFUSCATED] * 10)
        collapsed = np.array([BENIGN] * 10 + [MALICIOUS_PLAIN] * 20)
        m = compute_metrics(y, collapsed)
        assert m.accuracy == pytest.approx(2 / 3)
        assert m.obfuscated_recall == 0.0
        assert m.malicious_recall == 1.0  # still alarms on every attack
        assert m.macro_f1 < 0.6

    def test_false_positive_rate_counts_only_benign_flows(self) -> None:
        y = np.array([BENIGN, BENIGN, BENIGN, BENIGN, MALICIOUS_PLAIN])
        pred = np.array([BENIGN, MALICIOUS_PLAIN, BENIGN, BENIGN, MALICIOUS_PLAIN])
        assert compute_metrics(y, pred).false_positive_rate == pytest.approx(0.25)

    def test_malicious_recall_ignores_which_malicious_class(self) -> None:
        y = np.array([MALICIOUS_PLAIN, MALICIOUS_OBFUSCATED])
        pred = np.array([MALICIOUS_OBFUSCATED, MALICIOUS_PLAIN])
        m = compute_metrics(y, pred)
        assert m.malicious_recall == 1.0
        assert m.accuracy == 0.0

    def test_auc_needs_every_class_present(self) -> None:
        y = np.array([MALICIOUS_OBFUSCATED] * 4)
        proba = np.full((4, 3), 1 / 3)
        assert compute_metrics(y, y, proba).roc_auc_macro is None

    def test_auc_is_computed_when_all_classes_appear(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        proba = np.eye(3)[y] * 0.7 + 0.1  # rows must sum to 1
        auc = compute_metrics(y, y, proba).roc_auc_macro
        assert auc is not None and auc > 0.9

    def test_confusion_matrix_is_three_by_three(self) -> None:
        y = np.array([0, 1, 2])
        m = compute_metrics(y, y)
        assert len(m.confusion) == 3
        assert all(len(row) == 3 for row in m.confusion)

    def test_summary_and_table_render(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        m = compute_metrics(y, y)
        assert "obf-recall" in m.summary()
        assert "malicious_obfuscated" in m.table()

    def test_recall_by_group(self) -> None:
        y = np.array([1, 1, 2, 2])
        pred = np.array([1, 0, 2, 2])
        groups = np.array(["a", "a", "b", "b"])
        got = recall_by_group(y, pred, groups)
        assert got["a"] == pytest.approx(0.5)
        assert got["b"] == pytest.approx(1.0)

    def test_clean_vs_attacked_reports_the_drop(self) -> None:
        y = np.array([0, 1, 2, 0, 1, 2])
        clean = compute_metrics(y, y)
        attacked = compute_metrics(y, np.zeros_like(y))
        delta = clean_vs_attacked(clean, attacked)
        assert delta["accuracy_drop"] == pytest.approx(1 - 1 / 3)
        assert delta["obfuscated_recall_drop"] == pytest.approx(1.0)
