"""Tests for calibration, base-rate precision and sliding-window inference."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from antod.calibration import TemperatureScaler, calibrate, expected_calibration_error
from antod.data.datasets import FlowTensorDataset, build_dataset, stratified_split
from antod.data.profiles import web_browsing
from antod.data.synth import SynthConfig
from antod.inference import score_flow_windows, windows_of
from antod.models import build_model
from antod.train import TrainConfig, Trainer
from antod.utils.metrics import compute_metrics, precision_at_base_rate


@pytest.fixture(scope="module")
def trained():
    splits = stratified_split(build_dataset(SynthConfig(n_flows=300, seed=41)), seed=2)
    result = Trainer(TrainConfig(model="mlp", epochs=3, batch_size=64, device="cpu")).fit(splits)
    return result, splits


class TestCalibration:
    def test_ece_is_zero_for_perfectly_calibrated_bins(self) -> None:
        proba = np.array([[0.8, 0.1, 0.1]] * 10)
        y = np.array([0] * 8 + [1] * 2)  # 80% correct at 80% confidence
        assert expected_calibration_error(proba, y) == pytest.approx(0.0, abs=1e-9)

    def test_ece_grows_with_overconfidence(self) -> None:
        y = np.array([0] * 5 + [1] * 5)
        modest = np.array([[0.6, 0.4, 0.0]] * 10)
        cocky = np.array([[0.99, 0.01, 0.0]] * 10)
        assert expected_calibration_error(cocky, y) > expected_calibration_error(modest, y)

    def test_temperature_does_not_change_argmax(self) -> None:
        logits = torch.randn(50, 3) * 4
        y = logits.argmax(dim=1)
        ts = TemperatureScaler().fit(logits, y)
        torch.testing.assert_close(ts.apply(logits).argmax(dim=1), y)

    def test_overconfident_logits_get_a_temperature_above_one(self) -> None:
        torch.manual_seed(0)
        y = torch.randint(0, 3, (400,))
        # the right class wins only 60% of the time, but always by a huge margin
        logits = torch.randn(400, 3)
        hit = torch.rand(400) < 0.6
        logits[hit, y[hit]] += 8.0
        logits[~hit, (y[~hit] + 1) % 3] += 8.0
        assert TemperatureScaler().fit(logits, y).temperature > 1.0

    def test_calibrate_reports_before_and_after(self, trained) -> None:
        result, splits = trained
        val = FlowTensorDataset(splits.val, splits.scaler).tensors()
        test = FlowTensorDataset(splits.test, splits.scaler).tensors()
        out = calibrate(result.model, val, test)
        assert out["temperature"] > 0
        assert 0 <= out["ece_before"] <= 1 and 0 <= out["ece_after"] <= 1
        assert 0 <= out["accuracy"] <= 1


class TestBaseRate:
    def test_precision_collapses_at_realistic_prevalence(self) -> None:
        y = np.array([0] * 100 + [1] * 100)
        pred = y.copy()
        pred[:2] = 1  # 2% FPR, 100% recall
        m = compute_metrics(y, pred)
        balanced = precision_at_base_rate(m, benign_share=0.5)["precision"]
        realistic = precision_at_base_rate(m, benign_share=0.999)["precision"]
        assert balanced > 0.95
        assert realistic < 0.1  # the same detector floods the analyst

    def test_alert_counts_add_up(self) -> None:
        y = np.array([0, 0, 1, 1])
        m = compute_metrics(y, np.array([0, 1, 1, 2]))
        out = precision_at_base_rate(m, 0.99)
        assert out["alerts_per_10k"] == pytest.approx(
            out["true_alerts_per_10k"] + out["false_alerts_per_10k"]
        )

    def test_bad_share_is_rejected(self) -> None:
        m = compute_metrics(np.array([0, 1]), np.array([0, 1]))
        with pytest.raises(ValueError):
            precision_at_base_rate(m, 1.0)


class TestWindows:
    def test_short_flow_is_a_single_window(self) -> None:
        pkts = np.zeros((50, 3))
        pkts[:, 0] = np.arange(50)
        assert len(windows_of(pkts, window=128)) == 1

    def test_windows_cover_the_tail(self) -> None:
        pkts = np.zeros((300, 3))
        pkts[:, 0] = np.arange(300)
        ws = windows_of(pkts, window=128, stride=100)
        assert all(w.shape[0] == 128 for w in ws)
        assert len(ws) == 3  # starts at 0, 100, and the forced tail at 172

    def test_windows_are_rebased_to_zero(self) -> None:
        pkts = np.zeros((200, 3))
        pkts[:, 0] = np.arange(200) * 0.01 + 5.0
        for w in windows_of(pkts, window=64, stride=64):
            assert w[0, 0] == 0.0

    def test_score_flow_windows_runs_on_long_flow(self, trained) -> None:
        result, splits = trained
        rng = np.random.default_rng(0)
        pkts = np.concatenate([web_browsing(rng) for _ in range(6)])
        pkts[:, 0] = np.sort(pkts[:, 0])
        for combine in ("mean", "max_malicious"):
            out = score_flow_windows(result.model, splits.scaler, pkts, combine=combine)
            assert out.n_windows >= 1
            assert out.proba.shape == (3,)
            assert out.per_window.shape == (out.n_windows, 3)
            assert 0 <= out.label < 3
            assert out.label_name

    def test_max_malicious_is_at_least_as_suspicious_as_mean(self, trained) -> None:
        result, splits = trained
        rng = np.random.default_rng(1)
        pkts = np.concatenate([web_browsing(rng) for _ in range(4)])
        pkts[:, 0] = np.sort(pkts[:, 0])
        mean = score_flow_windows(result.model, splits.scaler, pkts, combine="mean")
        mx = score_flow_windows(result.model, splits.scaler, pkts, combine="max_malicious")
        assert (1 - mx.proba[0]) >= (1 - mean.proba[0]) - 1e-6

    def test_unknown_combine_rule_is_rejected(self, trained) -> None:
        result, splits = trained
        with pytest.raises(ValueError):
            score_flow_windows(result.model, splits.scaler, np.zeros((10, 3)), combine="vote")

    def test_works_with_a_sequence_model(self, trained) -> None:
        _, splits = trained
        pkts = web_browsing(np.random.default_rng(2))
        out = score_flow_windows(build_model("cnn1d").eval(), splits.scaler, pkts)
        assert out.proba.shape == (3,)
