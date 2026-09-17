"""Tests for the packet-space black-box attack and the GRU baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from antod.adversarial.packet_attack import (
    PacketAttackConfig,
    _propose,
    evaluate_packet_attack,
    packet_space_attack,
)
from antod.data.datasets import build_dataset, stratified_split
from antod.data.profiles import MTU, c2_beacon
from antod.data.synth import BENIGN, SynthConfig, generate_dataset
from antod.models import build_model
from antod.train import TrainConfig, Trainer


@pytest.fixture(scope="module")
def trained():
    splits = stratified_split(build_dataset(SynthConfig(n_flows=300, seed=51)), seed=2)
    result = Trainer(TrainConfig(model="hybrid", epochs=3, batch_size=64, device="cpu")).fit(splits)
    return result.model, splits.scaler


class TestProposal:
    def test_edits_are_pad_and_delay_only(self) -> None:
        rng = np.random.default_rng(0)
        pkts = c2_beacon(rng)
        for _ in range(20):
            out = _propose(pkts, PacketAttackConfig(), rng)
            assert out.shape == pkts.shape
            assert np.all(out[:, 1] >= pkts[:, 1]), "a packet shrank"
            assert np.all(out[:, 1] <= MTU)
            assert np.all(np.diff(out[:, 0]) >= -1e-12), "time ran backwards"
            assert np.all(out[:, 0] >= pkts[:, 0] - 1e-12), "a packet was sent earlier"
            np.testing.assert_array_equal(out[:, 2], pkts[:, 2])
            assert out[0, 0] == 0.0

    def test_single_packet_flow_does_not_crash(self) -> None:
        rng = np.random.default_rng(0)
        pkts = np.array([[0.0, 100.0, 1.0]])
        out = _propose(pkts, PacketAttackConfig(), rng)
        assert out.shape == (1, 3)


class TestAttack:
    def test_result_is_physically_legal(self, trained) -> None:
        model, scaler = trained
        pkts = c2_beacon(np.random.default_rng(3))
        res = packet_space_attack(model, scaler, pkts, PacketAttackConfig(max_queries=30))
        adv = res.adversarial
        assert np.all(adv[:, 1] >= pkts[:, 1])
        assert np.all(adv[:, 0] >= pkts[:, 0] - 1e-12)
        assert res.bytes_added >= 0 and res.delay_added >= -1e-9
        assert 1 <= res.queries <= 30
        assert 0 <= res.overhead

    def test_target_probability_never_decreases(self, trained) -> None:
        """Greedy acceptance means the final score is at least the starting score."""
        model, scaler = trained
        pkts = c2_beacon(np.random.default_rng(4))
        res = packet_space_attack(model, scaler, pkts, PacketAttackConfig(max_queries=40))
        assert res.p_target_after >= res.p_target_before - 1e-9

    def test_stops_at_one_query_if_already_benign(self, trained) -> None:
        model, scaler = trained
        # find a flow the model already calls benign, if any, in a small batch
        flows = generate_dataset(SynthConfig(n_flows=40, seed=8))
        for f in flows:
            res = packet_space_attack(model, scaler, f.packets, PacketAttackConfig(max_queries=5))
            if res.p_target_before == max(res.p_target_before, res.p_target_after) and res.success:
                assert res.queries == 1
                return
        pytest.skip("no flow in the sample was already read as benign")

    def test_reproducible_under_seed(self, trained) -> None:
        model, scaler = trained
        pkts = c2_beacon(np.random.default_rng(5))
        a = packet_space_attack(model, scaler, pkts, PacketAttackConfig(max_queries=20, seed=3))
        b = packet_space_attack(model, scaler, pkts, PacketAttackConfig(max_queries=20, seed=3))
        np.testing.assert_allclose(a.adversarial, b.adversarial)

    def test_evaluate_summarises_malicious_flows_only(self, trained) -> None:
        model, scaler = trained
        flows = generate_dataset(SynthConfig(n_flows=30, seed=9))
        out = evaluate_packet_attack(model, scaler, flows, PacketAttackConfig(max_queries=10))
        assert out["n_flows"] == sum(1 for f in flows if f.label != BENIGN)
        assert 0 <= out["evasion_before"] <= out["evasion_after"] <= 1
        assert out["max_queries"] == 10

    def test_evaluate_refuses_without_malicious_flows(self, trained) -> None:
        model, scaler = trained
        flows = [f for f in generate_dataset(SynthConfig(n_flows=30, seed=9)) if f.label == 0]
        with pytest.raises(ValueError):
            evaluate_packet_attack(model, scaler, flows, PacketAttackConfig())


class TestGRU:
    def test_registered_and_runs(self) -> None:
        model = build_model("gru")
        seq = torch.randn(4, 4, 128)
        seq[:, 3, :] = 1.0
        seq[:, 3, 100:] = 0.0  # last 28 positions are padding
        out = model(seq, torch.randn(4, 51))
        assert out.shape == (4, 3)
        assert model.uses_seq and not model.uses_stats

    def test_padding_does_not_change_the_embedding(self) -> None:
        """Packing to true length means garbage in the padded region cannot leak in."""
        model = build_model("gru").eval()
        seq = torch.randn(2, 4, 128)
        seq[:, 3, :] = 0.0
        seq[:, 3, :60] = 1.0
        seq[:, :3, 60:] = 0.0
        other = seq.clone()
        other[:, :3, 60:] = torch.randn(2, 3, 68)
        with torch.no_grad():
            torch.testing.assert_close(model.embed(seq), model.embed(other))

    def test_flows_of_different_lengths_in_one_batch(self) -> None:
        model = build_model("gru").eval()
        seq = torch.randn(3, 4, 128)
        seq[:, 3, :] = 0.0
        for i, n in enumerate((5, 64, 128)):
            seq[i, 3, :n] = 1.0
        with torch.no_grad():
            out = model(seq, torch.zeros(3, 51))
        assert out.shape == (3, 3) and torch.isfinite(out).all()

    def test_stays_small(self) -> None:
        assert build_model("gru").n_parameters() < 200_000
