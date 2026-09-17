"""Tests for the adversarial attacks, concentrating on the domain constraints.

If a constrained attack can shrink a packet, send one earlier, flip a direction
or write into a padding slot, then the "robustness" numbers in the report describe
an adversary who can do things no adversary can do. These tests are the guarantee
that the reported threat model is the one that was actually simulated.
"""

from __future__ import annotations

import pytest
import torch

from antod.adversarial.attacks import (
    MAX_DELAY_SCALED,
    AttackConfig,
    fgsm,
    make_adversary,
    perturbation_norms,
    pgd,
    run_attack,
    sequence_bounds,
    stats_bounds,
)
from antod.adversarial.defenses import DefenseConfig, smoothed_predict
from antod.data.datasets import build_dataset, stratified_split
from antod.data.features import N_FEATURES
from antod.data.synth import BENIGN, SynthConfig
from antod.models import build_model

_CH_SIZE, _CH_IAT, _CH_DIR, _CH_MASK = 0, 1, 2, 3


@pytest.fixture(scope="module")
def fixture():
    ds = build_dataset(SynthConfig(n_flows=240, seed=21))
    splits = stratified_split(ds, seed=3)
    from antod.data.datasets import FlowTensorDataset

    seq, stats, y = FlowTensorDataset(splits.test, splits.scaler).tensors()
    model = build_model("hybrid").eval()
    return model, seq, stats, y, splits.scaler


# --------------------------------------------------------------------------- #
# the feasible set
# --------------------------------------------------------------------------- #
class TestSequenceBounds:
    def test_constrained_box_allows_only_growth_and_delay(self, fixture) -> None:
        _, seq, _, _, _ = fixture
        lo, hi = sequence_bounds(seq, constrained=True)
        valid = seq[:, _CH_MASK, :] > 0

        size = seq[:, _CH_SIZE, :]
        # upstream packets (positive) may grow up to +1; the clean value is the floor
        up = valid & (size > 0)
        assert torch.all(lo[:, _CH_SIZE, :][up] == size[up])
        assert torch.all(hi[:, _CH_SIZE, :][up] == 1.0)
        # downstream packets (negative) grow towards -1, so the clean value is the ceiling
        down = valid & (size < 0)
        assert torch.all(hi[:, _CH_SIZE, :][down] == size[down])
        assert torch.all(lo[:, _CH_SIZE, :][down] == -1.0)

        iat = seq[:, _CH_IAT, :]
        assert torch.all(lo[:, _CH_IAT, :][valid] == iat[valid])
        assert torch.all(hi[:, _CH_IAT, :][valid] >= iat[valid])

    def test_box_is_never_empty(self, fixture) -> None:
        """A flow that already has a long gap must not end up with lo > hi."""
        _, seq, _, _, _ = fixture
        lo, hi = sequence_bounds(seq, constrained=True)
        assert bool((lo <= hi).all())

    def test_long_gap_flow_keeps_a_valid_box(self) -> None:
        seq = torch.zeros(1, 4, 8)
        seq[0, _CH_IAT, :] = MAX_DELAY_SCALED + 0.5  # longer than the cap
        seq[0, _CH_MASK, :] = 1.0
        lo, hi = sequence_bounds(seq, constrained=True)
        assert bool((lo <= hi).all())

    def test_direction_and_mask_are_pinned_in_both_modes(self, fixture) -> None:
        _, seq, _, _, _ = fixture
        for constrained in (True, False):
            lo, hi = sequence_bounds(seq, constrained=constrained)
            for ch in (_CH_DIR, _CH_MASK):
                torch.testing.assert_close(lo[:, ch, :], seq[:, ch, :])
                torch.testing.assert_close(hi[:, ch, :], seq[:, ch, :])

    def test_padding_positions_are_frozen(self, fixture) -> None:
        _, seq, _, _, _ = fixture
        lo, hi = sequence_bounds(seq, constrained=True)
        pad = (seq[:, _CH_MASK : _CH_MASK + 1, :] == 0).expand_as(seq)
        torch.testing.assert_close(lo[pad], seq[pad])
        torch.testing.assert_close(hi[pad], seq[pad])

    def test_unconstrained_box_is_wider(self, fixture) -> None:
        _, seq, _, _, _ = fixture
        c_lo, c_hi = sequence_bounds(seq, constrained=True)
        u_lo, u_hi = sequence_bounds(seq, constrained=False)
        assert float((u_hi - u_lo).sum()) > float((c_hi - c_lo).sum())


class TestStatsBounds:
    def test_constrained_bounds_form_a_box(self, fixture) -> None:
        *_, scaler = fixture
        lo, hi = stats_bounds(scaler, constrained=True)
        assert lo.shape == hi.shape == (N_FEATURES,)
        assert bool((lo <= hi).all())

    def test_unconstrained_bounds_are_just_the_clip_range(self, fixture) -> None:
        *_, scaler = fixture
        lo, hi = stats_bounds(scaler, constrained=False)
        assert float(lo.min()) == pytest.approx(-scaler.clip)
        assert float(hi.max()) == pytest.approx(scaler.clip)


# --------------------------------------------------------------------------- #
# the attacks themselves
# --------------------------------------------------------------------------- #
class TestAttacks:
    def test_constrained_attack_only_pads_and_delays(self, fixture) -> None:
        """The central guarantee: every perturbed flow is one a real attacker could send."""
        model, seq, stats, y, scaler = fixture
        adv_seq, _ = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "both", eps=0.3, steps=12), scaler
        )
        valid = seq[:, _CH_MASK, :] > 0

        grew = adv_seq[:, _CH_SIZE, :].abs() + 1e-6 >= seq[:, _CH_SIZE, :].abs()
        assert bool(grew[valid].all()), "a packet was made smaller"

        same_sign = adv_seq[:, _CH_SIZE, :][valid] * seq[:, _CH_SIZE, :][valid] >= 0
        assert bool(same_sign.all()), "a packet changed direction via its signed size"

        delayed = adv_seq[:, _CH_IAT, :] + 1e-6 >= seq[:, _CH_IAT, :]
        assert bool(delayed[valid].all()), "a packet was sent earlier than it was produced"

        torch.testing.assert_close(adv_seq[:, _CH_DIR, :], seq[:, _CH_DIR, :])
        torch.testing.assert_close(adv_seq[:, _CH_MASK, :], seq[:, _CH_MASK, :])

        pad = (seq[:, _CH_MASK : _CH_MASK + 1, :] == 0).expand_as(seq)
        torch.testing.assert_close(adv_seq[pad], seq[pad])

    def test_perturbation_respects_the_eps_ball(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        eps = 0.05
        adv_seq, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "both", eps=eps, steps=10), scaler
        )
        assert float((adv_seq - seq).abs().max()) <= eps + 1e-6
        assert float((adv_stats - stats).abs().max()) <= eps + 1e-6

    def test_stats_stay_inside_the_domain_box(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        _, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "stats", eps=0.5, steps=10), scaler
        )
        lo, hi = stats_bounds(scaler, constrained=True)
        assert bool((adv_stats >= lo.unsqueeze(0) - 1e-5).all())
        assert bool((adv_stats <= hi.unsqueeze(0) + 1e-5).all())

    def test_attack_reduces_accuracy(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        clean = float((model.predict(seq, stats) == y).float().mean())
        adv_seq, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "both", eps=0.2, steps=15), scaler
        )
        attacked = float((model.predict(adv_seq, adv_stats) == y).float().mean())
        assert attacked <= clean

    def test_unconstrained_attack_is_at_least_as_strong(self, fixture) -> None:
        """A wider feasible set cannot help the defender; this is the sanity check
        behind reporting both numbers in the report."""
        model, seq, stats, y, scaler = fixture
        acc = {}
        for constrained in (True, False):
            a_seq, a_stats = run_attack(
                model,
                seq,
                stats,
                y,
                AttackConfig("pgd", "both", eps=0.2, steps=20, constrained=constrained),
                scaler,
            )
            acc[constrained] = float((model.predict(a_seq, a_stats) == y).float().mean())
        assert acc[False] <= acc[True] + 0.02

    def test_pgd_is_at_least_as_strong_as_fgsm(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        f_seq, f_stats = fgsm(model, seq, stats, y, eps=0.1, surface="both", scaler=scaler)
        p_seq, p_stats = pgd(model, seq, stats, y, eps=0.1, steps=20, surface="both", scaler=scaler)
        f_acc = float((model.predict(f_seq, f_stats) == y).float().mean())
        p_acc = float((model.predict(p_seq, p_stats) == y).float().mean())
        assert p_acc <= f_acc + 0.05

    def test_zero_epsilon_is_the_identity(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        adv_seq, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "both", eps=0.0, steps=5), scaler
        )
        torch.testing.assert_close(adv_seq, seq)
        torch.testing.assert_close(adv_stats, stats)

    def test_attack_named_none_is_the_identity(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        adv_seq, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("none", "both", eps=0.3), scaler
        )
        torch.testing.assert_close(adv_seq, seq)
        torch.testing.assert_close(adv_stats, stats)

    def test_surface_selection_leaves_the_other_input_alone(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        adv_seq, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "stats", eps=0.2, steps=5), scaler
        )
        torch.testing.assert_close(adv_seq, seq)
        assert float((adv_stats - stats).abs().max()) > 0

    def test_attacking_a_surface_the_model_ignores_is_a_no_op(self, fixture) -> None:
        """Otherwise a sequence attack on the MLP would be reported as robustness."""
        _, seq, stats, y, scaler = fixture
        mlp = build_model("mlp").eval()
        adv_seq, adv_stats = run_attack(
            mlp, seq, stats, y, AttackConfig("pgd", "seq", eps=0.3, steps=5), scaler
        )
        torch.testing.assert_close(adv_seq, seq)
        torch.testing.assert_close(adv_stats, stats)

    def test_targeted_attack_pushes_towards_benign(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        malicious = y != BENIGN
        adv_seq, adv_stats = run_attack(
            model,
            seq,
            stats,
            y,
            AttackConfig("pgd", "both", eps=0.3, steps=20, targeted=True, target_class=BENIGN),
            scaler,
        )
        before = float((model.predict(seq, stats)[malicious] == BENIGN).float().mean())
        after = float((model.predict(adv_seq, adv_stats)[malicious] == BENIGN).float().mean())
        assert after >= before

    def test_stats_attack_without_a_scaler_is_refused(self, fixture) -> None:
        model, seq, stats, y, _ = fixture
        with pytest.raises(ValueError, match="scaler"):
            run_attack(model, seq, stats, y, AttackConfig("pgd", "stats", eps=0.1), None)

    def test_attack_is_reproducible_under_seed(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        cfg = AttackConfig("pgd", "both", eps=0.1, steps=5, seed=99)
        a = run_attack(model, seq, stats, y, cfg, scaler)
        b = run_attack(model, seq, stats, y, cfg, scaler)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])

    def test_model_training_flag_is_restored(self, fixture) -> None:
        """Adversarial training calls this mid-epoch; leaving the model in eval
        mode would silently disable dropout for the rest of training."""
        model, seq, stats, y, scaler = fixture
        model.train()
        run_attack(model, seq, stats, y, AttackConfig("pgd", "both", eps=0.1, steps=2), scaler)
        assert model.training
        model.eval()

    def test_gradients_do_not_leak_into_the_model(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        model.zero_grad(set_to_none=True)
        run_attack(model, seq, stats, y, AttackConfig("pgd", "both", eps=0.1, steps=3), scaler)
        assert all(p.grad is None for p in model.parameters())


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"name": "deepfool"},
            {"surface": "payload"},
            {"eps": -0.1},
            {"name": "pgd", "steps": 0},
        ],
    )
    def test_bad_configs_are_rejected(self, kwargs) -> None:
        with pytest.raises(ValueError):
            AttackConfig(**kwargs)

    def test_default_step_size_crosses_the_ball(self) -> None:
        cfg = AttackConfig("pgd", eps=0.1, steps=10)
        assert cfg.step_size * cfg.steps > cfg.eps
        assert AttackConfig("fgsm", eps=0.1).step_size == pytest.approx(0.1)

    def test_explicit_alpha_wins(self) -> None:
        assert AttackConfig("pgd", eps=0.1, steps=10, alpha=0.007).step_size == 0.007

    def test_label_describes_the_attack(self) -> None:
        assert AttackConfig("none").label() == "clean"
        label = AttackConfig("pgd", "seq", eps=0.1, steps=7, targeted=True).label()
        assert "pgd" in label and "seq" in label and "targeted" in label
        assert "unconstrained" in AttackConfig("fgsm", constrained=False).label()

    def test_unknown_defense_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown defense"):
            DefenseConfig(method="magic")


class TestHelpers:
    def test_perturbation_norms_report_the_budget(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        adv_seq, adv_stats = run_attack(
            model, seq, stats, y, AttackConfig("pgd", "both", eps=0.1, steps=5), scaler
        )
        norms = perturbation_norms(seq, adv_seq, stats, adv_stats)
        assert norms["seq_linf"] <= 0.1 + 1e-6
        assert norms["stats_linf"] <= 0.1 + 1e-6
        assert norms["stats_l2_mean"] >= 0

    def test_make_adversary_matches_run_attack(self, fixture) -> None:
        model, seq, stats, y, scaler = fixture
        cfg = AttackConfig("pgd", "both", eps=0.1, steps=4, seed=5)
        direct = run_attack(model, seq, stats, y, cfg, scaler)
        via = make_adversary(cfg, scaler)(model, seq, stats, y)
        torch.testing.assert_close(direct[0], via[0])
        torch.testing.assert_close(direct[1], via[1])


class TestSmoothing:
    def test_zero_sigma_matches_the_plain_prediction(self, fixture) -> None:
        model, seq, stats, y, _ = fixture
        smoothed = smoothed_predict(model, seq, stats, sigma=0.0, n_samples=3)
        torch.testing.assert_close(smoothed, model.predict(seq, stats))

    def test_output_shape_and_range(self, fixture) -> None:
        model, seq, stats, y, _ = fixture
        pred = smoothed_predict(model, seq, stats, sigma=0.1, n_samples=5)
        assert pred.shape == y.shape
        assert bool(((pred >= 0) & (pred < 3)).all())

    def test_noise_never_touches_direction_mask_or_padding(self, fixture) -> None:
        """Smoothing must not ask the model about packets that do not exist."""
        model, seq, stats, y, _ = fixture
        captured: list[torch.Tensor] = []
        original = type(model).forward

        def spy(self, s, st):
            captured.append(s.clone())
            return original(self, s, st)

        type(model).forward = spy
        try:
            smoothed_predict(model, seq, stats, sigma=0.3, n_samples=2)
        finally:
            type(model).forward = original

        assert captured
        for noisy in captured:
            torch.testing.assert_close(noisy[:, _CH_DIR, :], seq[:, _CH_DIR, :])
            torch.testing.assert_close(noisy[:, _CH_MASK, :], seq[:, _CH_MASK, :])
            pad = (seq[:, _CH_MASK : _CH_MASK + 1, :] == 0).expand_as(seq)
            torch.testing.assert_close(noisy[pad], seq[pad])

    def test_reproducible_under_seed(self, fixture) -> None:
        model, seq, stats, _, _ = fixture
        a = smoothed_predict(model, seq, stats, sigma=0.1, n_samples=4, seed=7)
        b = smoothed_predict(model, seq, stats, sigma=0.1, n_samples=4, seed=7)
        torch.testing.assert_close(a, b)
