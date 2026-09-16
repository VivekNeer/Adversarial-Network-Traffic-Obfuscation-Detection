"""Tests for the model zoo: shapes, gradient routing, determinism and baselines."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from antod.data.features import N_FEATURES, SEQ_CHANNELS, SEQ_LEN
from antod.models import available_models, build_baseline, build_model
from antod.models.baselines import BASELINES

BATCH = 8


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ_CHANNELS, SEQ_LEN), torch.randn(BATCH, N_FEATURES)


@pytest.mark.parametrize("name", available_models())
class TestNeuralModels:
    def test_forward_shape(self, name: str, inputs) -> None:
        logits = build_model(name)(*inputs)
        assert logits.shape == (BATCH, 3)
        assert torch.isfinite(logits).all()

    def test_gradients_reach_exactly_the_declared_surfaces(self, name: str, inputs) -> None:
        """A sequence attack on a model that ignores the sequence would look like robustness.

        The ``uses_seq`` / ``uses_stats`` flags are what the attack code trusts to
        avoid reporting that, so they have to match where gradient actually flows.
        """
        model = build_model(name)
        seq, stats = (t.clone().requires_grad_(True) for t in inputs)
        model(seq, stats).sum().backward()

        seq_grad = 0.0 if seq.grad is None else float(seq.grad.abs().sum())
        stats_grad = 0.0 if stats.grad is None else float(stats.grad.abs().sum())
        assert (seq_grad > 0) is model.uses_seq
        assert (stats_grad > 0) is model.uses_stats

    def test_every_parameter_receives_gradient(self, name: str, inputs) -> None:
        """A dead branch would silently shrink the model; catch it here."""
        model = build_model(name)
        model(*inputs).sum().backward()
        dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not dead, f"no gradient for {dead}"

    def test_predict_and_probabilities_agree(self, name: str, inputs) -> None:
        model = build_model(name).eval()
        probs = model.probabilities(*inputs)
        assert probs.shape == (BATCH, 3)
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(BATCH))
        torch.testing.assert_close(model.predict(*inputs), probs.argmax(dim=1))

    def test_eval_mode_is_deterministic(self, name: str, inputs) -> None:
        """Dropout left on at inference time would make every metric noisy."""
        model = build_model(name).eval()
        with torch.no_grad():
            torch.testing.assert_close(model(*inputs), model(*inputs))

    def test_train_mode_dropout_is_active(self, name: str, inputs) -> None:
        model = build_model(name).train()
        with torch.no_grad():
            assert not torch.allclose(model(*inputs), model(*inputs))

    def test_describe_mentions_the_surfaces(self, name: str) -> None:
        text = build_model(name).describe()
        assert "parameters" in text
        assert ("sequence" in text) is build_model(name).uses_seq

    def test_batch_of_one_works(self, name: str) -> None:
        """BatchNorm raises on a single sample in train mode; inference must still work."""
        model = build_model(name).eval()
        with torch.no_grad():
            out = model(torch.randn(1, SEQ_CHANNELS, SEQ_LEN), torch.randn(1, N_FEATURES))
        assert out.shape == (1, 3)


class TestArchitecture:
    def test_models_stay_small_enough_for_the_dataset(self) -> None:
        """12k training flows do not support a large network; guard against drift."""
        for name in available_models():
            assert build_model(name).n_parameters() < 500_000

    def test_capacity_ablation_variants_really_are_smaller(self) -> None:
        assert build_model("cnn1d_small").n_parameters() < build_model("cnn1d").n_parameters()
        assert build_model("mlp_wide").n_parameters() > build_model("mlp").n_parameters()

    def test_cnn_embedding_concatenates_average_and_max_pooling(self, inputs) -> None:
        cnn = build_model("cnn1d")
        emb = cnn.embed(inputs[0])
        assert emb.shape == (BATCH, cnn.embed_dim)
        assert cnn.embed_dim == 256, "avg + max over 128 channels"

    def test_mismatched_architecture_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            build_model("cnn1d", channels=(32, 64), kernels=(7,), pools=(2, 2))

    def test_unknown_model_name_is_reported_clearly(self) -> None:
        with pytest.raises(KeyError, match="unknown model"):
            build_model("transformer_xl")

    def test_state_dict_round_trip(self, inputs) -> None:
        a = build_model("hybrid").eval()
        b = build_model("hybrid").eval()
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(a(*inputs), b(*inputs))


class TestBaselines:
    @pytest.fixture
    def toy(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        y = rng.integers(0, 3, 120)
        x = rng.normal(0, 1, (120, N_FEATURES)).astype(np.float32)
        x[:, 0] += y * 3.0  # one genuinely informative feature
        return x, y

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_fit_predict_and_proba(self, name: str, toy) -> None:
        x, y = toy
        model = build_baseline(name).fit(x, y)
        pred = model.predict(x)
        assert pred.shape == y.shape
        assert set(np.unique(pred)) <= {0, 1, 2}

        proba = model.predict_proba(x)
        assert proba.shape == (len(y), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("name", sorted(BASELINES))
    def test_importance_is_either_absent_or_normalised(self, name: str, toy) -> None:
        x, y = toy
        model = build_baseline(name).fit(x, y)
        imp = model.feature_importance()
        if imp is None:  # the RBF SVM has no per-feature attribution
            assert name == "rbf_svm"
            assert model.top_features() == []
            return
        assert len(imp) == N_FEATURES
        assert sum(imp.values()) == pytest.approx(1.0, abs=1e-6)
        assert len(model.top_features(5)) == 5

    def test_random_forest_finds_the_informative_feature(self, toy) -> None:
        from antod.data.features import FEATURE_NAMES

        x, y = toy
        top = build_baseline("random_forest").fit(x, y).top_features(3)
        assert FEATURE_NAMES[0] in [name for name, _ in top]

    def test_save_and_load(self, toy, tmp_path) -> None:
        x, y = toy
        model = build_baseline("logistic_regression").fit(x, y)
        path = tmp_path / "rf.joblib"
        model.save(path)
        from antod.models.baselines import SklearnBaseline

        back = SklearnBaseline.load(path)
        assert back.name == model.name
        np.testing.assert_array_equal(back.predict(x), model.predict(x))

    def test_unknown_baseline_name_is_reported_clearly(self) -> None:
        with pytest.raises(KeyError, match="unknown baseline"):
            build_baseline("xgboost")
