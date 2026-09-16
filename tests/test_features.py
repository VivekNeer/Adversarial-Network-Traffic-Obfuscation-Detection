"""Tests for the two feature views and the dataset/scaling machinery."""

from __future__ import annotations

import numpy as np
import pytest

from antod.data.datasets import (
    FeatureConstraints,
    FlowDataset,
    FlowTensorDataset,
    StatsScaler,
    build_dataset,
    make_loaders,
    stratified_split,
)
from antod.data.features import (
    FEATURE_NAMES,
    N_FEATURES,
    SEQ_CHANNELS,
    SEQ_LEN,
    flow_statistics,
    sequence_tensor,
    stats_vector,
)
from antod.data.obfuscation import block_padding, constant_rate_shaping
from antod.data.profiles import MTU, c2_beacon, web_browsing
from antod.data.synth import SynthConfig


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(3)


@pytest.fixture(scope="module")
def dataset() -> FlowDataset:
    return build_dataset(SynthConfig(n_flows=300, seed=4))


# --------------------------------------------------------------------------- #
# sequence view
# --------------------------------------------------------------------------- #
class TestSequenceView:
    def test_shape_and_dtype(self, rng: np.random.Generator) -> None:
        seq = sequence_tensor(web_browsing(rng))
        assert seq.shape == (SEQ_CHANNELS, SEQ_LEN)
        assert seq.dtype == np.float32

    def test_short_flow_is_masked_not_faked(self, rng: np.random.Generator) -> None:
        """Padding must be distinguishable from a real zero-size, zero-delay packet."""
        pkts = np.array([[0.0, 100.0, 1.0], [0.01, 200.0, -1.0], [0.02, 150.0, 1.0]])
        seq = sequence_tensor(pkts, length=16)
        assert seq[3, :3].sum() == 3.0, "real packets must be marked valid"
        assert seq[3, 3:].sum() == 0.0, "padding must be marked invalid"
        assert np.all(seq[:3, 3:] == 0.0)

    def test_signed_size_channel_encodes_direction(self) -> None:
        pkts = np.array([[0.0, MTU, -1.0], [0.01, MTU, 1.0]])
        seq = sequence_tensor(pkts, length=4)
        assert seq[0, 0] == pytest.approx(-1.0)
        assert seq[0, 1] == pytest.approx(1.0)

    def test_long_flow_is_truncated(self, rng: np.random.Generator) -> None:
        pkts = web_browsing(rng)
        seq = sequence_tensor(pkts, length=8)
        assert seq.shape[1] == 8
        assert seq[3].sum() == 8.0

    def test_values_stay_in_a_sane_range(self, rng: np.random.Generator) -> None:
        for _ in range(20):
            seq = sequence_tensor(web_browsing(rng))
            assert np.isfinite(seq).all()
            assert np.abs(seq).max() <= 2.0


# --------------------------------------------------------------------------- #
# statistics view
# --------------------------------------------------------------------------- #
class TestStatistics:
    def test_names_match_the_dict(self, rng: np.random.Generator) -> None:
        feats = flow_statistics(web_browsing(rng))
        assert set(feats) == set(FEATURE_NAMES)
        assert len(FEATURE_NAMES) == N_FEATURES

    def test_vector_is_finite_for_every_profile(self, rng: np.random.Generator) -> None:
        from antod.data.profiles import ALL_PROFILES

        for gen in ALL_PROFILES.values():
            for _ in range(3):
                v = stats_vector(gen(rng))
                assert v.shape == (N_FEATURES,)
                assert np.isfinite(v).all()

    def test_one_directional_flow_does_not_produce_nan(self) -> None:
        """A scan or flood can be entirely upstream; the down-side stats must degrade to 0."""
        pkts = np.stack([np.arange(20) * 0.001, np.full(20, 60.0), np.ones(20)], axis=1)
        feats = flow_statistics(pkts)
        assert feats["size_down_mean"] == 0.0
        assert feats["bytes_down"] == 0.0
        assert feats["pkt_up_ratio"] == 1.0
        assert all(np.isfinite(v) for v in feats.values())

    def test_block_padding_shows_up_in_the_grid_features(self, rng: np.random.Generator) -> None:
        """This is the feature the padding detector is expected to lean on."""
        flow = web_browsing(rng)
        before = flow_statistics(flow)["size_grid_256"]
        after = flow_statistics(block_padding(flow, rng, block=256))["size_grid_256"]
        assert after > 0.95
        assert after > before

    def test_padding_collapses_size_diversity(self, rng: np.random.Generator) -> None:
        flow = web_browsing(rng)
        before = flow_statistics(flow)["unique_size_ratio"]
        after = flow_statistics(block_padding(flow, rng, block=256))["unique_size_ratio"]
        assert after <= before

    def test_constant_rate_shaping_flattens_timing_variation(
        self, rng: np.random.Generator
    ) -> None:
        flow = web_browsing(rng)
        before = flow_statistics(flow)["iat_cv"]
        after = flow_statistics(constant_rate_shaping(flow, rng, interval=0.02))["iat_cv"]
        assert after < before
        assert after < 0.5

    def test_beacon_periodicity_beats_web_periodicity(self, rng: np.random.Generator) -> None:
        """A metronome should be more self-similar than a person browsing."""
        beacon = np.mean([flow_statistics(c2_beacon(rng))["iat_up_cv"] for _ in range(30)])
        web = np.mean([flow_statistics(web_browsing(rng))["iat_up_cv"] for _ in range(30)])
        assert beacon < web


# --------------------------------------------------------------------------- #
# scaling
# --------------------------------------------------------------------------- #
class TestStatsScaler:
    def test_round_trip_is_accurate(self, dataset: FlowDataset) -> None:
        scaler = StatsScaler(clip=50.0).fit(dataset.stats)
        back = scaler.inverse_transform(scaler.transform(dataset.stats))
        rel = np.abs(back - dataset.stats) / (np.abs(dataset.stats) + 1.0)
        assert rel.max() < 1e-4

    def test_output_is_standardised(self, dataset: FlowDataset) -> None:
        scaler = StatsScaler().fit(dataset.stats)
        z = scaler.transform(dataset.stats)
        assert abs(float(z.mean())) < 1e-4
        assert abs(float(z.std()) - 1.0) < 0.2

    def test_constant_feature_does_not_divide_by_zero(self) -> None:
        x = np.zeros((10, N_FEATURES), dtype=np.float32)
        z = StatsScaler().fit_transform(x)
        assert np.isfinite(z).all()

    def test_unfitted_scaler_refuses_to_transform(self, dataset: FlowDataset) -> None:
        with pytest.raises(RuntimeError, match="fitted"):
            StatsScaler().transform(dataset.stats)

    def test_state_survives_a_round_trip(self, dataset: FlowDataset) -> None:
        a = StatsScaler().fit(dataset.stats)
        b = StatsScaler.from_state(a.state_dict())
        np.testing.assert_allclose(a.transform(dataset.stats), b.transform(dataset.stats))


# --------------------------------------------------------------------------- #
# domain constraints
# --------------------------------------------------------------------------- #
class TestFeatureConstraints:
    def test_ratios_are_clamped_to_the_unit_interval(self) -> None:
        c = FeatureConstraints()
        i = FEATURE_NAMES.index("size_grid_256")
        x = np.zeros((1, N_FEATURES), dtype=np.float32)
        x[0, i] = 1.4  # what an unconstrained gradient step would ask for
        assert c.project(x)[0, i] == pytest.approx(1.0)

    def test_counts_cannot_go_negative(self) -> None:
        c = FeatureConstraints()
        i = FEATURE_NAMES.index("n_packets")
        x = np.zeros((1, N_FEATURES), dtype=np.float32)
        x[0, i] = -5.0
        assert c.project(x)[0, i] == pytest.approx(0.0)

    def test_signed_features_are_left_alone(self) -> None:
        c = FeatureConstraints()
        i = FEATURE_NAMES.index("size_skew")
        x = np.zeros((1, N_FEATURES), dtype=np.float32)
        x[0, i] = -3.2
        assert c.project(x)[0, i] == pytest.approx(-3.2)

    def test_real_features_are_already_feasible(self, dataset: FlowDataset) -> None:
        """If genuine traffic violated the bounds, the bounds would be wrong."""
        c = FeatureConstraints()
        np.testing.assert_allclose(c.project(dataset.stats), dataset.stats, atol=1e-4)

    def test_scaled_bounds_form_a_valid_box(self, dataset: FlowDataset) -> None:
        scaler = StatsScaler().fit(dataset.stats)
        lo, hi = FeatureConstraints().torch_bounds(scaler)
        assert lo.shape == hi.shape == (N_FEATURES,)
        assert bool(lo.isfinite().all()) and bool(hi.isfinite().all())
        assert bool((lo <= hi).all())


# --------------------------------------------------------------------------- #
# splits and loaders
# --------------------------------------------------------------------------- #
class TestSplits:
    def test_partition_is_disjoint_and_complete(self, dataset: FlowDataset) -> None:
        sp = stratified_split(dataset, val_frac=0.15, test_frac=0.2, seed=1)
        assert len(sp.train) + len(sp.val) + len(sp.test) == len(dataset)
        assert len(sp.test) == pytest.approx(0.2 * len(dataset), abs=2)

    def test_class_balance_is_preserved_in_every_split(self, dataset: FlowDataset) -> None:
        sp = stratified_split(dataset, seed=1)
        overall = np.bincount(dataset.y, minlength=3) / len(dataset)
        for part in (sp.train, sp.val, sp.test):
            got = np.bincount(part.y, minlength=3) / len(part)
            np.testing.assert_allclose(got, overall, atol=0.05)

    def test_scaler_is_fitted_on_train_only(self, dataset: FlowDataset) -> None:
        """Fitting on the full set would leak test-set scale into the model."""
        sp = stratified_split(dataset, seed=1)
        train_only = StatsScaler().fit(sp.train.stats)
        np.testing.assert_allclose(sp.scaler.state_dict()["mean"], train_only.state_dict()["mean"])

    def test_loaders_yield_the_expected_batch_shapes(self, dataset: FlowDataset) -> None:
        sp = stratified_split(dataset, seed=1)
        loaders = make_loaders(sp, batch_size=16)
        seq, stats, y = next(iter(loaders["train"]))
        assert seq.shape[1:] == (SEQ_CHANNELS, SEQ_LEN)
        assert stats.shape[1] == N_FEATURES
        assert y.shape[0] == seq.shape[0] == stats.shape[0]

    def test_tensor_dataset_exposes_the_whole_split_at_once(self, dataset: FlowDataset) -> None:
        sp = stratified_split(dataset, seed=1)
        seq, stats, y = FlowTensorDataset(sp.test, sp.scaler).tensors()
        assert seq.shape[0] == stats.shape[0] == y.shape[0] == len(sp.test)


class TestPersistence:
    def test_save_and_load_round_trip(self, dataset: FlowDataset, tmp_path) -> None:
        path = tmp_path / "ds.npz"
        dataset.save(path)
        back = FlowDataset.load(path)
        np.testing.assert_allclose(back.stats, dataset.stats)
        np.testing.assert_allclose(back.seq, dataset.seq)
        np.testing.assert_array_equal(back.y, dataset.y)
        np.testing.assert_array_equal(back.recipe, dataset.recipe)

    def test_mismatched_row_counts_are_rejected(self, dataset: FlowDataset) -> None:
        with pytest.raises(ValueError, match="rows"):
            FlowDataset(
                seq=dataset.seq[:5],
                stats=dataset.stats,
                y=dataset.y,
                profile=dataset.profile,
                recipe=dataset.recipe,
            )
