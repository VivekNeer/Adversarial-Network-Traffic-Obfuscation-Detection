"""Invariants that every obfuscation transform must respect.

These are not cosmetic tests. If a transform can reorder time, emit an illegal
packet size, or quietly change the byte volume, then the "obfuscated" class in the
dataset is partly an artefact of the generator rather than a model of evasion, and
every accuracy number downstream becomes uninterpretable.
"""

from __future__ import annotations

import numpy as np
import pytest

from antod.data.obfuscation import (
    MAX_PACKETS,
    TRANSFORMS,
    block_padding,
    constant_rate_shaping,
    fragmentation,
    protocol_mimicry,
    sample_recipe,
    single_recipe,
    timing_jitter,
    tunnel_encapsulation,
)
from antod.data.profiles import ALL_PROFILES, MIN_PKT, MTU
from antod.data.synth import (
    BENIGN,
    LABEL_NAMES,
    MALICIOUS_OBFUSCATED,
    SynthConfig,
    generate_dataset,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def flow(rng: np.random.Generator) -> np.ndarray:
    from antod.data.profiles import web_browsing

    return web_browsing(rng)


def _assert_valid(pkts: np.ndarray) -> None:
    assert pkts.ndim == 2 and pkts.shape[1] == 3
    assert pkts.shape[0] >= 1
    assert np.all(np.isfinite(pkts))
    assert np.all(np.diff(pkts[:, 0]) >= -1e-9), "timestamps must be non-decreasing"
    assert pkts[0, 0] == pytest.approx(0.0), "flow time must be relative to its first packet"
    assert pkts[:, 1].min() >= MIN_PKT
    assert pkts[:, 1].max() <= MTU
    assert set(np.unique(pkts[:, 2])) <= {-1.0, 1.0}


@pytest.mark.parametrize("name", sorted(ALL_PROFILES))
def test_profiles_emit_valid_packets(name: str, rng: np.random.Generator) -> None:
    for _ in range(5):
        _assert_valid(ALL_PROFILES[name](rng))


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transforms_preserve_packet_validity(
    name: str, flow: np.ndarray, rng: np.random.Generator
) -> None:
    out = single_recipe(name, rng).apply(flow, rng)
    _assert_valid(out)


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transforms_stay_under_packet_cap(
    name: str, flow: np.ndarray, rng: np.random.Generator
) -> None:
    out = single_recipe(name, rng).apply(flow, rng)
    assert out.shape[0] <= MAX_PACKETS


def test_block_padding_quantises_sizes(flow: np.ndarray, rng: np.random.Generator) -> None:
    out = block_padding(flow, rng, block=256)
    # every packet is either on a 256-byte boundary or clamped at the MTU
    on_grid = (out[:, 1] % 256 == 0) | (out[:, 1] == MTU)
    assert on_grid.all()
    assert np.all(out[:, 1] >= flow[:, 1]), "padding may only grow a packet"
    assert out.shape[0] == flow.shape[0], "padding must not change the packet count"


def test_timing_jitter_keeps_sizes_and_order(flow: np.ndarray, rng: np.random.Generator) -> None:
    out = timing_jitter(flow, rng, scale=0.05)
    assert out.shape == flow.shape
    np.testing.assert_allclose(out[:, 1], flow[:, 1])
    np.testing.assert_allclose(out[:, 2], flow[:, 2])
    # jitter may only delay: the flow can never finish sooner than it did
    assert out[-1, 0] >= flow[-1, 0] - 1e-9


def test_fragmentation_preserves_byte_volume(flow: np.ndarray, rng: np.random.Generator) -> None:
    out = fragmentation(flow, rng, max_size=300)
    assert out.shape[0] >= flow.shape[0]
    assert out[:, 1].max() <= 300 or out[:, 1].max() <= MIN_PKT
    # fragments carry the same payload, modulo the MIN_PKT floor on the last shard
    assert out[:, 1].sum() >= flow[:, 1].sum() * 0.99


def test_constant_rate_shaping_produces_a_fixed_cadence(
    flow: np.ndarray, rng: np.random.Generator
) -> None:
    out = constant_rate_shaping(flow, rng, interval=0.02, cell_size=512)
    _assert_valid(out)
    for direction in (1.0, -1.0):
        sel = out[out[:, 2] == direction]
        if sel.shape[0] < 3:
            continue
        gaps = np.diff(sel[:, 0])
        # every gap is the slot interval (or a multiple, where the other direction interleaves)
        assert np.allclose(gaps % 0.02, 0.0, atol=1e-6) or np.allclose(gaps, 0.02, atol=1e-6)
        assert sel[:, 1].min() >= 512 or sel[:, 1].min() >= MIN_PKT


def test_protocol_mimicry_preserves_directional_byte_volume(
    flow: np.ndarray, rng: np.random.Generator
) -> None:
    """The volume constraint is what keeps the detection problem well posed."""
    out = protocol_mimicry(flow, rng, template="video_streaming")
    _assert_valid(out)
    for direction in (1.0, -1.0):
        before = flow[flow[:, 2] == direction, 1].sum()
        after = out[out[:, 2] == direction, 1].sum()
        if before == 0:
            continue
        assert after == pytest.approx(before, rel=0.05), (
            f"direction {direction}: {before} bytes became {after}"
        )


def test_tunnel_encapsulation_adds_overhead_and_handshake(
    flow: np.ndarray, rng: np.random.Generator
) -> None:
    out = tunnel_encapsulation(flow, rng, overhead=48, handshake=True)
    _assert_valid(out)
    assert out.shape[0] >= flow.shape[0] + 4, "handshake packets should be present"
    assert out[:, 1].sum() > flow[:, 1].sum(), "encapsulation must cost bytes"


def test_sample_recipe_is_ordered_and_reproducible() -> None:
    a = sample_recipe(np.random.default_rng(11), 3, 3)
    b = sample_recipe(np.random.default_rng(11), 3, 3)
    assert a.names == b.names
    assert a.describe() == b.describe()
    assert len(a.names) == 3
    assert len(set(a.names)) == 3, "a recipe should not repeat a transform"


class TestDataset:
    def test_class_balance_and_labels(self) -> None:
        cfg = SynthConfig(n_flows=300, seed=5)
        flows = generate_dataset(cfg)
        assert len(flows) == 300
        labels = np.array([f.label for f in flows])
        assert set(np.unique(labels)) == {0, 1, 2}
        for i, w in enumerate(cfg.class_weights):
            assert (labels == i).mean() == pytest.approx(w, abs=0.02)

    def test_benign_class_contains_obfuscated_flows(self) -> None:
        """Otherwise the model learns 'obfuscated implies malicious' and fails on VPNs."""
        cfg = SynthConfig(n_flows=300, seed=5, benign_obfuscation_rate=0.35)
        flows = generate_dataset(cfg)
        benign = [f for f in flows if f.label == BENIGN]
        obf = [f for f in benign if f.obfuscated]
        assert 0.2 < len(obf) / len(benign) < 0.5

    def test_malicious_obfuscated_always_carries_a_recipe(self) -> None:
        flows = generate_dataset(SynthConfig(n_flows=300, seed=5))
        for f in flows:
            if f.label == MALICIOUS_OBFUSCATED:
                assert f.obfuscated and f.recipe != "none"
            if f.label == 1:
                assert not f.obfuscated and f.recipe == "none"

    def test_generation_is_deterministic_under_seed(self) -> None:
        a = generate_dataset(SynthConfig(n_flows=100, seed=9))
        b = generate_dataset(SynthConfig(n_flows=100, seed=9))
        assert [f.recipe for f in a] == [f.recipe for f in b]
        np.testing.assert_allclose(a[0].packets, b[0].packets)

    def test_min_packets_is_respected(self) -> None:
        flows = generate_dataset(SynthConfig(n_flows=200, seed=2, min_packets=12))
        assert min(f.n_packets for f in flows) >= 12

    def test_label_names_match_class_count(self) -> None:
        assert len(LABEL_NAMES) == 3
