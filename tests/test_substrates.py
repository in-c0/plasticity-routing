"""Each substrate must actually have the character the design claims."""

from __future__ import annotations

import numpy as np
import pytest

from plasticity_routing.substrates import (
    EPISODIC, FAST, IGNORE, SLOW, AssociativeMatrix, EpisodicStore, SubstrateBank,
    SubstrateConfig, utility,
)


def _unit(rng, d):
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def test_utility_bounds():
    rng = np.random.default_rng(0)
    v = _unit(rng, 8)
    assert utility(v, v) == pytest.approx(1.0)
    assert utility(-v, v) == 0.0
    assert utility(np.zeros(8), v) == 0.0


def test_episodic_is_exact():
    rng = np.random.default_rng(0)
    st = EpisodicStore(4)
    v = _unit(rng, 8)
    st.write(1, v, t=0)
    assert np.allclose(st.read(1, t=1), v)


def test_episodic_evicts_least_recently_used():
    rng = np.random.default_rng(0)
    st = EpisodicStore(2)
    st.write(1, _unit(rng, 4), t=0)
    st.write(2, _unit(rng, 4), t=1)
    st.read(1, t=2)                    # refresh key 1
    st.write(3, _unit(rng, 4), t=3)    # must evict key 2
    assert st.read(1, t=4) is not None
    assert st.read(2, t=4) is None
    assert st.evictions == 1


def test_fast_decays_and_slow_does_not():
    rng = np.random.default_rng(0)
    k, v = _unit(rng, 16), _unit(rng, 8)
    fast = AssociativeMatrix(16, 8, lr=1.0, decay=0.99)
    slow = AssociativeMatrix(16, 8, lr=1.0, decay=1.0)
    fast.write(k, v)
    slow.write(k, v)
    for _ in range(400):
        fast.step()
        slow.step()
    assert np.linalg.norm(fast.read(k)) < 0.1 * np.linalg.norm(v)
    assert np.linalg.norm(slow.read(k)) == pytest.approx(np.linalg.norm(v), rel=1e-6)


def test_delta_rule_is_one_shot_exact_at_unit_lr():
    rng = np.random.default_rng(0)
    k, v = _unit(rng, 16), _unit(rng, 8)
    m = AssociativeMatrix(16, 8, lr=1.0, decay=1.0)
    m.write(k, v)
    assert np.allclose(m.read(k), v, atol=1e-9)


def test_parametric_writes_interfere():
    """Interference is the price of a fixed-footprint substrate. It must be real."""
    rng = np.random.default_rng(1)
    m = AssociativeMatrix(16, 8, lr=1.0, decay=1.0)
    k0, v0 = _unit(rng, 16), _unit(rng, 8)
    m.write(k0, v0)
    before = utility(m.read(k0), v0)
    for _ in range(40):
        m.write(_unit(rng, 16), _unit(rng, 8))
    after = utility(m.read(k0), v0)
    assert before > 0.99
    assert after < before - 0.05, "overloading the matrix must degrade earlier associations"


def test_ignore_changes_nothing():
    cfg = SubstrateConfig(key_dim=16, value_dim=8, episodic_capacity=4)
    bank = SubstrateBank(cfg)
    rng = np.random.default_rng(0)
    k, v = _unit(rng, 16), _unit(rng, 8)
    bank.apply(IGNORE, 1, k, v, t=0)
    assert len(bank.episodic) == 0
    assert bank.fast.norm == 0.0 and bank.slow.norm == 0.0


def test_probe_has_no_side_effects():
    """L7. If probes refreshed LRU order the evaluator would change the result."""
    cfg = SubstrateConfig(key_dim=16, value_dim=8, episodic_capacity=2)
    bank = SubstrateBank(cfg)
    rng = np.random.default_rng(0)
    ks = [_unit(rng, 16) for _ in range(3)]
    vs = [_unit(rng, 8) for _ in range(3)]
    bank.apply(EPISODIC, 0, ks[0], vs[0], t=0)
    bank.apply(EPISODIC, 1, ks[1], vs[1], t=1)

    for _ in range(5):
        bank.probe(0, ks[0])           # would refresh key 0 if it used recall()
    touch_before = dict(bank.episodic.last_touch)
    assert touch_before == {0: 0, 1: 1}

    bank.apply(EPISODIC, 2, ks[2], vs[2], t=2)   # evicts the true LRU, key 0
    assert bank.episodic.read(0, t=3) is None
    assert bank.episodic.read(1, t=3) is not None


def test_slow_write_runs_consolidation_steps():
    cfg = SubstrateConfig(key_dim=16, value_dim=8, slow_lr=0.3, slow_consolidation_steps=4)
    bank = SubstrateBank(cfg)
    rng = np.random.default_rng(0)
    k, v = _unit(rng, 16), _unit(rng, 8)
    bank.apply(SLOW, 1, k, v, t=0)
    single = AssociativeMatrix(16, 8, lr=0.3, decay=1.0)
    single.write(k, v)
    assert utility(bank.slow.read(k), v) >= utility(single.read(k), v)
    assert np.linalg.norm(bank.slow.read(k)) > np.linalg.norm(single.read(k))


def test_fast_does_not_touch_slow():
    cfg = SubstrateConfig(key_dim=16, value_dim=8)
    bank = SubstrateBank(cfg)
    rng = np.random.default_rng(0)
    bank.apply(FAST, 1, _unit(rng, 16), _unit(rng, 8), t=0)
    assert bank.slow.norm == 0.0
    assert len(bank.episodic) == 0
