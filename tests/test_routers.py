"""Router contracts."""

from __future__ import annotations

import numpy as np
import pytest

from plasticity_routing.features import FEATURE_NAMES
from plasticity_routing.routers import (
    INTENDED_MAPPING, HeuristicRouter, LearnedRouter, OracleRouter, PrivilegedTaskIdRouter,
    RandomMatchedRouter, constant_routers, is_bijective,
)
from plasticity_routing.substrates import ACTIONS, EPISODIC, FAST, IGNORE, N_ACTIONS, SLOW

NF = len(FEATURE_NAMES)


def test_constant_routers_cover_every_action():
    actions = {r.act(np.zeros(NF), np.random.default_rng(0)) for r in constant_routers().values()}
    assert actions == set(ACTIONS)


def test_all_constant_routers_are_legal_and_free():
    for r in constant_routers().values():
        assert r.legal and r.privileged_fields == () and r.n_params == 0


def test_heuristic_is_legal_but_not_free():
    h = HeuristicRouter()
    assert h.legal and h.privileged_fields == ()
    assert h.n_params > 0, "a non-trivial router must pay decision-time compute"


def test_heuristic_returns_valid_actions_over_random_features():
    h = HeuristicRouter()
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = h.act(rng.random(NF), rng)
        assert a in ACTIONS


def test_learned_router_outputs_a_distribution():
    lr = LearnedRouter(seed=0)
    p = lr.probs(np.random.default_rng(0).random(NF))
    assert p.shape == (N_ACTIONS,)
    assert p.sum() == pytest.approx(1.0)
    assert (p > 0).all()


def test_learned_router_is_deterministic_in_greedy_mode():
    lr = LearnedRouter(seed=0)
    lr.greedy = True
    x = np.random.default_rng(1).random(NF)
    a = {lr.act(x, np.random.default_rng(s)) for s in range(20)}
    assert len(a) == 1


def test_learned_router_charges_its_real_parameter_count():
    lr = LearnedRouter(hidden=16, seed=0)
    assert lr.n_params == lr.p.size == 16 * NF + 16 + N_ACTIONS * 16 + N_ACTIONS


def test_random_matched_reproduces_target_distribution():
    target = np.array([0.4, 0.3, 0.2, 0.1])
    r = RandomMatchedRouter(target)
    rng = np.random.default_rng(0)
    counts = np.zeros(N_ACTIONS)
    for _ in range(40000):
        counts[r.act(np.zeros(NF), rng)] += 1
    assert np.allclose(counts / counts.sum(), target, atol=0.01)


def test_oracle_applies_its_mapping():
    o = OracleRouter((SLOW, IGNORE, EPISODIC, FAST))
    rng = np.random.default_rng(0)
    for cls, expected in enumerate((SLOW, IGNORE, EPISODIC, FAST)):
        assert o.act(np.zeros(NF), rng, {"hidden_class": cls}) == expected


def test_privileged_routers_are_marked_illegal():
    assert OracleRouter().legal is False
    assert PrivilegedTaskIdRouter(HeuristicRouter()).legal is False


def test_bijectivity_predicate():
    assert is_bijective((0, 1, 2, 3))
    assert is_bijective(INTENDED_MAPPING)
    assert not is_bijective((0, 0, 1, 3))
    assert not is_bijective((2, 2, 2, 2))


def test_gradient_moves_probability_toward_a_rewarded_action():
    """A minimal sanity check that REINFORCE has the right sign."""
    lr = LearnedRouter(hidden=8, seed=0)
    lr.record = True
    x = np.random.default_rng(0).random(NF)
    rng = np.random.default_rng(0)
    lr.trace = []
    for _ in range(64):
        lr.trace.append((x, np.tanh(lr.p.W1 @ x + lr.p.b1), SLOW))
    before = lr.probs(x)[SLOW]
    g = lr.grad(np.ones(64), entropy_beta=0.0)
    lr.apply_grad(g, lr=0.05, state={})
    assert lr.probs(x)[SLOW] > before


def test_heuristic_thresholds_are_all_live():
    """Every declared hyperparameter must change behaviour on some input.

    A dead knob silently shrinks the calibration grid, which under-tunes the
    primary comparator -- the exact direction that would flatter the method
    under test.
    """
    rng = np.random.default_rng(0)
    feats = [rng.random(NF) for _ in range(400)]

    def actions(**kw):
        h = HeuristicRouter(**kw)
        return [h.act(x, rng) for x in feats]

    base = actions()
    assert actions(seen_threshold=3) != base
    assert actions(revision_tolerance=0.2) != base
    assert actions(error_floor=0.9) != base
