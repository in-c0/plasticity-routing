"""Leakage tests L1-L7. See docs/LEAKAGE.md.

These gate every run. A failure invalidates the run; it is not a warning.
"""

from __future__ import annotations

import copy
import inspect
from dataclasses import replace

import numpy as np
import pytest

from plasticity_routing import features as F
from plasticity_routing.agent import rollout
from plasticity_routing.ledger import CostConfig
from plasticity_routing.metrics import first_encounter_class_dependence
from plasticity_routing.routers import (
    HeuristicRouter, LearnedRouter, OracleRouter, PrivilegedTaskIdRouter, constant_routers,
)
from plasticity_routing.substrates import SubstrateBank, SubstrateConfig
from plasticity_routing.world import HIDDEN_FIELDS, WorldConfig, build_world


@pytest.fixture(scope="module")
def cfgs():
    return WorldConfig(), SubstrateConfig(), CostConfig()


def legal_routers(seed: int = 0):
    rs = list(constant_routers().values())
    rs.append(HeuristicRouter())
    lr = LearnedRouter(seed=seed)
    lr.greedy = True
    rs.append(lr)
    return rs


# ---------------------------------------------------------------- L1


def test_L1_whitelist_excludes_every_forbidden_source():
    for bad in F.FORBIDDEN_SOURCES:
        assert bad not in F.FEATURE_NAMES


def test_L1_feature_vector_matches_whitelist_length(cfgs):
    wcfg, scfg, _ = cfgs
    w = build_world(wcfg, seed=1)
    e = next(x for x in w.events if x.kind == "WRITE")
    feats = F.extract(
        key_id=e.key_id, key=e.key, value=e.value, t=e.t,
        obs=F.ObserverState(lifetime=wcfg.lifetime), bank=SubstrateBank(scfg),
        write_budget_remaining=1.0,
    )
    assert feats.shape == (len(F.FEATURE_NAMES),)
    assert np.isfinite(feats).all()


def test_L1_extract_is_keyword_only_and_takes_no_event():
    """An `Event` carries hidden fields, so it must not be passable at all."""
    sig = inspect.signature(F.extract)
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values())
    assert "event" not in sig.parameters
    for hidden in HIDDEN_FIELDS:
        assert hidden not in sig.parameters


def test_L1_observer_state_stores_no_hidden_fields():
    tr = F.KeyTrace()
    for hidden in HIDDEN_FIELDS:
        assert not hasattr(tr, hidden)


# ---------------------------------------------------------------- L2


@pytest.mark.parametrize("router_idx", range(6))
def test_L2_legal_routers_are_invariant_to_permuted_hidden_labels(cfgs, router_idx):
    """Permute hidden class labels, leave the observable stream identical."""
    wcfg, scfg, ccfg = cfgs
    base = build_world(wcfg, seed=2)

    perm = build_world(wcfg, seed=2)
    mapping = {0: 2, 1: 3, 2: 0, 3: 1}
    for e in perm.events:
        e.hidden_class = mapping[e.hidden_class]
        e.ideal_action = mapping[e.ideal_action]

    r1 = legal_routers()[router_idx]
    r2 = copy.deepcopy(r1)
    a = rollout(base, r1, ccfg, scfg, seed=2, keep_decisions=True)
    b = rollout(perm, r2, ccfg, scfg, seed=2, keep_decisions=True)

    assert [d.action for d in a.decisions] == [d.action for d in b.decisions]
    assert a.task_utility == pytest.approx(b.task_utility)


# ---------------------------------------------------------------- L3


def test_L3_first_encounter_actions_are_independent_of_hidden_class(cfgs):
    """The information-theoretic test: a legal router cannot know an
    indistinguishable class. Runs on an untrained policy and the heuristic."""
    wcfg, scfg, ccfg = cfgs
    w = build_world(wcfg, seed=4)
    for router in (HeuristicRouter(), LearnedRouter(seed=1)):
        res = rollout(w, router, ccfg, scfg, seed=4)
        stat = first_encounter_class_dependence(res.first_encounter_actions)
        assert stat["n"] > 50
        assert stat["p_value"] > 0.05, f"{router.name} leaks on first encounters: {stat}"


def test_L3_detects_a_deliberate_leak(cfgs):
    """The test must be able to fail. A router that reads the hidden class on
    first encounters must be caught."""
    wcfg, scfg, ccfg = cfgs
    w = build_world(wcfg, seed=4)
    res = rollout(w, OracleRouter(), ccfg, scfg, seed=4)
    stat = first_encounter_class_dependence(res.first_encounter_actions)
    assert stat["p_value"] < 0.05, "L3 failed to detect an outright oracle leak"
    assert stat["mutual_information_bits"] > stat["null_p95_bits"]


# ---------------------------------------------------------------- L4


def test_L4_decisions_depend_only_on_the_prefix(cfgs):
    """Truncating the future must not change any past decision."""
    wcfg, scfg, ccfg = cfgs
    full = build_world(wcfg, seed=6)
    cut_t = wcfg.lifetime // 2

    truncated = build_world(wcfg, seed=6)
    truncated.events = [e for e in truncated.events if e.t < cut_t]

    for router in legal_routers(seed=3):
        r2 = copy.deepcopy(router)
        a = rollout(full, router, ccfg, scfg, seed=6, keep_decisions=True)
        b = rollout(truncated, r2, ccfg, scfg, seed=6, keep_decisions=True)
        a_pref = [(d.t, d.action) for d in a.decisions if d.t < cut_t]
        b_pref = [(d.t, d.action) for d in b.decisions if d.t < cut_t]
        assert a_pref == b_pref, f"{router.name} decisions depend on the future"


# ---------------------------------------------------------------- L6


def test_L6_legal_routers_never_receive_privileged_data(cfgs, monkeypatch):
    wcfg, scfg, ccfg = cfgs
    w = build_world(wcfg, seed=8)

    for router in legal_routers(seed=4):
        assert router.privileged_fields == ()
        seen = []
        orig = router.act

        def spy(feats, rng, privileged=None, _orig=orig, _seen=seen):
            _seen.append(privileged)
            return _orig(feats, rng, privileged)

        monkeypatch.setattr(router, "act", spy)
        rollout(w, router, ccfg, scfg, seed=8)
        assert seen and all(p is None for p in seen), f"{router.name} was handed privileged data"


def test_L6_privileged_routers_declare_and_require_their_fields():
    oracle = OracleRouter()
    assert oracle.legal is False and oracle.privileged_fields == ("hidden_class",)
    with pytest.raises(RuntimeError):
        oracle.act(np.zeros(len(F.FEATURE_NAMES)), np.random.default_rng(0), None)

    priv = PrivilegedTaskIdRouter(HeuristicRouter())
    assert priv.legal is False and priv.privileged_fields == ("regime_id",)
    with pytest.raises(RuntimeError):
        priv.act(np.zeros(len(F.FEATURE_NAMES)), np.random.default_rng(0), None)


# ---------------------------------------------------------------- L7


def test_L7_audit_probes_do_not_change_the_run(cfgs):
    """Auditing must not perturb the system it measures."""
    wcfg, scfg, ccfg = cfgs
    w = build_world(replace(wcfg, lifetime=1200), seed=9)
    for router in legal_routers(seed=5):
        r2 = copy.deepcopy(router)
        audited = rollout(w, router, ccfg, scfg, seed=9, audit_every=50, keep_decisions=True)
        clean = rollout(w, r2, ccfg, scfg, seed=9, audit_every=0, keep_decisions=True)
        assert [d.action for d in audited.decisions] == [d.action for d in clean.decisions]
        assert audited.task_utility == pytest.approx(clean.task_utility)
        assert audited.ledger["write_elements"] == clean.ledger["write_elements"]
        assert audited.ledger["read_compute_elements"] == clean.ledger["read_compute_elements"]
        assert audited.episodic_evictions == clean.episodic_evictions
