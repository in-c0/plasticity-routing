"""Mechanical preconditions for the L5b cross-world negative control.

These must hold *before* the audit is interpreted. They check the things that
would silently invalidate the comparison: a policy artefact drifting after
Amendment L pinned it, an audit seed leaking into a training or confirmatory
set, the two arms not actually being identically specified, or the four cells
running under different resource settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from plasticity_routing.agent import rollout
from plasticity_routing.config import (
    AUDIT_SEEDS, CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001, SELECTED_POLICY_SHA256,
)
from plasticity_routing.features import FEATURE_NAMES
from plasticity_routing.train import load_policy
from plasticity_routing.world import build_world, time_shuffled_world

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "results/policies"


def _policies_present() -> bool:
    return all((POLICY_DIR / f"{w}_selected.json").exists() for w in ("real", "shuffled"))


requires_policies = pytest.mark.skipif(
    not _policies_present(), reason="frozen policy artefacts not present"
)


# ---- seed hygiene ---------------------------------------------------------


def test_audit_seeds_are_disjoint_from_dev_and_confirmatory():
    audit = set(AUDIT_SEEDS)
    assert not (audit & set(DEV_SEEDS)), "audit seeds overlap the training seeds"
    assert not (audit & set(CONFIRMATORY_SEEDS)), "audit seeds overlap the held-out seeds"


def test_audit_seed_set_is_the_preregistered_one():
    assert AUDIT_SEEDS == tuple(range(91001, 91033))
    assert len(set(AUDIT_SEEDS)) == 32


# ---- policy immutability --------------------------------------------------


@requires_policies
@pytest.mark.parametrize("name", ["real_selected.json", "shuffled_selected.json"])
def test_selected_policy_unchanged_since_amendment_L(name):
    """The audit compares two frozen artefacts. If either changed, the
    preregistered comparison is not the one being run."""
    path = POLICY_DIR / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == SELECTED_POLICY_SHA256[name], (
        f"{name} changed after Amendment L pinned it:\n"
        f"  pinned {SELECTED_POLICY_SHA256[name]}\n  actual {actual}"
    )


@requires_policies
def test_pinned_hashes_cover_exactly_the_two_audit_policies():
    assert set(SELECTED_POLICY_SHA256) == {"real_selected.json", "shuffled_selected.json"}


# ---- identical specification ---------------------------------------------


@requires_policies
def test_both_policies_are_identically_specified_and_legal():
    R, _ = load_policy(POLICY_DIR / "real_selected.json")
    S, _ = load_policy(POLICY_DIR / "shuffled_selected.json")
    assert R.p.W1.shape == S.p.W1.shape
    assert R.p.W2.shape == S.p.W2.shape
    assert R.n_params == S.n_params
    assert R.temperature == S.temperature
    for pol in (R, S):
        assert pol.legal is True
        assert pol.privileged_fields == ()
        assert pol.p.W1.shape[1] == len(FEATURE_NAMES)


@requires_policies
def test_both_policies_were_trained_under_the_same_procedure():
    _, mR = load_policy(POLICY_DIR / "real_selected.json")
    _, mS = load_policy(POLICY_DIR / "shuffled_selected.json")
    assert mR["generations"] == mS["generations"]
    assert mR["dev_seeds"] == mS["dev_seeds"]
    assert mR["es_config"] == mS["es_config"]
    assert mR["world"] == "real" and mS["world"] == "shuffled"


@requires_policies
def test_both_policies_receive_the_identical_feature_whitelist(monkeypatch):
    """Neither arm may see anything the other does not: same whitelist, same
    vector width, and `privileged=None` on every decision."""
    from plasticity_routing import features as F

    world = build_world(EXP001.world, seed=AUDIT_SEEDS[0])
    seen: dict[str, list] = {}
    for label, name in (("R", "real_selected.json"), ("S", "shuffled_selected.json")):
        pol, _ = load_policy(POLICY_DIR / name)
        pol.greedy = True
        widths, privs = [], []
        original = pol.act

        def spy(feats, rng, privileged=None, _o=original, _w=widths, _p=privs):
            _w.append(len(feats))
            _p.append(privileged)
            return _o(feats, rng, privileged)

        monkeypatch.setattr(pol, "act", spy)
        rollout(world, pol, EXP001.cost, EXP001.substrate, seed=AUDIT_SEEDS[0])
        assert widths and set(widths) == {len(F.FEATURE_NAMES)}
        assert all(p is None for p in privs), f"{label} was handed privileged data"
        seen[label] = widths

    assert len(seen["R"]) == len(seen["S"]), "arms saw different numbers of decisions"


# ---- identical resource settings across all four cells --------------------


@requires_policies
def test_all_four_cells_use_identical_cost_and_substrate_settings():
    """Every cell must be scored under one cost table and one substrate config;
    otherwise a cross-world difference could be a budget difference."""
    R, _ = load_policy(POLICY_DIR / "real_selected.json")
    S, _ = load_policy(POLICY_DIR / "shuffled_selected.json")
    R.greedy = S.greedy = True

    seed = AUDIT_SEEDS[0]
    real = build_world(EXP001.world, seed=seed)
    shuf = time_shuffled_world(real, seed=seed)

    results = {
        "J_RR": rollout(real, R, EXP001.cost, EXP001.substrate, seed=seed),
        "J_SR": rollout(real, S, EXP001.cost, EXP001.substrate, seed=seed),
        "J_RS": rollout(shuf, R, EXP001.cost, EXP001.substrate, seed=seed),
        "J_SS": rollout(shuf, S, EXP001.cost, EXP001.substrate, seed=seed),
    }
    cost_tables = {json.dumps(r.ledger["cost_config"], sort_keys=True) for r in results.values()}
    assert len(cost_tables) == 1, "cells ran under different cost tables"

    ceiling = EXP001.cost.write_element_ceiling
    for k, r in results.items():
        assert r.ledger["write_elements"] <= ceiling, f"{k} exceeded the write ceiling"


def test_shuffle_preserves_writes_and_query_timing_on_an_audit_seed():
    """The shuffled world must differ only in which key each query targets."""
    seed = AUDIT_SEEDS[0]
    real = build_world(EXP001.world, seed=seed)
    shuf = time_shuffled_world(real, seed=seed)
    rw = [(e.t, e.key_id, e.value.tobytes()) for e in real.events if e.kind == "WRITE"]
    sw = [(e.t, e.key_id, e.value.tobytes()) for e in shuf.events if e.kind == "WRITE"]
    assert rw == sw
    assert ([e.t for e in real.events if e.kind == "QUERY"]
            == [e.t for e in shuf.events if e.kind == "QUERY"])


# ---- the statistic itself -------------------------------------------------


def test_interaction_identity_used_by_the_bootstrap():
    """I = (J_RR-J_SR)-(J_RS-J_SS) = (J_RR-J_RS)-(J_SR-J_SS).

    The audit bootstraps the second form so the paired estimator applies; the
    two must be algebraically identical.
    """
    rng = np.random.default_rng(0)
    rr, sr, rs, ss = (rng.normal(size=50) for _ in range(4))
    lhs = (rr - sr) - (rs - ss)
    rhs = (rr - rs) - (sr - ss)
    assert np.allclose(lhs, rhs)


def test_audit_refuses_unpinned_policies(tmp_path):
    """The audit must not silently run against a drifted artefact."""
    import subprocess
    import sys

    fake = tmp_path / "policies"
    fake.mkdir()
    for name in ("real_selected.json", "shuffled_selected.json"):
        src = json.loads((POLICY_DIR / name).read_text()) if (POLICY_DIR / name).exists() else {
            "schema": "plasticity-routing/policy/1", "hidden": 16, "temperature": 1.0,
            "n_params": 1, "params": [0.0], "meta": {},
        }
        src["meta"] = dict(src.get("meta", {}), tampered=True)
        (fake / name).write_text(json.dumps(src))

    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/audit_l5b.py"), "--policy-dir", str(fake),
         "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0
    assert "pinned by Amendment L" in (proc.stdout + proc.stderr)
