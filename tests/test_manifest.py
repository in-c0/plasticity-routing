"""Manifests must be complete, machine-readable, and honestly classified."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from plasticity_routing.agent import rollout
from plasticity_routing.config import CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001
from plasticity_routing.manifest import (
    CLASSIFICATIONS, build_manifest, config_hash, source_tree_sha256, write_manifest,
)
from plasticity_routing.routers import constant_routers
from plasticity_routing.world import build_world

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_runs import validate  # noqa: E402

PASSING_AUDIT = {"passed": True, "checks": []}


def _manifest(arm="ALL_FAST", seed=11, classification="DEV_CALIBRATION", audit=PASSING_AUDIT):
    w = build_world(EXP001.world, seed=seed)
    router = constant_routers()[arm] if arm in constant_routers() else constant_routers()["ALL_FAST"]
    res = rollout(w, router, EXP001.cost, EXP001.substrate, seed=seed)
    m = build_manifest(
        root=ROOT, classification=classification, arm=arm, result=res,
        world_cfg=EXP001.world, sub_cfg=EXP001.substrate, cost_cfg=EXP001.cost,
        train_cfg=EXP001.train, dev_seeds=list(DEV_SEEDS), leakage=audit,
    )
    return m


def test_dev_and_confirmatory_seeds_are_disjoint():
    assert not (set(DEV_SEEDS) & set(CONFIRMATORY_SEEDS))
    assert len(CONFIRMATORY_SEEDS) >= 5


def test_manifest_has_every_required_field():
    m = _manifest()
    for field in ("schema", "classification", "arm", "decision_time_legal", "seed",
                  "git_sha", "source_tree_sha256", "config_hash", "environment",
                  "world_config", "substrate_config", "cost_config", "metrics",
                  "resources", "action_histogram", "diagnostics", "leakage_audit"):
        assert field in m, field
    for field in ("write_elements", "storage_element_steps", "router_compute_elements",
                  "total_compute_elements", "normalized", "penalties"):
        assert field in m["resources"], field


def test_manifest_is_json_serialisable(tmp_path):
    p = tmp_path / "m.json"
    write_manifest(p, _manifest())
    assert json.loads(p.read_text())["arm"] == "ALL_FAST"


def test_classification_is_validated():
    with pytest.raises(ValueError):
        _manifest(classification="TOTALLY_FINE_HONEST")
    assert "CONFIRMATORY" in CLASSIFICATIONS and "DEV_CALIBRATION" in CLASSIFICATIONS


def test_source_tree_hash_is_deterministic_and_content_sensitive(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    h1 = source_tree_sha256(tmp_path)
    assert h1 == source_tree_sha256(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    assert source_tree_sha256(tmp_path) != h1


def test_config_hash_distinguishes_configs():
    from dataclasses import replace
    other = replace(EXP001.world, lifetime=EXP001.world.lifetime + 1)
    assert config_hash(EXP001.world) != config_hash(other)


# ---- validator ------------------------------------------------------------


def _arm_set(seeds=(11,)):
    from plasticity_routing.routers import (
        ExtendedHeuristicRouter, HeuristicRouter, RandomMatchedRouter,
    )
    import numpy as np

    out = []
    extra = {"HEURISTIC": HeuristicRouter(),
             "HEURISTIC_EXT": ExtendedHeuristicRouter(),
             "RANDOM_MATCHED": RandomMatchedRouter(np.array([0.25] * 4)),
             "LEARNED": HeuristicRouter(),
             "SHUFFLE_TRAINED": HeuristicRouter()}
    for seed in seeds:
        w = build_world(EXP001.world, seed=seed)
        for arm in ("ALL_IGNORE", "ALL_EPISODIC", "ALL_FAST", "ALL_SLOW",
                    "HEURISTIC", "HEURISTIC_EXT", "RANDOM_MATCHED", "LEARNED",
                    "SHUFFLE_TRAINED"):
            router = constant_routers().get(arm) or extra[arm]
            res = rollout(w, router, EXP001.cost, EXP001.substrate, seed=seed)
            m = build_manifest(root=ROOT, classification="DEV_CALIBRATION", arm=arm,
                               result=res, world_cfg=EXP001.world, sub_cfg=EXP001.substrate,
                               cost_cfg=EXP001.cost, train_cfg=EXP001.train,
                               dev_seeds=list(DEV_SEEDS), leakage=PASSING_AUDIT)
            out.append(m)
    return out


def test_validator_accepts_a_complete_set():
    assert validate(_arm_set())["passed"]


def test_validator_rejects_a_missing_arm():
    ms = [m for m in _arm_set() if m["arm"] != "HEURISTIC"]
    r = validate(ms)
    assert not r["passed"]
    assert any("missing_required_arms" in x for x in r["reasons"])


def test_validator_rejects_a_failed_leakage_audit():
    ms = _arm_set()
    ms[0]["leakage_audit"] = {"passed": False, "checks": []}
    r = validate(ms)
    assert not r["passed"]
    assert any("leakage_audit_failed" in x for x in r["reasons"])


def test_validator_rejects_a_missing_leakage_audit():
    ms = _arm_set()
    ms[0]["leakage_audit"] = None
    assert any("missing_leakage_audit" in x for x in validate(ms)["reasons"])


def test_validator_rejects_a_privileged_claim_arm():
    ms = _arm_set()
    ms[0]["decision_time_legal"] = False
    r = validate(ms)
    assert any("privileged_arm_among_claim_arms" in x for x in r["reasons"])


def test_validator_rejects_mismatched_source_trees():
    ms = _arm_set()
    ms[0]["source_tree_sha256"] = "deadbeef"
    assert any("different_source_trees" in x for x in validate(ms)["reasons"])


def test_validator_rejects_a_confirmatory_run_on_a_dev_seed():
    ms = _arm_set()
    for m in ms:
        m["classification"] = "CONFIRMATORY"
    r = validate(ms)
    assert any("confirmatory_seed_overlaps_dev_seed" in x for x in r["reasons"])


def test_validator_rejects_a_breached_write_ceiling():
    ms = _arm_set()
    ms[0]["resources"]["write_ceiling_exceeded"] = True
    assert any("write_ceiling_exceeded" in x for x in validate(ms)["reasons"])


def test_validator_rejects_differing_write_ceilings():
    ms = _arm_set()
    ms[0]["cost_config"] = dict(ms[0]["cost_config"], write_element_ceiling=1)
    assert any("different_write_ceilings" in x for x in validate(ms)["reasons"])
