"""Refusal paths for the EXP-002 Stage-1 freeze (Amendment O-1).

The freeze exists so that "smallest sufficient capacity" is a selection rule
rather than a tuning knob. These tests drive each way a configuration could be
incomplete or internally inconsistent and assert the validator refuses.

No model, no task, no training loop is exercised here -- the module under test
contains none.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from plasticity_routing.config import EXP002_DEV_SEEDS
from plasticity_routing.exp002_stage1 import (
    REQUIRED_FIELDS, TRANSIENCE_MECHANISMS, blank_stage1_freeze, is_frozen,
    validate_stage1_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def complete_freeze() -> dict:
    """A minimal internally-consistent configuration.

    Deliberately *not* the proposed EXP-002 values: those are owner decisions
    that are not frozen yet. This is a fixture proving the validator accepts a
    coherent configuration, not a preregistration.
    """
    return {
        "backbone": {
            "repo_id": "fixture/model", "revision_sha": "0" * 40,
            "tokenizer_revision": "0" * 40,
            "tokenizer_asset_sha256": {"tokenizer.json": "a" * 64},
            "dtype": "float32", "device": "cpu",
        },
        "task": {
            "generator_module": "fixture.generator", "generator_source_sha256": "b" * 64,
            "n_bindings": 6, "presentations_per_binding": 8,
            "train_template_ids": [0, 1, 2, 3, 4, 5], "eval_template_ids": [6, 7],
            "template_set": list(range(8)), "seeds": list(EXP002_DEV_SEEDS),
        },
        "candidates": {
            "families": ["low_rank_delta"],
            "capacity_grid": {"low_rank_delta": [1, 2, 4]},
            "learning_rate_grid": [3e-4, 1e-3, 3e-3],
            "batch_size_grid": [1, 2, 4],
        },
        "budgets": {
            "update_budget_steps": 48, "token_entitlement": 10080,
            "write_element_ceiling": 1_000_000,
        },
        "mechanisms": {"fast_transience_mechanism": "interference"},
        "interference": {
            "sets": [
                {"name": "A", "members": [1, 2, 3]},
                {"name": "B", "members": [4, 5, 6]},
            ],
            "lifetime_horizon": 3000,
        },
        "selection": {"rule": "smallest sufficient capacity", "tie_breaks": ["worst_seed", "lr"]},
        "accelerator": {"equivalence_report_sha256": "c" * 64, "equivalence_passed": True},
        "provenance": {"git_sha": "d" * 40, "source_tree_sha256": "e" * 64,
                       "config_hash": "f" * 16},
    }


def test_the_fixture_is_actually_freezable():
    assert validate_stage1_freeze(complete_freeze()) == []
    assert is_frozen(complete_freeze())


def test_blank_skeleton_refuses_on_every_required_field():
    blockers = validate_stage1_freeze(blank_stage1_freeze())
    n_required = sum(len(v) for v in REQUIRED_FIELDS.values())
    assert len(blockers) == n_required


@pytest.mark.parametrize(
    "group,field",
    [(g, f) for g, fields in REQUIRED_FIELDS.items() for f in fields],
)
def test_each_required_field_independently_causes_refusal(group, field):
    """Every declared field must actually be enforced.

    Parametrised rather than spot-checked: a field listed as required but never
    read would otherwise be silently optional.
    """
    cfg = complete_freeze()
    cfg[group][field] = None
    blockers = validate_stage1_freeze(cfg)
    assert any(f"{group}.{field}" in b for b in blockers), (
        f"{group}.{field} is declared required but does not cause a refusal"
    )


def test_missing_group_refuses():
    cfg = complete_freeze()
    del cfg["budgets"]
    assert any("missing group: budgets" in b for b in validate_stage1_freeze(cfg))


# ---- Amendment O-2: presentation split, not binding split -----------------


def test_overlapping_train_and_eval_templates_refuse():
    cfg = complete_freeze()
    cfg["task"]["eval_template_ids"] = [5, 6]          # 5 is also in train
    blockers = validate_stage1_freeze(cfg)
    assert any("templates overlap" in b for b in blockers)


def test_empty_eval_split_refuses():
    cfg = complete_freeze()
    cfg["task"]["train_template_ids"] = list(range(8))
    cfg["task"]["eval_template_ids"] = []
    assert validate_stage1_freeze(cfg), "an empty eval split must not validate"


def test_split_must_account_for_every_presentation():
    cfg = complete_freeze()
    cfg["task"]["presentations_per_binding"] = 9
    assert any("does not equal" in b for b in validate_stage1_freeze(cfg))


def test_templates_must_come_from_the_declared_set():
    cfg = complete_freeze()
    cfg["task"]["eval_template_ids"] = [6, 99]
    assert any("not in template_set" in b for b in validate_stage1_freeze(cfg))


# ---- Amendment O-3: transience mechanism must be declared -----------------


def test_transience_mechanism_must_be_declared_explicitly():
    cfg = complete_freeze()
    cfg["mechanisms"]["fast_transience_mechanism"] = "whatever"
    assert any("fast_transience_mechanism" in b for b in validate_stage1_freeze(cfg))


def test_explicit_decay_requires_a_decay_constant():
    cfg = complete_freeze()
    cfg["mechanisms"]["fast_transience_mechanism"] = "explicit_decay"
    assert any("decay_constant" in b for b in validate_stage1_freeze(cfg))
    cfg["mechanisms"]["decay_constant"] = 0.997
    assert validate_stage1_freeze(cfg) == []


def test_both_transience_mechanisms_are_reachable():
    for m in TRANSIENCE_MECHANISMS:
        cfg = complete_freeze()
        cfg["mechanisms"]["fast_transience_mechanism"] = m
        if m == "explicit_decay":
            cfg["mechanisms"]["decay_constant"] = 0.997
        assert validate_stage1_freeze(cfg) == [], m


# ---- Amendment O-2: durability needs intervening writes -------------------


def test_single_interference_set_refuses():
    """Durability with nothing intervening is satisfied by doing nothing."""
    cfg = complete_freeze()
    cfg["interference"]["sets"] = [{"name": "A", "members": [1, 2, 3]}]
    assert any("at least two sets" in b for b in validate_stage1_freeze(cfg))


def test_interference_sets_must_be_disjoint():
    cfg = complete_freeze()
    cfg["interference"]["sets"][1]["members"] = [3, 4, 5]   # 3 repeats from A
    assert any("disjoint" in b for b in validate_stage1_freeze(cfg))


def test_empty_interference_set_refuses():
    cfg = complete_freeze()
    cfg["interference"]["sets"][1]["members"] = []
    assert validate_stage1_freeze(cfg)


# ---- seeds, grids, budgets, accelerator -----------------------------------


def test_seeds_must_be_the_preregistered_development_set():
    cfg = complete_freeze()
    cfg["task"]["seeds"] = [1, 2, 3]
    assert any("EXP002_DEV_SEEDS" in b for b in validate_stage1_freeze(cfg))


def test_every_declared_family_needs_a_capacity_grid():
    cfg = complete_freeze()
    cfg["candidates"]["families"] = ["low_rank_delta", "prompt_prefix"]
    assert any("prompt_prefix" in b for b in validate_stage1_freeze(cfg))


def test_non_positive_budgets_refuse():
    for field in ("update_budget_steps", "token_entitlement", "write_element_ceiling"):
        cfg = complete_freeze()
        cfg["budgets"][field] = 0
        assert any(field in b for b in validate_stage1_freeze(cfg)), field


def test_accelerator_must_have_passed():
    cfg = complete_freeze()
    cfg["accelerator"]["equivalence_passed"] = False
    assert any("equivalence_passed" in b for b in validate_stage1_freeze(cfg))


# ---- the script refuses ---------------------------------------------------


def test_freeze_script_refuses_an_incomplete_draft(tmp_path):
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(blank_stage1_freeze()))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_exp002_stage1.py"),
         "--draft", str(draft), "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0
    assert "REFUSED" in proc.stdout
    assert not (tmp_path / "out.json").exists(), "refused run must not emit an artifact"


def test_freeze_script_refuses_a_missing_draft(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/freeze_exp002_stage1.py"),
         "--draft", str(tmp_path / "absent.json"), "--out", str(tmp_path / "out.json")],
        capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode != 0 and "REFUSED" in proc.stdout


def test_no_stage1_freeze_artifact_exists_yet():
    """The proposed values are owner decisions and are not frozen."""
    assert not (ROOT / "results/exp002/stage1_freeze.json").exists()


def test_module_contains_no_model_or_training_code():
    """Stage-1 harness must not smuggle in experimental code."""
    src = (ROOT / "src/plasticity_routing/exp002_stage1.py").read_text().lower()
    for forbidden in ("import torch", "transformers", "from_pretrained", "backward()",
                      "optimizer", "loss"):
        assert forbidden not in src, f"harness references {forbidden!r}"
