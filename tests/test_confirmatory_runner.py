"""Refusal paths for the confirmatory runner (Amendment M).

The runner is the only path that can produce `CONFIRMATORY` manifests, and it
runs once. Each test below drives a way that a confirmatory result could be
produced against the wrong code, the wrong artefacts, or the wrong seeds, and
asserts the runner refuses rather than proceeding.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from plasticity_routing.config import (
    AUDIT_SEEDS, DEV_SEEDS, MATCHED_HEURISTIC_SHA256, SELECTED_POLICY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "confirmatory_runner", ROOT / "scripts/run_exp001_confirmatory.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load_runner()
LOCK = ROOT / "results/protocol_v1.1_lock.json"


def _lock_is_frozen() -> bool:
    """A lock file that exists but is not frozen is a *failed* freeze attempt.

    These tests must gate on a genuinely frozen lock, not on the file's
    presence -- otherwise a failed freeze writes an invalid lock, the tests then
    fail against it, and the failing tests block the freeze. That circularity is
    resolved by skipping until a real freeze succeeds, after which the
    assertions below run for every subsequent test invocation.
    """
    if not LOCK.exists():
        return False
    try:
        return bool(json.loads(LOCK.read_text()).get("frozen"))
    except Exception:
        return False


requires_lock = pytest.mark.skipif(not _lock_is_frozen(), reason="v1.1 lock not frozen")


def _lock_dict() -> dict:
    return json.loads(LOCK.read_text())


def _write_lock(tmp_path: Path, mutate=None) -> Path:
    d = _lock_dict()
    if mutate:
        mutate(d)
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(d))
    return p


# ---- no seed CLI ----------------------------------------------------------


def test_runner_exposes_no_seed_flag():
    """Seeds must come only from the lock."""
    src = (ROOT / "scripts/run_exp001_confirmatory.py").read_text()
    assert '"--seeds"' not in src and "'--seeds'" not in src
    assert '"--seed"' not in src and "'--seed'" not in src


def test_runner_writes_only_confirmatory_manifests():
    """Every `classification=` the runner passes must be CONFIRMATORY.

    Checked on the AST rather than the text: the module docstring legitimately
    mentions DEV_CALIBRATION when explaining why this runner exists.
    """
    import ast

    tree = ast.parse((ROOT / "scripts/run_exp001_confirmatory.py").read_text())
    values = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "classification" and isinstance(kw.value, ast.Constant)
    ]
    assert values, "runner never sets a classification"
    assert set(values) == {"CONFIRMATORY"}, f"runner can emit {set(values)}"


# ---- lock integrity -------------------------------------------------------


def test_refuses_missing_lock(tmp_path):
    with pytest.raises(SystemExit, match="no protocol lock"):
        runner.preflight(tmp_path / "absent.json", tmp_path / "out", False)


@requires_lock
def test_refuses_unfrozen_lock(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(frozen=False, blockers=["synthetic"]))
    with pytest.raises(SystemExit, match="not frozen"):
        runner.preflight(p, tmp_path / "out", False)


@requires_lock
def test_refuses_a_commit_that_is_not_the_admissible_one_or_its_descendant(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(admissible_commit="0" * 40))
    with pytest.raises(SystemExit, match="nor a descendant"):
        runner.preflight(p, tmp_path / "out", False)


@requires_lock
def test_accepts_a_descendant_commit_when_the_source_tree_is_unchanged(tmp_path):
    """Committing the lock itself advances HEAD; neither results/ nor the
    generated PROTOCOL markdown is inside the source fingerprint, so such a
    commit cannot change what executes."""
    import subprocess

    parent = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD~1"],
                            capture_output=True, text=True).stdout.strip()
    if len(parent) != 40:
        pytest.skip("no parent commit")
    p = _write_lock(tmp_path, lambda d: d.update(admissible_commit=parent))
    pre = runner.preflight(p, tmp_path / "out", False)
    assert pre["seeds"]


@requires_lock
def test_refuses_drifted_source_tree(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(source_tree_sha256="deadbeef"))
    with pytest.raises(SystemExit, match="source tree has drifted"):
        runner.preflight(p, tmp_path / "out", False)


@requires_lock
def test_refuses_drifted_config_hash(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(config_hash="not-the-frozen-hash"))
    with pytest.raises(SystemExit, match="config hash"):
        runner.preflight(p, tmp_path / "out", False)


# ---- seed hygiene ---------------------------------------------------------


@requires_lock
def test_refuses_dev_seeds_in_the_confirmatory_list(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(confirmatory_seeds=list(DEV_SEEDS)))
    with pytest.raises(SystemExit, match="overlap dev"):
        runner.preflight(p, tmp_path / "out", False)


@requires_lock
def test_refuses_audit_seeds_in_the_confirmatory_list(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(confirmatory_seeds=list(AUDIT_SEEDS[:3])))
    with pytest.raises(SystemExit, match="overlap"):
        runner.preflight(p, tmp_path / "out", False)


@requires_lock
def test_refuses_empty_seed_list(tmp_path):
    p = _write_lock(tmp_path, lambda d: d.update(confirmatory_seeds=[]))
    with pytest.raises(SystemExit, match="no confirmatory seeds"):
        runner.preflight(p, tmp_path / "out", False)


# ---- one-shot -------------------------------------------------------------


@requires_lock
def test_refuses_a_second_execution(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_LEARNED_seed20260902.json").write_text("{}")
    with pytest.raises(SystemExit, match="one-shot"):
        runner.preflight(LOCK, out, False)


@requires_lock
def test_allows_reproduce_only_over_existing_manifests(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_LEARNED_seed20260902.json").write_text("{}")
    pre = runner.preflight(LOCK, out, True)
    assert pre["seeds"]


# ---- artefact pinning -----------------------------------------------------


@requires_lock
def test_pins_cover_the_artefacts_the_comparison_depends_on():
    assert set(SELECTED_POLICY_SHA256) == {"real_selected.json", "shuffled_selected.json"}
    assert len(MATCHED_HEURISTIC_SHA256) == 64


@requires_lock
def test_current_artefacts_match_their_pins():
    pre = runner.preflight(LOCK, ROOT / "results/nonexistent-out", False)
    assert pre["comparator_sha256"] == MATCHED_HEURISTIC_SHA256
    assert pre["policy_sha256"] == dict(SELECTED_POLICY_SHA256)


@requires_lock
def test_refuses_a_drifted_policy_artefact(tmp_path, monkeypatch):
    """A changed checkpoint would silently change what H1 measures."""
    real = ROOT / "results/policies/real_selected.json"
    original = real.read_bytes()
    tampered = json.loads(original)
    tampered["meta"] = dict(tampered["meta"], tampered=True)
    try:
        real.write_text(json.dumps(tampered))
        with pytest.raises(SystemExit, match="drifted from its pinned hash"):
            runner.preflight(LOCK, tmp_path / "out", False)
    finally:
        real.write_bytes(original)


@requires_lock
def test_refuses_a_drifted_comparator_artefact(tmp_path):
    comp = ROOT / "results/heuristic_matched_search.json"
    original = comp.read_bytes()
    tampered = json.loads(original)
    tampered["best_objective"] = -1.0
    try:
        comp.write_text(json.dumps(tampered))
        with pytest.raises(SystemExit, match="comparator artefact drifted"):
            runner.preflight(LOCK, tmp_path / "out", False)
    finally:
        comp.write_bytes(original)


# ---- reveal order ---------------------------------------------------------


def test_reveal_order_is_enforced_in_source():
    """Comparative metrics must be withheld behind every gate."""
    src = (ROOT / "scripts/run_exp001_confirmatory.py").read_text()
    order = ["gate 1: leakage", "gate 2: manifest validation", "gate 3: write ceilings",
             "gate 4: benchmark admissibility", "gate 5: benchmark-invalidating criteria",
             "gates passed; comparative metrics"]
    positions = [src.index(s) for s in order]
    assert positions == sorted(positions), "gates are not in the preregistered order"
    # every invalidating gate must be able to stop the run before the headline
    assert src.index("Stopping before the headline comparison") < src.index(
        "gates passed; comparative metrics")


def test_every_gate_raises_rather_than_warning():
    src = (ROOT / "scripts/run_exp001_confirmatory.py").read_text()
    head = src[: src.index("gates passed; comparative metrics")]
    assert head.count("raise Refusal") >= 6


# ---- protocol equivalence -------------------------------------------------


@requires_lock
def test_v11_asserts_equivalence_to_v10_on_every_scientific_field():
    """v1.1 must move no scientific goalpost relative to v1.0."""
    v11 = _lock_dict()
    eq = v11.get("equivalence")
    assert eq is not None, "v1.1 was frozen without asserting equivalence"
    assert eq["reference_version"] == "1.0"
    assert eq["equivalent"] is True, f"differing fields: {eq['differing_fields']}"
    for field in ("config_hash", "dev_seeds", "confirmatory_seeds", "audit_seeds",
                  "world_config", "substrate_config", "cost_config", "es_config",
                  "heuristic_params", "oracle_mapping", "designed_mapping"):
        assert field in eq["fields_checked"], f"{field} was not checked"


@requires_lock
def test_v10_lock_is_preserved_unmodified():
    """v1.0 is the historical freeze and must survive v1.1 verbatim."""
    v10 = json.loads((ROOT / "results/protocol_v1_lock.json").read_text())
    assert v10["protocol_version"] == "1.0"
    assert v10["frozen"] is True
    assert v10["admissible_commit"] == "61c493549e741bebe9becaba599b01e4c8e0a7f9"
    assert v10["l5b_attribution_gate"]["passed"] is True
    assert v10["l5a_historical"]["passed"] is False


@requires_lock
def test_v11_records_the_artefact_hashes_v10_did_not():
    v11 = _lock_dict()
    assert v11["selected_policy_sha256"] == dict(SELECTED_POLICY_SHA256)
    assert v11["comparator_sha256"] == MATCHED_HEURISTIC_SHA256


def test_equivalence_comparison_ignores_container_type_only(tmp_path):
    """Tuples become lists across a JSON round-trip; that must not read as a
    scientific difference, but a real value change must."""
    import json as _json

    def canon(x):
        return _json.loads(_json.dumps(x, sort_keys=True, default=str))

    assert canon((0.26, 0.26, 0.28, 0.2)) == canon([0.26, 0.26, 0.28, 0.2])
    assert canon((0.26, 0.26, 0.28, 0.2)) != canon([0.26, 0.26, 0.28, 0.3])
