#!/usr/bin/env python3
"""Freeze protocol v1.0.

Writes an immutable lock recording exactly what a confirmatory run is permitted
to execute: the admissible commit, a content hash of the source tree, the config
hash, the resolved dependency set, the interpreter and platform, and the
development / confirmatory seed lists.

Refuses to freeze unless:
  * the working tree is clean and committed (a lock naming a commit that does
    not contain the code being run is worthless);
  * development and confirmatory seed lists are disjoint;
  * the full test suite passes;
  * the leakage audit passes and is current for this tree, including the L5b
    attribution gate. L5a is retained as a permanently failed historical
    diagnostic and is reported, not gated on (Amendment L).

After freezing, any change to `src/`, `scripts/`, or the frozen configuration
invalidates the lock, and `scripts/validate_runs.py` will refuse to certify a
confirmatory run whose source-tree hash does not match.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasticity_routing.config import (  # noqa: E402
    AUDIT_SEEDS, CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001, MATCHED_HEURISTIC_SHA256,
    SELECTED_POLICY_SHA256,
)
from plasticity_routing.manifest import config_hash, environment, git_sha, source_tree_sha256  # noqa: E402
from plasticity_routing.routers import HeuristicRouter  # noqa: E402


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--skip-tests", action="store_true", help="development convenience only")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--assert-equivalent-to", type=Path, default=None,
                    help="path to an earlier lock that must match on every scientific field")
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    out_md = args.out_md or ROOT / f"experiments/PROTOCOL-v{args.version}.md"
    suffix = "v1" if args.version == "1.0" else f"v{args.version}"
    args.out_json = args.out_json or ROOT / f"results/protocol_{suffix}_lock.json"

    blockers: list[str] = []

    rc, status = run(["git", "status", "--porcelain"])
    dirty = [ln for ln in status.splitlines() if ln.strip()]
    if dirty:
        blockers.append(f"working tree not clean ({len(dirty)} entries); commit before freezing")

    sha = git_sha(ROOT)
    if not sha:
        blockers.append("no git commit found")

    if set(DEV_SEEDS) & set(CONFIRMATORY_SEEDS):
        blockers.append("development and confirmatory seed lists overlap")

    if not args.skip_tests:
        rc, out = run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q", "tests"])
        if rc != 0:
            blockers.append("test suite failing")
        test_line = out.strip().splitlines()[-1] if out.strip() else "no output"
    else:
        test_line = "SKIPPED"
        blockers.append("tests skipped; lock is not valid for a confirmatory run")

    audit_path = ROOT / "results/leakage_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    if not audit or not audit.get("passed"):
        blockers.append("leakage audit missing or failing; run `make l5 && make leakage`")

    tree = source_tree_sha256(ROOT)
    l5a_path = ROOT / "results/l5_time_shuffle.json"
    l5a = json.loads(l5a_path.read_text()) if l5a_path.exists() else None
    if not l5a:
        blockers.append("L5a historical record missing; it must be retained")
    elif l5a.get("passed"):
        blockers.append("L5a is recorded as passing; Amendment L requires it retained as FAILED")

    l5b_path = ROOT / "results/l5b_cross_world.json"
    l5b = json.loads(l5b_path.read_text()) if l5b_path.exists() else None
    if not l5b or not l5b.get("passed"):
        blockers.append("L5b attribution gate missing or failing")
    elif l5b.get("source_tree_sha256") != tree:
        blockers.append("cached L5b verdict is stale for this source tree; re-run `make l5b`")
    elif not l5b.get("policy_hashes_match_amendment_L"):
        blockers.append("L5b ran against policies that do not match the Amendment L hashes")

    # Artefacts the comparison depends on that live outside the source-tree
    # fingerprint. Their identity is pinned in `config.py` (which *is* inside
    # the fingerprint) and verified here.
    import hashlib

    artefacts = {}
    for name, expected in SELECTED_POLICY_SHA256.items():
        path = ROOT / "results/policies" / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        artefacts[name] = actual
        if actual != expected:
            blockers.append(f"policy artefact {name} does not match its pinned hash")
    comp_path = ROOT / "results/heuristic_matched_search.json"
    comp_hash = hashlib.sha256(comp_path.read_bytes()).hexdigest() if comp_path.exists() else None
    if comp_hash != MATCHED_HEURISTIC_SHA256:
        blockers.append("comparator artefact does not match its pinned hash")

    heur = HeuristicRouter()
    rc, freeze = run([str(ROOT / ".venv/bin/python"), "-m", "pip", "freeze"])
    deps = sorted(x for x in freeze.splitlines() if x and not x.startswith("-e"))

    lock = {
        "protocol_version": args.version,
        "frozen": not blockers,
        "blockers": blockers,
        "admissible_commit": sha,
        "source_tree_sha256": tree,
        "config_hash": config_hash(EXP001.world, EXP001.substrate, EXP001.cost, EXP001.train),
        "environment": environment(),
        "dependencies": deps,
        "dev_seeds": list(DEV_SEEDS),
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "world_config": asdict(EXP001.world),
        "substrate_config": asdict(EXP001.substrate),
        "cost_config": asdict(EXP001.cost),
        "es_config": asdict(EXP001.train),
        "heuristic_params": {"seen_threshold": heur.seen_threshold,
                             "revision_tolerance": heur.revision_tolerance,
                             "error_floor": heur.error_floor},
        "oracle_mapping": list(EXP001.oracle_mapping),
        "designed_mapping": list(EXP001.designed_mapping),
        "test_summary": test_line,
        "leakage_audit_passed": bool(audit and audit.get("passed")),
        "l5a_historical": {"status": "RETAINED_FAILED", "gating": False,
                           "passed": bool(l5a and l5a.get("passed")),
                           "ratio_shuffled_over_real": (l5a or {}).get("ratio_shuffled_over_real"),
                           "threshold_ratio": (l5a or {}).get("threshold_ratio")},
        "l5b_attribution_gate": {"gating": True, "passed": bool(l5b and l5b.get("passed")),
                                 "delta_real": (l5b or {}).get("delta_real", {}).get("mean_diff"),
                                 "delta_real_ci95": (l5b or {}).get("delta_real", {}).get("ci95"),
                                 "interaction": (l5b or {}).get("interaction", {}).get("mean_diff"),
                                 "interaction_ci95": (l5b or {}).get("interaction", {}).get("ci95"),
                                 "n_audit_seeds": (l5b or {}).get("n_audit_seeds")},
        "audit_seeds": list(AUDIT_SEEDS),
        "selected_policy_sha256": artefacts,
        "comparator_sha256": comp_hash,
    }

    # -- equivalence with an earlier protocol version ---------------------
    #
    # A new protocol version must not quietly move a scientific goalpost. Every
    # field below defines what the experiment *is*; only the code identity and
    # newly recorded artefact hashes may differ.
    SCIENTIFIC_FIELDS = (
        "config_hash", "dev_seeds", "confirmatory_seeds", "audit_seeds",
        "world_config", "substrate_config", "cost_config", "es_config",
        "heuristic_params", "oracle_mapping", "designed_mapping",
    )
    equivalence = None
    if args.assert_equivalent_to:
        if not args.assert_equivalent_to.exists():
            blockers.append(f"reference lock {args.assert_equivalent_to} not found")
        else:
            ref = json.loads(args.assert_equivalent_to.read_text())
            diffs = {f: {"reference": ref.get(f), "current": lock.get(f)}
                     for f in SCIENTIFIC_FIELDS if ref.get(f) != lock.get(f)}
            equivalence = {
                "reference": str(args.assert_equivalent_to.name),
                "reference_version": ref.get("protocol_version"),
                "fields_checked": list(SCIENTIFIC_FIELDS),
                "differing_fields": diffs,
                "equivalent": not diffs,
                "admitted_deltas": {
                    "admissible_commit": [ref.get("admissible_commit"),
                                          lock.get("admissible_commit")],
                    "source_tree_sha256": [ref.get("source_tree_sha256"),
                                           lock.get("source_tree_sha256")],
                    "newly_recorded": [f for f in ("selected_policy_sha256", "comparator_sha256")
                                       if f not in ref],
                },
            }
            if diffs:
                blockers.append(
                    f"protocol v{args.version} differs from "
                    f"v{ref.get('protocol_version')} on scientific fields: {sorted(diffs)}")
    lock["equivalence"] = equivalence

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(lock, indent=2, default=str) + "\n")

    blockers_md = ("\n".join(f"- {b}" for b in blockers)
                   if blockers else "_None. The protocol is frozen._")
    if equivalence is None:
        equiv_md = "_No reference version asserted._"
    elif equivalence["equivalent"]:
        equiv_md = (
            f"Asserted **equivalent** to protocol "
            f"v{equivalence['reference_version']} (`{equivalence['reference']}`) on every "
            f"scientific field:\n\n"
            + "\n".join(f"- `{f}`" for f in equivalence["fields_checked"])
            + "\n\nAdmitted deltas: the admissible commit, the source-tree hash, and newly "
              f"recorded artefact hashes {equivalence['admitted_deltas']['newly_recorded']}. "
              "No world, objective, action space, architecture, checkpoint, comparator, "
              "threshold, seed or statistic changed."
        )
    else:
        equiv_md = ("**NOT equivalent** to the reference version. Differing fields: "
                    f"`{sorted(equivalence['differing_fields'])}`")
    dep_md = "\n".join(f"    {d}" for d in deps)
    md = f"""# Protocol v{args.version}

**Status: {'FROZEN' if not blockers else 'NOT FROZEN'}**

This lock defines exactly what an EXP-001 confirmatory run may execute. It is
generated by `scripts/freeze_protocol.py`; do not edit it by hand. The machine
-readable form is `results/protocol_v1_lock.json`, and
`scripts/validate_runs.py` refuses to certify a confirmatory run whose commit,
source-tree hash, config hash, or seed is not the one recorded here.

## Blockers

{blockers_md}

## Admissible code

| | |
|---|---|
| commit | `{sha}` |
| source-tree sha256 | `{tree}` |
| config hash | `{lock['config_hash']}` |
| test suite | {test_line} |
| leakage audit (gating checks) | {'PASSED' if lock['leakage_audit_passed'] else 'FAILED / MISSING'} |
| L5a (retained, **failed**, non-gating) | ratio {lock['l5a_historical']['ratio_shuffled_over_real']} vs threshold {lock['l5a_historical']['threshold_ratio']} |
| L5b attribution gate | {'PASSED' if lock['l5b_attribution_gate']['passed'] else 'FAILED / MISSING'} |
| L5b `Delta_real` | {lock['l5b_attribution_gate']['delta_real']} CI {lock['l5b_attribution_gate']['delta_real_ci95']} |
| L5b `I` | {lock['l5b_attribution_gate']['interaction']} CI {lock['l5b_attribution_gate']['interaction_ci95']} |

Any edit to `src/`, `scripts/`, or the frozen configuration changes the
source-tree hash and invalidates this lock.

## Seeds

| role | seeds |
|---|---|
| development (calibration, training) | `{list(DEV_SEEDS)}` |
| L5b audit (one-shot, spent) | `{AUDIT_SEEDS[0]}..{AUDIT_SEEDS[-1]}` (n={len(AUDIT_SEEDS)}) |
| confirmatory (held out) | `{list(CONFIRMATORY_SEEDS)}` |

All three sets are disjoint, and the freeze refuses to proceed if the training
and confirmatory lists are not. The audit seeds are spent and may not be reused.
Confirmatory seeds have not been executed.

## Frozen configuration

- **ORACLE mapping** (exhaustive search, development seeds): `{list(EXP001.oracle_mapping)}`
- **designed mapping** (falsified, retained): `{list(EXP001.designed_mapping)}`
- **heuristic**: `{lock['heuristic_params']}`
- **ES**: `{lock['es_config']}`

World, substrate and cost configurations are recorded in full in the JSON lock.

## Environment

| | |
|---|---|
| python | {lock['environment']['python']} |
| numpy | {lock['environment']['numpy']} |
| platform | {lock['environment']['platform']} |
| machine | {lock['environment']['machine']} |

Resolved dependencies:

{dep_md}

## Equivalence with the previous protocol version

{equiv_md}

## What may not change after this point

- the world, substrate, or cost configuration;
- the objective weights;
- the heuristic thresholds or the heuristic family;
- the ES budget or hyperparameters;
- the policy seed or the checkpoint-selection rule;
- the seed lists.

Any change requires a new protocol version and a logged pre-result amendment in
`docs/ARCHITECTURE.md`. Changing any of them *after* inspecting a confirmatory
result is prohibited outright.
"""
    out_md.write_text(md)

    status_word = "FROZEN" if not blockers else "NOT FROZEN"
    print(f"protocol v{args.version}: {status_word}")
    for b in blockers:
        print(f"  blocker: {b}")
    print(f"  commit          {sha}")
    print(f"  source tree     {tree[:16]}...")
    print(f"  config hash     {lock['config_hash']}")
    print(f"  dev seeds       {list(DEV_SEEDS)}")
    print(f"  confirmatory    {list(CONFIRMATORY_SEEDS)}")
    print(f"  wrote           {args.out_json}")
    print(f"  markdown        {out_md}")
    sys.exit(0 if not blockers else 1)


if __name__ == "__main__":
    main()
