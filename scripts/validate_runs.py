#!/usr/bin/env python3
"""Validate run manifests before a comparison may be labelled as evidence.

A run that fails validation may be reported as a pilot. It may not support a
claim. Exit code is non-zero when validation fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CLAIM_ARMS = {"ALL_IGNORE", "ALL_EPISODIC", "ALL_FAST", "ALL_SLOW",
              "HEURISTIC", "HEURISTIC_EXT", "RANDOM_MATCHED", "LEARNED"}


def validate(manifests: list[dict], tolerance: float = 0.02) -> dict:
    reasons: list[str] = []
    arms = {m["arm"] for m in manifests}

    missing = CLAIM_ARMS - arms
    if missing:
        reasons.append(f"missing_required_arms:{sorted(missing)}")

    claim = [m for m in manifests if m["arm"] in CLAIM_ARMS]

    # -- provenance --------------------------------------------------------
    trees = {m.get("source_tree_sha256") for m in claim}
    if not trees or None in trees or "" in trees:
        reasons.append("missing_source_tree_hash")
    elif len(trees) > 1:
        reasons.append("arms_use_different_source_trees")

    if len({m.get("config_hash") for m in claim}) > 1:
        reasons.append("arms_use_different_configs")

    # -- pairing -----------------------------------------------------------
    by_seed: dict[int, set[str]] = {}
    for m in claim:
        by_seed.setdefault(m["seed"], set()).add(m["arm"])
    unpaired = [s for s, a in by_seed.items() if a != CLAIM_ARMS - {x for x in CLAIM_ARMS if x not in arms}]
    if any(len(a) != len(CLAIM_ARMS & arms) for a in by_seed.values()):
        reasons.append(f"arms_not_paired_on_every_seed:{sorted(unpaired)[:5]}")

    # -- legality ----------------------------------------------------------
    illegal = [m["arm"] for m in claim if not m.get("decision_time_legal", False)]
    if illegal:
        reasons.append(f"privileged_arm_among_claim_arms:{sorted(set(illegal))}")

    # -- leakage -----------------------------------------------------------
    for m in manifests:
        audit = m.get("leakage_audit")
        if audit is None:
            reasons.append(f"missing_leakage_audit:{m['arm']}")
        elif not audit.get("passed", False):
            reasons.append(f"leakage_audit_failed:{m['arm']}")

    # -- resources ---------------------------------------------------------
    for m in claim:
        r = m.get("resources", {})
        if r.get("write_ceiling_exceeded"):
            reasons.append(f"write_ceiling_exceeded:{m['arm']}")
        for field in ("write_elements", "storage_element_steps", "router_compute_elements",
                      "total_compute_elements"):
            if field not in r:
                reasons.append(f"missing_resource_field:{m['arm']}:{field}")

    ceilings = {m["cost_config"]["write_element_ceiling"] for m in claim}
    if len(ceilings) > 1:
        reasons.append("arms_have_different_write_ceilings")

    # -- classification ----------------------------------------------------
    classes = {m.get("classification") for m in manifests}
    if "CONFIRMATORY" in classes and len(classes) > 1:
        reasons.append("mixed_confirmatory_and_non_confirmatory_manifests")

    lock_path = Path(__file__).resolve().parents[1] / "results" / "protocol_v1_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    for m in manifests:
        if m.get("classification") == "CONFIRMATORY":
            if not m.get("git_sha"):
                reasons.append(f"confirmatory_run_without_git_commit:{m['arm']}")
            if lock is None or not lock.get("frozen"):
                reasons.append(f"confirmatory_run_without_frozen_protocol:{m['arm']}")
            else:
                if m.get("source_tree_sha256") != lock.get("source_tree_sha256"):
                    reasons.append(f"confirmatory_run_off_frozen_source_tree:{m['arm']}")
                if m.get("config_hash") != lock.get("config_hash"):
                    reasons.append(f"confirmatory_run_off_frozen_config:{m['arm']}")
                if m.get("seed") not in lock.get("confirmatory_seeds", []):
                    reasons.append(f"confirmatory_run_on_unlisted_seed:{m['arm']}:{m['seed']}")
            if m["seed"] in m.get("dev_seeds", []):
                reasons.append(f"confirmatory_seed_overlaps_dev_seed:{m['arm']}:{m['seed']}")
        if m.get("invalidation_reasons"):
            reasons.append(f"run_level_invalidation:{m['arm']}")

    # -- benchmark validity (K4/K5/K6 mechanical parts) --------------------
    utils = {m["arm"]: m["metrics"]["task_utility"] for m in manifests}
    if utils and min(utils.values()) > 0.95:
        reasons.append("ceiling_effect_all_arms_above_0.95")

    return {"passed": not reasons, "reasons": reasons,
            "n_manifests": len(manifests), "arms": sorted(arms),
            "classifications": sorted(c for c in classes if c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paths = [p for p in args.runs if p.exists()]
    if not paths:
        print("no manifests found", file=sys.stderr)
        sys.exit(2)

    manifests = [json.loads(p.read_text()) for p in paths]
    result = validate(manifests)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    print(f"manifests: {result['n_manifests']}  arms: {', '.join(result['arms'])}")
    print(f"classifications: {', '.join(result['classifications'])}")
    if result["passed"]:
        print("VALIDATION PASSED")
    else:
        print("VALIDATION FAILED")
        for r in result["reasons"]:
            print(f"  - {r}")
    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
