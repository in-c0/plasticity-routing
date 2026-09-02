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
  * the leakage audit -- including the cached L5 verdict -- passes and is
    current for this tree.

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

from plasticity_routing.config import CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001  # noqa: E402
from plasticity_routing.manifest import config_hash, environment, git_sha, source_tree_sha256  # noqa: E402
from plasticity_routing.routers import HeuristicRouter  # noqa: E402


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0")
    ap.add_argument("--skip-tests", action="store_true", help="development convenience only")
    ap.add_argument("--out-json", type=Path, default=ROOT / "results/protocol_v1_lock.json")
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()
    out_md = args.out_md or ROOT / f"experiments/PROTOCOL-v{args.version}.md"

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

    l5_path = ROOT / "results/l5_time_shuffle.json"
    l5 = json.loads(l5_path.read_text()) if l5_path.exists() else None
    tree = source_tree_sha256(ROOT)
    if not l5 or not l5.get("passed"):
        blockers.append("L5 verdict missing or failing")
    elif l5.get("source_tree_sha256") != tree:
        blockers.append("cached L5 verdict is stale for this source tree; re-run `make l5`")

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
        "l5": {"passed": bool(l5 and l5.get("passed")),
               "ratio_shuffled_over_real": (l5 or {}).get("ratio_shuffled_over_real")},
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(lock, indent=2, default=str) + "\n")

    blockers_md = ("\n".join(f"- {b}" for b in blockers)
                   if blockers else "_None. The protocol is frozen._")
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
| leakage audit L1-L7 | {'PASSED' if lock['leakage_audit_passed'] else 'FAILED / MISSING'} |
| L5 shuffled/real advantage ratio | {lock['l5']['ratio_shuffled_over_real']} |

Any edit to `src/`, `scripts/`, or the frozen configuration changes the
source-tree hash and invalidates this lock.

## Seeds

| role | seeds |
|---|---|
| development (calibration, training) | `{list(DEV_SEEDS)}` |
| confirmatory (held out) | `{list(CONFIRMATORY_SEEDS)}` |

The two lists are disjoint, and the freeze refuses to proceed if they are not.
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
