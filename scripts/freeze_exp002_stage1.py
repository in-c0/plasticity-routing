#!/usr/bin/env python3
"""Freeze the EXP-002 Stage-1 configuration, or refuse and say why.

Amendment O-1. Emits `results/exp002/stage1_freeze.json` only when every field
required by the preregistration §13 is set and internally consistent. Until
then it exits non-zero and lists each blocker.

This is the terminal artifact of the Stage-1 plan: the Stage-1 runner takes its
parameters only from the emitted file. There is deliberately no CLI for `N`, the
grids, or the budgets -- the same posture as `run_exp001_confirmatory.py`, which
takes its seeds only from the protocol lock.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasticity_routing.exp002_stage1 import (  # noqa: E402
    REQUIRED_FIELDS, blank_stage1_freeze, validate_stage1_freeze,
)
from plasticity_routing.manifest import config_hash, git_sha, source_tree_sha256  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=ROOT / "experiments/exp002_stage1_draft.json",
                    help="candidate configuration authored by the owner")
    ap.add_argument("--out", type=Path, default=ROOT / "results/exp002/stage1_freeze.json")
    ap.add_argument("--emit-skeleton", action="store_true",
                    help="write a fully-unset skeleton to --draft and exit")
    args = ap.parse_args()

    if args.emit_skeleton:
        args.draft.parent.mkdir(parents=True, exist_ok=True)
        args.draft.write_text(json.dumps(blank_stage1_freeze(), indent=2) + "\n")
        n = sum(len(v) for v in REQUIRED_FIELDS.values())
        print(f"wrote unset skeleton with {n} required fields to {args.draft}")
        print("fill it in, then re-run without --emit-skeleton")
        return

    if not args.draft.exists():
        print(f"REFUSED: no draft at {args.draft}")
        print("  run with --emit-skeleton to author one")
        sys.exit(1)

    freeze = json.loads(args.draft.read_text())
    freeze.setdefault("provenance", {})
    freeze["provenance"]["git_sha"] = git_sha(ROOT)
    freeze["provenance"]["source_tree_sha256"] = source_tree_sha256(ROOT)
    freeze["provenance"]["config_hash"] = config_hash(freeze.get("task"), freeze.get("candidates"),
                                                      freeze.get("budgets"))

    blockers = validate_stage1_freeze(freeze)
    if blockers:
        print(f"REFUSED: Stage-1 configuration is not freezable ({len(blockers)} blockers)")
        for b in blockers:
            print(f"  - {b}")
        print("\nNo Stage-1 candidate result may be produced until these are resolved.")
        sys.exit(1)

    freeze["frozen"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    print("Stage-1 configuration FROZEN")
    print(f"  git sha      {freeze['provenance']['git_sha']}")
    print(f"  source tree  {freeze['provenance']['source_tree_sha256'][:16]}...")
    print(f"  config hash  {freeze['provenance']['config_hash']}")
    print(f"  wrote        {args.out}")


if __name__ == "__main__":
    main()
