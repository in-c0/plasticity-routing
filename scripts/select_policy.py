#!/usr/bin/env python3
"""Apply the frozen policy-selection rule across cached policy seeds.

Selection rule (frozen before any confirmatory run): train `POLICY_SEEDS`
independent policies on development seeds and select the one with the highest
**development** objective. Confirmatory seeds play no part in selection.

This is ordinary model selection on development data, and it is only fair
because the comparator receives a matched search budget
(`scripts/search_heuristic_matched.py`). Selecting a policy seed by
*confirmatory* performance would be seed-shopping and is prohibited.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["real", "shuffled"], default="real")
    ap.add_argument("--dir", type=Path, default=ROOT / "results/policies")
    args = ap.parse_args()

    cands = sorted(args.dir.glob(f"{args.world}_seed*.json"))
    if not cands:
        raise SystemExit(f"no cached policies for world={args.world} in {args.dir}")

    rows = []
    for p in cands:
        d = json.loads(p.read_text())
        rows.append((d["meta"]["dev_objective"], d["meta"]["policy_seed"], p, d))
    rows.sort(key=lambda r: -r[0])

    print(f"{args.world} policies by development objective:")
    for obj, ps, p, d in rows:
        print(f"  seed {ps}: {obj:.4f}  actions {[round(x, 3) for x in d['meta']['action_probs']]}")

    best_obj, best_seed, best_path, _ = rows[0]
    dest = args.dir / f"{args.world}_selected.json"
    shutil.copyfile(best_path, dest)
    spread = rows[0][0] - rows[-1][0]
    print(f"\nselected: seed {best_seed} (dev objective {best_obj:.4f})")
    print(f"seed spread across {len(rows)} policies: {spread:.4f}")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
