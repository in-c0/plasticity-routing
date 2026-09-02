#!/usr/bin/env python3
"""Select the fixed-heuristic thresholds on development seeds.

The heuristic is the **primary comparator** for H1, because the closest
published comparison (Yoon 2026, arXiv:2606.30067) found a simple rule matching
or beating a learned allocation controller. Deliberately leaving it
under-tuned would be the most direct way to manufacture a positive result.

This script therefore searches the predeclared grid in
`experiments/EXP-001-PREREG.md` §8 and selects the setting that **maximises**
heuristic performance on development seeds. Confirmatory seeds are never used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001  # noqa: E402
from plasticity_routing.routers import HeuristicRouter  # noqa: E402
from plasticity_routing.world import build_world  # noqa: E402

GRID = {
    "seen_threshold": [1, 2, 3],
    "revision_tolerance": [0.7, 0.8, 0.9],
    "error_floor": [0.15, 0.25, 0.35],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--out", type=Path, default=Path("results/heuristic_calibration.json"))
    args = ap.parse_args()

    if set(args.dev_seeds) & set(CONFIRMATORY_SEEDS):
        raise SystemExit("refusing to calibrate on confirmatory seeds")

    worlds = {s: build_world(EXP001.world, seed=s) for s in args.dev_seeds}
    rows = []
    for st, rt, ef in itertools.product(*GRID.values()):
        r = HeuristicRouter(seen_threshold=st, revision_tolerance=rt, error_floor=ef)
        res = [rollout(worlds[s], r, EXP001.cost, EXP001.substrate, seed=s) for s in args.dev_seeds]
        rows.append({
            "seen_threshold": st, "revision_tolerance": rt, "error_floor": ef,
            "objective": float(np.mean([x.objective for x in res])),
            "task_utility": float(np.mean([x.task_utility for x in res])),
            "forgetting": float(np.mean([x.forgetting for x in res])),
            "action_probs": np.mean([x.action_probs for x in res], axis=0).tolist(),
        })

    rows.sort(key=lambda r: -r["objective"])
    best = rows[0]
    default = next(r for r in rows if (r["seen_threshold"], r["revision_tolerance"], r["error_floor"])
                   == (2, 0.8, 0.25))

    print(f"{'seen':>5} {'revtol':>7} {'errfloor':>9} | {'obj':>7} {'util':>6} {'forget':>7}")
    for r in rows:
        mark = "  <- selected" if r is best else ("  (current default)" if r is default else "")
        print(f"{r['seen_threshold']:>5} {r['revision_tolerance']:>7.2f} {r['error_floor']:>9.2f} | "
              f"{r['objective']:>7.4f} {r['task_utility']:>6.3f} {r['forgetting']:>7.3f}{mark}")

    print(f"\nbest: seen_threshold={best['seen_threshold']} "
          f"revision_tolerance={best['revision_tolerance']} error_floor={best['error_floor']} "
          f"objective={best['objective']:.4f}")
    print(f"current default objective={default['objective']:.4f}  "
          f"(gain from calibration: {best['objective'] - default['objective']:+.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"dev_seeds": args.dev_seeds, "grid": GRID,
                                    "rows": rows, "selected": best,
                                    "previous_default": default}, indent=2) + "\n")


if __name__ == "__main__":
    main()
