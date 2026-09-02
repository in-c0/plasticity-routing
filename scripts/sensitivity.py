#!/usr/bin/env python3
"""Robustness probe: is the optimal class->action mapping a knife-edge?

Development-only. The learned router is never run here.

A benchmark whose ground truth flips under a small change to a nuisance
parameter is a weak benchmark. This sweeps the neighbourhood of the candidate
configuration and reports where the optimal mapping changes. A *large* change
in a resource (say, 60% more episodic capacity) is expected to shift the optimum
-- that is the resource-dependence the whole track is about -- so only the local
neighbourhood is required to be stable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity_routing.ledger import CostConfig  # noqa: E402
from plasticity_routing.routers import is_bijective, search_best_mapping  # noqa: E402
from plasticity_routing.substrates import SubstrateConfig  # noqa: E402
from plasticity_routing.world import WorldConfig, build_world  # noqa: E402

BASE_W = dict(key_dim=96, n_stable_keys=70, n_local_slots=44,
              one_off_delay_max=60, stable_query_horizon=2200,
              class_prior=(0.26, 0.26, 0.28, 0.20))
BASE_S = dict(key_dim=96, episodic_capacity=24, fast_lr=1.0, fast_decay=0.997, slow_lr=0.7)
BASE_C = dict(key_dim=96, write_element_ceiling=3_000_000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    ap.add_argument("--out", type=Path, default=Path("results/sensitivity.json"))
    args = ap.parse_args()

    wcfg = replace(WorldConfig(), **BASE_W)
    ccfg = replace(CostConfig(), **BASE_C)
    worlds = {s: build_world(wcfg, seed=s) for s in args.seeds}

    axes = {
        "episodic_capacity": [20, 22, 24, 26, 28],
        "fast_decay": [0.995, 0.996, 0.997, 0.998, 0.999],
        "slow_lr": [0.5, 0.7, 0.9],
        "n_local_slots": [40, 44, 48],
    }
    rows = []
    for axis, values in axes.items():
        for v in values:
            if axis in BASE_W:
                wc = replace(WorldConfig(), **(BASE_W | {axis: v}))
                ws = {s: build_world(wc, seed=s) for s in args.seeds}
                scfg = replace(SubstrateConfig(), **BASE_S)
            else:
                ws = worlds
                scfg = replace(SubstrateConfig(), **(BASE_S | {axis: v}))
            bm, bo, _ = search_best_mapping(ws, ccfg, scfg, args.seeds, method="exhaustive")
            row = {"axis": axis, "value": v, "mapping": list(bm),
                   "bijective": is_bijective(bm), "objective": round(bo, 4)}
            rows.append(row)
            print(f"{axis:>18} = {v:<7} -> map={list(bm)} bij={is_bijective(bm)} obj={bo:.3f}", flush=True)

    mappings = {tuple(r["mapping"]) for r in rows}
    stable = len(mappings) == 1
    print(f"\ndistinct optimal mappings across neighbourhood: {len(mappings)} -> {sorted(mappings)}")
    print(f"LOCALLY STABLE: {stable}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seeds": args.seeds, "base_world": BASE_W,
                                    "base_substrate": BASE_S, "base_cost": BASE_C,
                                    "rows": rows, "locally_stable": stable,
                                    "distinct_mappings": [list(m) for m in sorted(mappings)]},
                                   indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
