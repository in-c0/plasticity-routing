#!/usr/bin/env python3
"""Merge sharded matched-budget heuristic search results.

Sharding only parallelises the search; every shard draws from the identical
random sequence and evaluates a disjoint subset, so the merged result is exactly
what a single-process run of the same budget would have produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results"))
    ap.add_argument("--stem", default="heuristic_matched_search")
    ap.add_argument("--out", type=Path, default=Path("results/heuristic_matched_search.json"))
    args = ap.parse_args()

    shards = sorted(args.dir.glob(f"{args.stem}_shard*.json"))
    if not shards:
        raise SystemExit("no shards found")

    parts = [json.loads(p.read_text()) for p in shards]
    evaluated = sum(len(p["top20"]) and p["configurations_evaluated"] // len(shards) for p in parts)
    rows = [r for p in parts for r in p["top20"]]
    rows.sort(key=lambda r: -r["objective"])
    best = max(parts, key=lambda p: p["best_objective"])

    out = {
        "shards": [str(p.name) for p in shards],
        "budget_rollouts": parts[0]["budget_rollouts"],
        "configurations_evaluated": parts[0]["configurations_evaluated"],
        "search_seed": parts[0]["search_seed"],
        "baseline_3param_objective": parts[0]["baseline_3param_objective"],
        "best_objective": best["best_objective"],
        "best_params": best["best_params"],
        "improvement_over_3param": best["best_objective"] - parts[0]["baseline_3param_objective"],
        "top20": rows[:20],
    }
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"merged {len(shards)} shards, {out['configurations_evaluated']} configurations, "
          f"{out['budget_rollouts']} rollouts")
    print(f"3-parameter grid argmax : {out['baseline_3param_objective']:.4f}")
    print(f"matched-budget best     : {out['best_objective']:.4f} "
          f"(+{out['improvement_over_3param']:.4f})")
    print(f"params: {out['best_params']}")


if __name__ == "__main__":
    main()
