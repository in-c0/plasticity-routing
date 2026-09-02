#!/usr/bin/env python3
"""Search the fixed-rule family under a budget matched to the learned router.

The primary comparator must not lose because it was searched less. This script
takes the learned router's total ES rollout count and spends the *same* number
of rollouts searching `ExtendedHeuristicRouter`, on development seeds only.

Search is random over the declared ranges rather than a grid, because the family
has eight knobs of mixed type and a grid dense enough to matter would be far
larger than the budget. The seed is fixed, so the search is reproducible.

Selection rule, fixed before running: take the argmax of the mean development
objective. No confirmatory seed is touched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001  # noqa: E402
from plasticity_routing.routers import ExtendedHeuristicRouter, HeuristicRouter  # noqa: E402
from plasticity_routing.substrates import EPISODIC, FAST  # noqa: E402
from plasticity_routing.world import build_world  # noqa: E402


def sample(rng) -> ExtendedHeuristicRouter:
    return ExtendedHeuristicRouter(
        seen_threshold=int(rng.integers(1, 5)),
        revision_tolerance=float(rng.uniform(0.4, 0.98)),
        error_floor=float(rng.uniform(0.05, 0.6)),
        slow_recurrence=int(rng.integers(1, 9)),
        query_evidence=float(rng.uniform(0.0, 4.0)),
        budget_guard=float(rng.uniform(0.0, 0.8)),
        recurrent_default=int(rng.choice([EPISODIC, FAST])),
        novel_action=int(rng.choice([EPISODIC, FAST])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--budget-rollouts", type=int, required=True,
                    help="total rollouts the learned router's ES consumes")
    ap.add_argument("--search-seed", type=int, default=20260902)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("results/heuristic_matched_search.json"))
    args = ap.parse_args()

    if set(args.seeds) & set(CONFIRMATORY_SEEDS):
        raise SystemExit("refusing to search on confirmatory seeds")

    worlds = {s: build_world(EXP001.world, seed=s) for s in args.seeds}
    n_configs = max(1, args.budget_rollouts // len(args.seeds))
    print(f"matched budget: {args.budget_rollouts} rollouts -> {n_configs} configurations "
          f"x {len(args.seeds)} dev seeds")

    def score(router) -> float:
        return float(np.mean([
            rollout(worlds[s], router, EXP001.cost, EXP001.substrate, seed=s).objective
            for s in args.seeds
        ]))

    base = HeuristicRouter()
    base_score = score(base)
    print(f"3-parameter HeuristicRouter (grid argmax): {base_score:.4f}")

    rng = np.random.default_rng(args.search_seed)
    rows = []
    best, best_score = None, -np.inf
    for i in range(n_configs):
        r = sample(rng)                       # drawn for every i so shards agree on the sequence
        if i % args.n_shards != args.shard:
            continue
        sc = score(r)
        rows.append({"params": r.params(), "objective": sc})
        if sc > best_score:
            best, best_score = r, sc
            print(f"  [{i:5d}] new best {sc:.4f}  {r.params()}")

    out = {
        "seeds": args.seeds,
        "budget_rollouts": args.budget_rollouts,
        "configurations_evaluated": n_configs,
        "search_seed": args.search_seed,
        "baseline_3param_objective": base_score,
        "best_objective": best_score,
        "best_params": best.params(),
        "improvement_over_3param": best_score - base_score,
        "top20": sorted(rows, key=lambda r: -r["objective"])[:20],
    }
    if args.n_shards > 1:
        args.out = args.out.with_name(f"{args.out.stem}_shard{args.shard}.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nbest extended heuristic: {best_score:.4f} (+{best_score - base_score:.4f} over 3-param)")
    print(f"params: {best.params()}")


if __name__ == "__main__":
    main()
