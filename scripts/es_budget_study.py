#!/usr/bin/env python3
"""ES budget convergence study. Development seeds only.

EXP-000 used 60 generations and its development objective was still rising at
the last generation (0.371 -> 0.385 -> 0.386 over the final twenty), so the
frozen budget may simply be too small. The budget question must be settled
*once*, before protocol freeze, on development seeds, and on evidence about
convergence -- not by trying budgets until one wins.

This runs a long ES trace and reports where the development objective plateaus.
The decision rule, fixed before running:

    Increase the budget to the smallest value at which the development objective
    has plateaued -- no improvement above `--plateau-eps` over a window of
    `--window` generations. If the curve has not plateaued even at the longest
    budget tried, take the longest budget tried and say so.

Any increase must be matched by an equivalent increase in the *heuristic's*
search budget (see scripts/search_heuristic_matched.py), because a comparator
given a hundredth of the search effort is not a fair test.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasticity_routing.config import DEV_SEEDS, EXP001  # noqa: E402
from plasticity_routing.train import train_router_es  # noqa: E402
from plasticity_routing.world import build_world  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--plateau-eps", type=float, default=0.004)
    ap.add_argument("--out", type=Path, default=Path("results/es_budget_study.json"))
    args = ap.parse_args()

    worlds = {s: build_world(EXP001.world, seed=s) for s in args.seeds}
    cfg = replace(EXP001.train, generations=args.generations)

    traces = {}
    for ps in args.policy_seeds:
        print(f"-- ES policy seed {ps}, {args.generations} generations --", flush=True)
        _, hist = train_router_es(EXP001.world, EXP001.substrate, EXP001.cost, args.seeds,
                                  cfg, policy_seed=ps, verbose=True, worlds=worlds)
        curve = [h["dev_objective"] for h in hist if "generation" in h]
        traces[f"policy_seed_{ps}"] = curve

    # Merge in any sibling single-seed runs so the study can be parallelised
    # across processes without changing the procedure being measured.
    for extra in sorted(args.out.parent.glob(f"{args.out.stem}_seed*.json")):
        d = json.loads(extra.read_text())
        traces.update(d["traces"])

    n = min(len(v) for v in traces.values())
    curves = np.array([traces[k][:n] for k in sorted(traces)])
    running_best = np.maximum.accumulate(curves, axis=1).mean(axis=0)

    plateau_at = None
    for g in range(args.window, len(running_best)):
        if running_best[g] - running_best[g - args.window] < args.plateau_eps:
            plateau_at = g
            break

    evals_per_gen = EXP001.train.population * EXP001.train.seeds_per_generation + len(args.seeds)
    chosen = plateau_at if plateau_at is not None else args.generations

    print(f"\nmean running-best at gen 60:  {running_best[min(59, len(running_best) - 1)]:.4f}")
    print(f"mean running-best at gen {len(running_best)}: {running_best[-1]:.4f}")
    print(f"plateau (no +{args.plateau_eps} over {args.window} gens) at generation: {plateau_at}")
    print(f"recommended generations: {chosen}")
    print(f"rollouts per generation: {evals_per_gen}  -> total ES rollouts: {chosen * evals_per_gen}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seeds": args.seeds, "policy_seeds": args.policy_seeds,
        "generations_tried": args.generations, "window": args.window,
        "plateau_eps": args.plateau_eps, "traces": traces,
        "mean_running_best": running_best.tolist(),
        "plateau_generation": plateau_at, "recommended_generations": chosen,
        "rollouts_per_generation": evals_per_gen,
        "total_es_rollouts_at_recommendation": chosen * evals_per_gen,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
