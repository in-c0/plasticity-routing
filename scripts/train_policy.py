#!/usr/bin/env python3
"""Train one routing policy and cache it. Development seeds only.

One process per policy seed, so the declared multi-seed training procedure can
be run in parallel without changing it. `scripts/select_policy.py` then applies
the frozen selection rule across the cached seeds.

`--world shuffled` trains in the L5 time-shuffled control world, using the
identical procedure and budget, so that the two conditions differ only in
whether future utility is predictable.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001  # noqa: E402
from plasticity_routing.manifest import source_tree_sha256  # noqa: E402
from plasticity_routing.train import save_policy, train_router_es  # noqa: E402
from plasticity_routing.world import build_world, time_shuffled_world  # noqa: E402


def make_worlds(kind: str, seeds: list[int]) -> dict:
    real = {s: build_world(EXP001.world, seed=s) for s in seeds}
    if kind == "real":
        return real
    if kind == "shuffled":
        return {s: time_shuffled_world(real[s], seed=s) for s in seeds}
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["real", "shuffled"], default="real")
    ap.add_argument("--policy-seed", type=int, required=True)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/policies")
    args = ap.parse_args()

    if set(args.seeds) & set(CONFIRMATORY_SEEDS):
        raise SystemExit("refusing to train on confirmatory seeds")

    cfg = EXP001.train if args.generations is None else replace(EXP001.train,
                                                               generations=args.generations)
    worlds = make_worlds(args.world, args.seeds)
    router, hist = train_router_es(EXP001.world, EXP001.substrate, EXP001.cost, args.seeds,
                                   cfg, policy_seed=args.policy_seed, verbose=True, worlds=worlds)
    router.greedy = True
    rs = [rollout(worlds[s], router, EXP001.cost, EXP001.substrate, seed=s) for s in args.seeds]
    probs = np.mean([r.action_probs for r in rs], axis=0)

    out = args.out_dir / f"{args.world}_seed{args.policy_seed}.json"
    save_policy(out, router, {
        "world": args.world,
        "policy_seed": args.policy_seed,
        "dev_seeds": args.seeds,
        "generations": cfg.generations,
        "es_config": cfg.__dict__,
        "dev_objective": float(np.mean([r.objective for r in rs])),
        "dev_task_utility": float(np.mean([r.task_utility for r in rs])),
        "action_probs": probs.tolist(),
        "source_tree_sha256": source_tree_sha256(ROOT),
        "classification": "DEV_CALIBRATION",
    })
    print(f"\n{args.world} seed {args.policy_seed}: dev objective "
          f"{float(np.mean([r.objective for r in rs])):.4f}  actions {[round(x, 3) for x in probs]}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
