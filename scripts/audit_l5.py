#!/usr/bin/env python3
"""Leakage test L5 — time-shuffled control.

Retrains the routing policy in a world where the future-utility schedule has
been randomised (`world.time_shuffled_world`), and checks that its advantage
over budget-matched random routing collapses.

Why the criterion is *comparative* rather than absolute. The shuffle keeps the
write stream, the query count, and the query timing identical, and redraws each
query's target uniformly over the keys already written at that moment. That
destroys the coupling between an item's observable prefix and whether it will be
needed later, but it cannot destroy every regularity: keys written earlier are
live longer and so are queried more often, leaving a residual
past-to-future query correlation (measured and reported below). A learned router
can legitimately exploit that residue, so L5 does not require the shuffled
advantage to be zero. It requires it to be a small fraction of the real
advantage.

Preregistered criterion (frozen before this script was first run):

    L5 passes iff  shuffled_advantage <= 0.25 * real_advantage,
    or the shuffled advantage's paired-bootstrap 95% CI includes zero.

where advantage = objective(LEARNED) - objective(RANDOM_MATCHED), each measured
in its own world with its own budget-matched control.

This is expensive (it trains a second policy), so it is run separately and its
verdict is cached in results/l5_time_shuffle.json together with the source-tree
hash it was computed under. `scripts/audit_leakage.py` verifies that the cached
verdict exists, passes, and is current.
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
from plasticity_routing.config import DEV_SEEDS, EXP001  # noqa: E402
from plasticity_routing.manifest import source_tree_sha256  # noqa: E402
from plasticity_routing.metrics import paired_bootstrap  # noqa: E402
from plasticity_routing.routers import RandomMatchedRouter  # noqa: E402
from plasticity_routing.train import load_policy  # noqa: E402
from plasticity_routing.world import build_world, time_shuffled_world  # noqa: E402

THRESHOLD = 0.25


def residual_query_correlation(world) -> float:
    from collections import Counter

    half = EXP001.world.lifetime // 2
    past = Counter(e.key_id for e in world.events if e.kind == "QUERY" and e.t < half)
    future = Counter(e.key_id for e in world.events if e.kind == "QUERY" and e.t >= half)
    keys = set(past) | set(future)
    if len(keys) < 3:
        return float("nan")
    a = np.array([past[k] for k in keys], float)
    b = np.array([future[k] for k in keys], float)
    return float(np.corrcoef(a, b)[0, 1])


def advantage(worlds, seeds, label: str, policy_path: Path):
    """Measure a cached policy against its own budget-matched random control.

    The policy is trained beforehand by `scripts/train_policy.py` under the
    frozen procedure -- the same number of policy seeds, the same generations,
    the same selection rule -- in each world. Training here would risk the two
    conditions silently diverging from the procedure being frozen.
    """
    learned, meta = load_policy(policy_path)
    learned.greedy = True
    lr = [rollout(worlds[s], learned, EXP001.cost, EXP001.substrate, seed=s) for s in seeds]
    probs = np.mean([r.action_probs for r in lr], axis=0)
    rnd = RandomMatchedRouter(probs)
    rr = [rollout(worlds[s], rnd, EXP001.cost, EXP001.substrate, seed=s) for s in seeds]
    st = paired_bootstrap([r.objective for r in lr], [r.objective for r in rr])
    return {
        "label": label,
        "policy": str(policy_path.name),
        "policy_seed": meta.get("policy_seed"),
        "generations": meta.get("generations"),
        "learned_objective": float(np.mean([r.objective for r in lr])),
        "random_matched_objective": float(np.mean([r.objective for r in rr])),
        "advantage": st["mean_diff"],
        "advantage_ci95": st["ci95"],
        "advantage_excludes_zero": st["excludes_zero"],
        "learned_action_probs": probs.tolist(),
        "per_seed_learned": [r.objective for r in lr],
        "per_seed_random": [r.objective for r in rr],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--policy-dir", type=Path, default=ROOT / "results/policies")
    ap.add_argument("--out", type=Path, default=Path("results/l5_time_shuffle.json"))
    args = ap.parse_args()

    real = {s: build_world(EXP001.world, seed=s) for s in args.seeds}
    shuf = {s: time_shuffled_world(real[s], seed=s) for s in args.seeds}

    # Structural checks: the shuffle must change only which key each query targets.
    for s in args.seeds:
        rw = [(e.t, e.key_id, e.value.tobytes()) for e in real[s].events if e.kind == "WRITE"]
        sw = [(e.t, e.key_id, e.value.tobytes()) for e in shuf[s].events if e.kind == "WRITE"]
        assert rw == sw, "shuffle altered the write stream"
        assert ([e.t for e in real[s].events if e.kind == "QUERY"]
                == [e.t for e in shuf[s].events if e.kind == "QUERY"]), "shuffle altered query timing"

    print("== L5 time-shuffled control ==")
    print("residual past->future query correlation:")
    for s in args.seeds:
        print(f"  seed {s}: real {residual_query_correlation(real[s]):+.3f}  "
              f"shuffled {residual_query_correlation(shuf[s]):+.3f}")

    real_res = advantage(real, args.seeds, "real", args.policy_dir / "real_selected.json")
    shuf_res = advantage(shuf, args.seeds, "shuffled", args.policy_dir / "shuffled_selected.json")
    a_util = real_res["advantage"] - shuf_res["advantage"]

    ratio = (shuf_res["advantage"] / real_res["advantage"]) if real_res["advantage"] > 0 else float("inf")
    passed = bool(ratio <= THRESHOLD or not shuf_res["advantage_excludes_zero"])

    out = {
        "check": "L5",
        "name": "time_shuffled_control",
        "threshold_ratio": THRESHOLD,
        "seeds": args.seeds,
        "real": real_res,
        "shuffled": shuf_res,
        "ratio_shuffled_over_real": ratio,
        "utility_attributable_advantage": a_util,
        "passed": passed,
        "source_tree_sha256": source_tree_sha256(ROOT),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\nreal     advantage {real_res['advantage']:+.4f} "
          f"[{real_res['advantage_ci95'][0]:+.4f}, {real_res['advantage_ci95'][1]:+.4f}]")
    print(f"shuffled advantage {shuf_res['advantage']:+.4f} "
          f"[{shuf_res['advantage_ci95'][0]:+.4f}, {shuf_res['advantage_ci95'][1]:+.4f}]")
    print(f"ratio {ratio:.3f}  (threshold {THRESHOLD})")
    print(f"utility-attributable advantage A_util = real - shuffled = {a_util:+.4f}")
    print(f"L5: {'PASSED' if passed else 'FAILED'} -> {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
