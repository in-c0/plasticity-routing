#!/usr/bin/env python3
"""Durable-write discoverability diagnostic. Development seeds only.

EXP-000 found that the learned policy never selects `SLOW`, while the calibrated
heuristic selects it 13.2% of the time and scores *higher*. Two explanations are
possible and they have opposite consequences:

  (a) `SLOW` is genuinely not worth using for a decision-time-legal policy, and
      the ORACLE's durable consolidation only pays because ORACLE knows the
      hidden class. Then K1 is a fact about the problem.

  (b) `SLOW` is worth using but the frozen ES budget cannot find it, because its
      cost is immediate and its payoff arrives hundreds of steps later. Then K1
      is partly a fact about our optimiser, and reporting it as a fact about
      learned routing would be wrong.

The diagnostic distinguishes them by initialising the search *inside* the region
(b) says is unreachable, and seeing whether ES stays there:

  1. `random`      -- ES from the frozen initialisation (the EXP-000 condition).
  2. `slow_primed` -- ES from a policy whose output bias strongly favours SLOW.
  3. `cloned`      -- ES from a policy behaviour-cloned onto the calibrated
                      heuristic, which does use SLOW and does score 0.405.

If `slow_primed` and `cloned` retain SLOW and end above `random`, explanation (b)
holds. If they drift away from SLOW and converge to the same objective as
`random`, explanation (a) holds and the frozen budget is not the limitation.

This is a development-only diagnostic. It does not touch confirmatory seeds and
its outcome may not be used to retune anything after confirmatory results exist.
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
from plasticity_routing.routers import HeuristicRouter, LearnedRouter  # noqa: E402
from plasticity_routing.substrates import ACTION_NAMES, N_ACTIONS, SLOW  # noqa: E402
from plasticity_routing.train import evaluate_policy, train_router_es  # noqa: E402
from plasticity_routing.world import build_world  # noqa: E402


def clone_heuristic(worlds, seeds, hidden: int, policy_seed: int, epochs: int = 400):
    """Behaviour-clone the calibrated heuristic into the policy network."""
    heur = HeuristicRouter()
    X, Y = [], []
    for s in seeds:
        res = rollout(worlds[s], heur, EXP001.cost, EXP001.substrate, seed=s, keep_decisions=True)
        for d in res.decisions:
            X.append(d.feats)
            Y.append(d.action)
    X = np.stack(X)
    Y = np.asarray(Y)

    net = LearnedRouter(hidden=hidden, seed=policy_seed)
    rng = np.random.default_rng(policy_seed)
    lr = 0.05
    for ep in range(epochs):
        idx = rng.permutation(len(X))[:512]
        xb, yb = X[idx], Y[idx]
        h = np.tanh(xb @ net.p.W1.T + net.p.b1)
        logits = h @ net.p.W2.T + net.p.b2
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        probs = e / e.sum(axis=1, keepdims=True)
        d_logits = probs.copy()
        d_logits[np.arange(len(yb)), yb] -= 1.0
        d_logits /= len(yb)
        gW2 = d_logits.T @ h
        gb2 = d_logits.sum(axis=0)
        dh = (d_logits @ net.p.W2) * (1 - h ** 2)
        gW1 = dh.T @ xb
        gb1 = dh.sum(axis=0)
        net.p.W1 -= lr * gW1
        net.p.b1 -= lr * gb1
        net.p.W2 -= lr * gW2
        net.p.b2 -= lr * gb2

    net.greedy = True
    acc = float((np.argmax(np.tanh(X @ net.p.W1.T + net.p.b1) @ net.p.W2.T + net.p.b2, axis=1) == Y).mean())
    return net, acc


def profile(router, worlds, seeds) -> dict:
    router.greedy = True
    rs = [rollout(worlds[s], router, EXP001.cost, EXP001.substrate, seed=s) for s in seeds]
    probs = np.mean([r.action_probs for r in rs], axis=0)
    return {
        "objective": float(np.mean([r.objective for r in rs])),
        "task_utility": float(np.mean([r.task_utility for r in rs])),
        "action_probs": {ACTION_NAMES[a]: float(probs[a]) for a in range(N_ACTIONS)},
        "slow_share": float(probs[SLOW]),
        "utility_by_class": {k: float(np.nanmean([r.utility_by_class[k] for r in rs]))
                             for k in rs[0].utility_by_class},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--out", type=Path, default=Path("results/slow_discoverability.json"))
    ap.add_argument("--conditions", nargs="+", default=["primed", "cloned"],
                    choices=["random", "primed", "cloned"],
                    help="random-init seeds are covered by scripts/train_policy.py; "
                         "this script defaults to the two initialisation probes")
    ap.add_argument("--random-policies", type=Path, default=Path("results/policies"))
    args = ap.parse_args()

    cfg = EXP001.train
    worlds = {s: build_world(EXP001.world, seed=s) for s in args.seeds}
    rows = {}

    heur = HeuristicRouter()
    rows["heuristic_reference"] = profile(heur, worlds, args.seeds)
    print(f"heuristic reference: obj {rows['heuristic_reference']['objective']:.4f} "
          f"SLOW share {rows['heuristic_reference']['slow_share']:.3f}")

    conditions = []
    if "random" in args.conditions:
        for ps in (0, 1, 2):
            conditions.append((f"random_seed{ps}", dict(policy_seed=ps)))
    bias = np.zeros(N_ACTIONS)
    bias[SLOW] = 3.0
    if "primed" in args.conditions:
        conditions.append(("slow_primed", dict(policy_seed=0, init_bias=bias)))

    cloned, acc = clone_heuristic(worlds, args.seeds, cfg.hidden, policy_seed=0)
    rows["cloned_init_before_es"] = profile(cloned, worlds, args.seeds) | {"clone_accuracy": acc}
    print(f"behaviour clone of heuristic: accuracy {acc:.3f}  "
          f"obj {rows['cloned_init_before_es']['objective']:.4f}  "
          f"SLOW share {rows['cloned_init_before_es']['slow_share']:.3f}")
    if "cloned" in args.conditions:
        conditions.append(("cloned", dict(policy_seed=0, init_params=cloned.p.flat().copy())))

    for name, kw in conditions:
        print(f"\n-- ES from {name} --")
        router, hist = train_router_es(
            EXP001.world, EXP001.substrate, EXP001.cost, args.seeds, cfg,
            verbose=True, worlds=worlds, **kw,
        )
        pr = profile(router, worlds, args.seeds)
        pr["initial_objective"] = hist[0]["dev_objective"]
        rows[name] = pr
        print(f"   final obj {pr['objective']:.4f}  SLOW share {pr['slow_share']:.3f}  "
              f"actions {pr['action_probs']}")

    if "random" in args.conditions:
        best_random = max(rows[f"random_seed{i}"]["objective"] for i in (0, 1, 2))
        random_source = "this run"
    else:
        # Reuse the cached random-init policies rather than retraining them.
        cached = sorted(args.random_policies.glob("real_seed*.json"))
        objs = [json.loads(p.read_text())["meta"]["dev_objective"] for p in cached]
        best_random = max(objs) if objs else float("-inf")
        random_source = f"cached policies ({len(objs)} seeds)"
    rows["random_init_reference"] = {"best_objective": best_random, "source": random_source}
    primed_keeps_slow = rows["slow_primed"]["slow_share"] > 0.05
    cloned_keeps_slow = rows["cloned"]["slow_share"] > 0.05
    any_beats_random = max(rows["slow_primed"]["objective"], rows["cloned"]["objective"]) > best_random + 0.005
    verdict = "optimiser_limited" if (any_beats_random and (primed_keeps_slow or cloned_keeps_slow)) \
        else "slow_reachable_from_random_init"

    out = {"seeds": args.seeds, "es_config": cfg.__dict__, "rows": rows,
           "best_random_objective": best_random,
           "primed_retains_slow": primed_keeps_slow,
           "cloned_retains_slow": cloned_keeps_slow,
           "verdict": verdict}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"\nbest random-init objective: {best_random:.4f}")
    print(f"slow_primed: obj {rows['slow_primed']['objective']:.4f} SLOW {rows['slow_primed']['slow_share']:.3f}")
    print(f"cloned     : obj {rows['cloned']['objective']:.4f} SLOW {rows['cloned']['slow_share']:.3f}")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
