#!/usr/bin/env python3
"""Development-only benchmark calibration for SDW-1.

Calibration is deliberately blind to the method under test. Only the
depth-agnostic controls, the fixed heuristic, and class-conditional ORACLE
mappings are run. `LearnedRouter` is never executed here, so the benchmark
cannot be tuned to make the proposed method win.

Stage 1 sweeps world/substrate parameters cheaply, scoring each candidate by
the headroom of the *intended* mapping over the best single-depth control.
Stage 2 takes the leading candidates and replaces the assumed mapping with a
real upper bound: an exhaustive search over all 4^4 class-conditional mappings.

A configuration is admissible when all of:
  C1  ORACLE objective >= --floor            world is solvable at all
  C2  no arm's task utility > --ceiling      not trivially easy
  C3  ORACLE - best single-depth >= --margin depth genuinely stratifies
  C4  ORACLE - HEURISTIC       >= --margin   benchmark can discriminate a
                                             learned policy from a fixed rule
C3 and C4 are *discriminability* criteria: they ensure the experiment is
capable of separating hypotheses. They are computed without ever running the
learned router, so satisfying them cannot bias the outcome in its favour.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.ledger import CostConfig  # noqa: E402
from plasticity_routing.routers import (  # noqa: E402
    INTENDED_MAPPING, HeuristicRouter, OracleRouter, constant_routers, is_bijective,
    search_best_mapping,
)
from plasticity_routing.substrates import SubstrateConfig  # noqa: E402
from plasticity_routing.world import WorldConfig, build_world  # noqa: E402

SINGLE_DEPTH = ("ALL_EPISODIC", "ALL_FAST", "ALL_SLOW")


def make_configs(cap, decay, n_local, key_dim, one_off_max, share):
    rest = 1.0 - share
    prior = (share / 2, share / 2, rest * 0.65, rest * 0.35)
    wcfg = replace(WorldConfig(), class_prior=prior, n_local_slots=n_local,
                   key_dim=key_dim, one_off_delay_max=one_off_max)
    scfg = replace(SubstrateConfig(), episodic_capacity=cap, fast_decay=decay, key_dim=key_dim)
    ccfg = replace(CostConfig(), key_dim=key_dim)
    return wcfg, scfg, ccfg


def score_arms(worlds, arms, cost_cfg, sub_cfg, seeds):
    out = {}
    for name, router in arms.items():
        objs, utils, forgets = [], [], []
        for s in seeds:
            r = rollout(worlds[s], router, cost_cfg, sub_cfg, seed=s)
            objs.append(r.objective); utils.append(r.task_utility); forgets.append(r.forgetting)
        out[name] = {"objective": float(np.mean(objs)), "task_utility": float(np.mean(utils)),
                     "forgetting": float(np.mean(forgets))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--floor", type=float, default=0.45)
    ap.add_argument("--ceiling", type=float, default=0.95)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--top-k", type=int, default=4, help="stage-2 candidates")
    ap.add_argument("--out", type=Path, default=Path("results/calibration.json"))
    args = ap.parse_args()

    grid = {
        "episodic_capacity": [12, 20],
        "fast_decay": [0.988, 0.994],
        "n_local_slots": [30, 44],
        "key_dim": [96],
        "one_off_delay_max": [30, 80],
        "noise_one_off_share": [0.55],
    }
    stage1_seeds = args.dev_seeds[:2]

    print("== stage 1: coarse sweep, coordinate-ascent mapping search ==")
    print(f"{'cap':>4} {'decay':>6} {'nloc':>5} {'kdim':>5} {'ood':>4} | "
          f"{'ORACLE':>7} {'bestSD':>7} {'HEUR':>7} | {'h_sd':>6} {'h_hr':>6} {'map':>13} bij")
    stage1 = []
    for params in itertools.product(*grid.values()):
        cap, decay, n_local, key_dim, one_off_max, share = params
        wcfg, scfg, ccfg = make_configs(*params)
        worlds = {s: build_world(wcfg, seed=s) for s in stage1_seeds}
        best_map, _, _ = search_best_mapping(worlds, ccfg, scfg, stage1_seeds, method="coordinate")
        arms = dict(constant_routers())
        arms["HEURISTIC"] = HeuristicRouter()
        arms["ORACLE"] = OracleRouter(best_map)
        res = score_arms(worlds, arms, ccfg, scfg, stage1_seeds)
        best_sd = max(SINGLE_DEPTH, key=lambda k: res[k]["objective"])
        h_sd = res["ORACLE"]["objective"] - res[best_sd]["objective"]
        h_hr = res["ORACLE"]["objective"] - res["HEURISTIC"]["objective"]
        bij = is_bijective(best_map)
        stage1.append({"params": params, "res": res, "best_single_depth": best_sd,
                       "mapping": list(best_map), "bijective": bij,
                       "h_sd": h_sd, "h_heur": h_hr,
                       "score": (1 if bij else 0, min(h_sd, h_hr))})
        print(f"{cap:>4} {decay:>6.3f} {n_local:>5} {key_dim:>5} {one_off_max:>4} | "
              f"{res['ORACLE']['objective']:>7.3f} {res[best_sd]['objective']:>7.3f} "
              f"{res['HEURISTIC']['objective']:>7.3f} | {h_sd:>6.3f} {h_hr:>6.3f} "
              f"{str(list(best_map)):>13} {'Y' if bij else '.'}")

    stage1.sort(key=lambda r: (-r["score"][0], -r["score"][1]))
    candidates = stage1[: args.top_k]

    print(f"\n== stage 2: exhaustive {4 ** 4}-mapping oracle search on top {len(candidates)} ==")
    rows = []
    for cand in candidates:
        cap, decay, n_local, key_dim, one_off_max, share = cand["params"]
        wcfg, scfg, ccfg = make_configs(*cand["params"])
        worlds = {s: build_world(wcfg, seed=s) for s in args.dev_seeds}
        best_map, _, ranked = search_best_mapping(worlds, ccfg, scfg, args.dev_seeds, method="exhaustive")
        # C6 robustness: the same mapping must remain optimal at a neighbouring
        # episodic capacity, so the ground truth is not a capacity coincidence.
        alt_scfg = replace(scfg, episodic_capacity=int(round(cap * 1.5)))
        alt_map, _, _ = search_best_mapping(worlds, ccfg, alt_scfg, args.dev_seeds, method="coordinate")
        robust = tuple(alt_map) == tuple(best_map)

        arms = dict(constant_routers())
        arms["HEURISTIC"] = HeuristicRouter()
        arms["ORACLE"] = OracleRouter(best_map)
        arms["INTENDED"] = OracleRouter(INTENDED_MAPPING)
        res = score_arms(worlds, arms, ccfg, scfg, args.dev_seeds)

        best_sd = max(SINGLE_DEPTH, key=lambda k: res[k]["objective"])
        h_sd = res["ORACLE"]["objective"] - res[best_sd]["objective"]
        h_heur = res["ORACLE"]["objective"] - res["HEURISTIC"]["objective"]
        max_util = max(v["task_utility"] for v in res.values())
        admissible = (res["ORACLE"]["objective"] >= args.floor and max_util <= args.ceiling
                      and h_sd >= args.margin and h_heur >= args.margin
                      and is_bijective(best_map) and robust)
        rows.append({
            "episodic_capacity": cap, "fast_decay": decay, "n_local_slots": n_local,
            "key_dim": key_dim, "one_off_delay_max": one_off_max, "noise_one_off_share": share,
            "world_config": {k: v for k, v in wcfg.__dict__.items()},
            "substrate_config": {k: v for k, v in scfg.__dict__.items()},
            "arms": res, "best_single_depth": best_sd,
            "oracle_mapping": list(best_map),
            "oracle_matches_intended": tuple(best_map) == tuple(INTENDED_MAPPING),
            "bijective": is_bijective(best_map),
            "robust_to_capacity_change": bool(robust), "alt_capacity_mapping": list(alt_map),
            "top5_mappings": [[list(m), o] for m, o in ranked[:5]],
            "oracle_headroom_over_single_depth": h_sd,
            "oracle_headroom_over_heuristic": h_heur,
            "max_task_utility": max_util, "admissible": bool(admissible),
        })
        print(f"cap={cap} decay={decay} nloc={n_local} ood={one_off_max} | "
              f"ORACLE={res['ORACLE']['objective']:.3f} map={list(best_map)} "
              f"bij={is_bijective(best_map)} robust={robust} | "
              f"h_sd={h_sd:.3f} h_heur={h_heur:.3f} {'ADMISSIBLE' if admissible else 'rejected'}")

    rows.sort(key=lambda r: (-r["admissible"],
                             -min(r["oracle_headroom_over_single_depth"], r["oracle_headroom_over_heuristic"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"dev_seeds": args.dev_seeds,
         "criteria": {"floor": args.floor, "ceiling": args.ceiling, "margin": args.margin},
         "stage1": [{"params": list(r["params"]), "mapping": r["mapping"],
                     "bijective": r["bijective"], "h_sd": r["h_sd"], "h_heur": r["h_heur"]} for r in stage1],
         "stage2": rows}, indent=2, default=str) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
