#!/usr/bin/env python3
"""EXP-000 -- harness smoke test and development calibration.

**This is not evidence.** Every manifest it writes is classified
`DEV_CALIBRATION`. Its job is to show the machinery is mechanically valid:
budgets are matched and enforced, leakage tests pass, the controls fail for the
reasons the benchmark design predicts, and the statistics run end to end.

The learned router is trained on development seeds and evaluated on the same
seeds here, which is exactly why this cannot be confirmatory. EXP-001 trains on
development seeds and evaluates on held-out confirmatory seeds; it is gated on
issue #1.

Order of inspection is deliberate and follows the sibling track's protocol:
mechanical validity first, comparative metrics last.
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

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import DEV_SEEDS, EXP001, POLICY_SEEDS  # noqa: E402
from plasticity_routing.manifest import build_manifest, write_manifest  # noqa: E402
from plasticity_routing.metrics import (  # noqa: E402
    confusion_table, first_encounter_class_dependence, paired_bootstrap, summarize,
)
from plasticity_routing.routers import (  # noqa: E402
    ExtendedHeuristicRouter, HeuristicRouter, OracleRouter, PrivilegedTaskIdRouter,
    RandomMatchedRouter, constant_routers,
)
from plasticity_routing.substrates import ACTION_NAMES  # noqa: E402
from plasticity_routing.train import load_policy  # noqa: E402
from plasticity_routing.world import CLASS_NAMES, build_world  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from audit_leakage import audit as leakage_audit  # noqa: E402


def _matched_heuristic() -> ExtendedHeuristicRouter:
    """The best fixed rule found under a search budget matched to the ES budget.

    This, not the three-parameter grid argmax, is the primary comparator for H1:
    a comparator searched a hundred times less hard is not a fair test.
    """
    from plasticity_routing.substrates import ACTION_NAMES

    path = ROOT / "results/heuristic_matched_search.json"
    if not path.exists():
        raise SystemExit("run scripts/search_heuristic_matched.py (+ merge) first")
    prm = dict(json.loads(path.read_text())["best_params"])
    inv = {v: k for k, v in ACTION_NAMES.items()}
    prm["recurrent_default"] = inv[prm["recurrent_default"]]
    prm["novel_action"] = inv[prm["novel_action"]]
    return ExtendedHeuristicRouter(**prm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-seeds", type=int, nargs="+", default=list(DEV_SEEDS))
    ap.add_argument("--policy", type=Path, default=ROOT / "results/policies/real_selected.json")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-leakage", action="store_true", help="development convenience only")
    ap.add_argument("--proceed-on-audit-failure", action="store_true",
                    help="continue past a failing leakage audit and stamp every manifest with "
                         "an invalidation reason. The numbers become labelled diagnostics, not "
                         "evidence, and scripts/validate_runs.py will reject them.")
    args = ap.parse_args()

    wcfg, scfg, ccfg = EXP001.world, EXP001.substrate, EXP001.cost
    tcfg = EXP001.train
    seeds = args.dev_seeds
    worlds = {s: build_world(wcfg, seed=s) for s in seeds}

    # ---- 1. leakage audit BEFORE any comparative metric is inspected -----
    print("== 1. leakage audit ==")
    if args.skip_leakage:
        audit = {"passed": False, "checks": [], "note": "SKIPPED -- run is not interpretable"}
        print("  SKIPPED (development convenience; manifests will be invalid)")
    else:
        from plasticity_routing.routers import LearnedRouter
        factories = {n: (lambda r=r: r) for n, r in constant_routers().items()}
        factories["HEURISTIC"] = HeuristicRouter
        factories["LEARNED_UNTRAINED"] = lambda: LearnedRouter(seed=0)
        audit = leakage_audit(seeds, factories)
        for c in audit["checks"]:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']:<10} {c['name']}")
        if not audit["passed"]:
            failed = [c["id"] for c in audit["checks"] if not c["passed"]]
            if not args.proceed_on_audit_failure:
                print("\nLEAKAGE AUDIT FAILED -- stopping before any comparative metric "
                      "is inspected.")
                print(f"  failing checks: {failed}")
                print("  pass --proceed-on-audit-failure to record labelled diagnostics anyway.")
                sys.exit(1)
            print(f"\nLEAKAGE AUDIT FAILED ({failed}) -- proceeding under "
                  "--proceed-on-audit-failure.")
            print("  Every manifest will be stamped invalid. These numbers are diagnostics, "
                  "not evidence.")

    # ---- 2. train the learned router on development seeds ---------------
    print("\n== 2. load the selected learned router ==")
    if not args.policy.exists():
        raise SystemExit(f"no selected policy at {args.policy}; run scripts/train_policy.py "
                         "for every POLICY_SEED then scripts/select_policy.py")
    learned, history = load_policy(args.policy)
    learned.greedy = True
    print(f"  {args.policy.name}: policy seed {history['policy_seed']}, "
          f"{history['generations']} generations, dev objective {history['dev_objective']:.4f}")
    print(f"  selection rule: best development objective over POLICY_SEEDS={list(POLICY_SEEDS)}")

    # ---- 3. build the arm set -------------------------------------------
    arms: dict[str, object] = dict(constant_routers())
    arms["HEURISTIC"] = HeuristicRouter()
    arms["HEURISTIC_EXT"] = _matched_heuristic()
    arms["LEARNED"] = learned

    results: dict[str, list] = {name: [] for name in arms}
    for name, router in arms.items():
        for s in seeds:
            results[name].append(rollout(worlds[s], router, ccfg, scfg, seed=s))

    # A5: random routing matched to LEARNED's realised action histogram.
    probs = np.mean([r.action_probs for r in results["LEARNED"]], axis=0)
    arms["RANDOM_MATCHED"] = RandomMatchedRouter(probs)
    results["RANDOM_MATCHED"] = [
        rollout(worlds[s], arms["RANDOM_MATCHED"], ccfg, scfg, seed=s) for s in seeds
    ]

    # Ceiling and privileged probes -- never claim-eligible.
    arms["ORACLE"] = OracleRouter(EXP001.oracle_mapping)
    results["ORACLE"] = [rollout(worlds[s], arms["ORACLE"], ccfg, scfg, seed=s) for s in seeds]
    arms["DESIGNED_MAPPING"] = OracleRouter(EXP001.designed_mapping)
    results["DESIGNED_MAPPING"] = [
        rollout(worlds[s], arms["DESIGNED_MAPPING"], ccfg, scfg, seed=s) for s in seeds
    ]
    arms["PRIVILEGED_TASKID"] = PrivilegedTaskIdRouter(HeuristicRouter())
    results["PRIVILEGED_TASKID"] = [
        rollout(worlds[s], arms["PRIVILEGED_TASKID"], ccfg, scfg, seed=s) for s in seeds
    ]

    # C1: capacity-matched control. The learned router has three substrates
    # available; a depth-agnostic control has one. Give the strongest
    # single-depth control the *element* equivalent of all three: the two
    # parametric matrices occupy 2*key_dim*value_dim elements, and an episodic
    # entry occupies key_dim + value_dim (it must store its key to be
    # retrievable), so the parametric footprint buys this many extra entries.
    entry_elements = scfg.key_dim + scfg.value_dim
    extra_items = (2 * scfg.key_dim * scfg.value_dim) // entry_elements
    cap_scfg = replace(scfg, episodic_capacity=scfg.episodic_capacity + extra_items)
    print(f"\n  C1 capacity match: episodic {scfg.episodic_capacity} -> "
          f"{cap_scfg.episodic_capacity} entries (+{extra_items} = parametric footprint / "
          f"{entry_elements} elements per entry)")
    results["CAPACITY_MATCHED_EPISODIC"] = [
        rollout(worlds[s], constant_routers()["ALL_EPISODIC"], ccfg, cap_scfg, seed=s) for s in seeds
    ]

    # ---- 4. mechanical validity -----------------------------------------
    print("\n== 3. mechanical validity ==")
    ceiling = ccfg.write_element_ceiling
    for name, rs in results.items():
        w = max(r.ledger["write_elements"] for r in rs)
        ok = w <= ceiling
        print(f"  {'ok ' if ok else 'BAD'} {name:<26} max writes {w:>9,} / {ceiling:,} "
              f"({w / ceiling:5.1%})  forced-ignores {max(r.ledger['forced_ignores_budget'] for r in rs):>5}")

    # ---- 5. comparative metrics (inspected last) ------------------------
    print("\n== 4. comparative metrics (DEV_CALIBRATION -- not evidence) ==")
    print(f"{'arm':<26} {'legal':>5} {'obj':>7} {'util':>6} {'forget':>7} "
          f"{'w_norm':>7} {'s_norm':>7} {'c_norm':>7} {'routerFLOP%':>11}")
    summaries = {}
    for name, rs in results.items():
        s = summarize(rs)
        summaries[name] = s
        led = rs[0].ledger
        rc = np.mean([r.ledger["router_compute_elements"] / max(1, r.ledger["total_compute_elements"])
                      for r in rs])
        print(f"{name:<26} {'Y' if s['legal'] else 'N':>5} {s['objective']['mean']:>7.3f} "
              f"{s['task_utility']['mean']:>6.3f} {s['forgetting']['mean']:>7.3f} "
              f"{np.mean([r.ledger['normalized']['write'] for r in rs]):>7.3f} "
              f"{np.mean([r.ledger['normalized']['storage'] for r in rs]):>7.3f} "
              f"{np.mean([r.ledger['normalized']['compute'] for r in rs]):>7.3f} "
              f"{rc:>10.2%}")

    print("\naction distribution (fraction of write events):")
    print(f"{'arm':<26} " + " ".join(f"{ACTION_NAMES[a]:>9}" for a in range(4)))
    for name, rs in results.items():
        p = np.mean([r.action_probs for r in rs], axis=0)
        print(f"{name:<26} " + " ".join(f"{x:>9.3f}" for x in p))

    print("\nper-class recall utility:")
    print(f"{'arm':<26} " + " ".join(f"{c:>9}" for c in CLASS_NAMES.values()))
    for name, rs in results.items():
        row = summaries[name]["utility_by_class_mean"]
        print(f"{name:<26} " + " ".join(
            f"{row[c]:>9.3f}" if row[c] == row[c] else f"{'n/a':>9}" for c in CLASS_NAMES.values()))

    # ---- 6. preregistered contrasts -------------------------------------
    print("\n== 5. preregistered contrasts (paired bootstrap, 95% CI) ==")
    contrasts = {}
    def contrast(a: str, b: str, label: str):
        if a in summaries and b in summaries:
            st = paired_bootstrap(summaries[a]["objective"]["values"],
                                  summaries[b]["objective"]["values"])
            contrasts[label] = st
            flag = "excludes 0" if st["excludes_zero"] else "includes 0"
            print(f"  {label:<44} {st['mean_diff']:+.4f}  "
                  f"[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]  {flag}")

    contrast("LEARNED", "HEURISTIC_EXT", "K1  LEARNED - HEURISTIC_EXT (matched budget)")
    contrast("LEARNED", "HEURISTIC", "    LEARNED - HEURISTIC (3-param grid)")
    contrast("LEARNED", "RANDOM_MATCHED", "K2  LEARNED - RANDOM_MATCHED")
    best_sd = max(("ALL_EPISODIC", "ALL_FAST", "ALL_SLOW"),
                  key=lambda k: summaries[k]["objective"]["mean"])
    contrast("LEARNED", best_sd, f"K4  LEARNED - {best_sd} (best single depth)")
    contrast("LEARNED", "CAPACITY_MATCHED_EPISODIC", "K7  LEARNED - CAPACITY_MATCHED")
    contrast("PRIVILEGED_TASKID", "HEURISTIC", "    PRIVILEGED_TASKID - HEURISTIC")
    contrast("ORACLE", "LEARNED", "    ORACLE - LEARNED (headroom)")

    # Compute-attribution check: how large must the compute penalty be before
    # the learned router's advantage over the heuristic disappears?
    def obj_at(lam_c: float, name: str) -> list[float]:
        out = []
        for r in results[name]:
            base = r.objective + ccfg.lam_compute * r.ledger["normalized"]["compute"]
            out.append(base - lam_c * r.ledger["normalized"]["compute"])
        return out

    if np.mean(obj_at(ccfg.lam_compute, "LEARNED")) <= np.mean(obj_at(ccfg.lam_compute, "HEURISTIC")):
        breakeven = None
        print("\n  compute break-even lambda_compute: n/a -- LEARNED does not lead HEURISTIC at "
              f"the configured lambda_compute ({ccfg.lam_compute:.4g}), so there is no advantage "
              "for extra compute to explain.")
    else:
        lo, hi = ccfg.lam_compute, 1e6
        for _ in range(60):
            mid = (lo + hi) / 2
            if np.mean(obj_at(mid, "LEARNED")) > np.mean(obj_at(mid, "HEURISTIC")):
                lo = mid
            else:
                hi = mid
        breakeven = lo
        print(f"\n  compute break-even lambda_compute (LEARNED leads HEURISTIC below this): {lo:.4g}")
        print(f"  configured lambda_compute: {ccfg.lam_compute:.4g}  "
              f"-> margin x{lo / ccfg.lam_compute:.1f}")
    print(f"  router decision compute as share of total: "
          f"{np.mean([r.ledger['router_compute_elements'] / max(1, r.ledger['total_compute_elements']) for r in results['LEARNED']]):.2%} "
          f"(LEARNED) vs "
          f"{np.mean([r.ledger['router_compute_elements'] / max(1, r.ledger['total_compute_elements']) for r in results['HEURISTIC']]):.2%} "
          f"(HEURISTIC)")

    # ---- 7. manifests ----------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, rs in results.items():
        for r in rs:
            m = build_manifest(
                root=ROOT, classification="DEV_CALIBRATION", arm=name, result=r,
                world_cfg=wcfg, sub_cfg=scfg, cost_cfg=ccfg, train_cfg=tcfg,
                dev_seeds=seeds, leakage=audit,
                notes="EXP-000 development calibration. Trained and evaluated on development "
                      "seeds; not evidence. See experiments/EXP-000-RESULT.md.",
            )
            if not audit.get("passed", False):
                failing = [c["id"] for c in audit.get("checks", []) if not c["passed"]] or ["unknown"]
                m["invalidation_reasons"] = [f"leakage_audit_failed:{c}" for c in failing]
            m["first_encounter_leakage"] = first_encounter_class_dependence(r.first_encounter_actions)
            m["diagnostics"]["confusion_hidden_class_by_action"] = confusion_table(r.confusion)
            write_manifest(args.out / f"run_{name}_seed{r.seed}.json", m)
            n += 1

    (args.out / "exp000_summary.json").write_text(json.dumps({
        "classification": "DEV_CALIBRATION",
        "dev_seeds": seeds,
        "oracle_mapping": list(EXP001.oracle_mapping),
        "designed_mapping": list(EXP001.designed_mapping),
        "summaries": summaries,
        "contrasts": contrasts,
        "compute_breakeven_lambda": breakeven,
        "policy_meta": history,
        "leakage_passed": audit["passed"],
    }, indent=2, default=str) + "\n")
    print(f"\nwrote {n} manifests + exp000_summary.json to {args.out}/")


if __name__ == "__main__":
    main()
