#!/usr/bin/env python3
"""Runtime leakage audit. Writes results/leakage_audit.json.

Complements tests/test_leakage.py: the tests assert structural properties, this
audit measures the information-theoretic ones on the configuration actually
being run, and its verdict is embedded in every run manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plasticity_routing import features as F  # noqa: E402
from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import EXP001  # noqa: E402
from plasticity_routing.metrics import first_encounter_class_dependence  # noqa: E402
from plasticity_routing.routers import (  # noqa: E402
    HeuristicRouter, LearnedRouter, OracleRouter, constant_routers,
)
from plasticity_routing.world import HIDDEN_FIELDS, build_world  # noqa: E402


def audit(seeds: list[int], router_factories: dict) -> dict:
    wcfg, scfg, ccfg = EXP001.world, EXP001.substrate, EXP001.cost
    checks: list[dict] = []

    # L1 -- whitelist purity
    forbidden_present = [n for n in F.FORBIDDEN_SOURCES if n in F.FEATURE_NAMES]
    hidden_present = [n for n in HIDDEN_FIELDS if n in F.FEATURE_NAMES]
    checks.append({"id": "L1", "name": "feature_whitelist_purity",
                   "passed": not forbidden_present and not hidden_present,
                   "detail": {"forbidden_in_whitelist": forbidden_present,
                              "hidden_in_whitelist": hidden_present,
                              "n_features": len(F.FEATURE_NAMES)}})

    # L2 -- permuting hidden labels must not change any legal decision
    l2 = []
    for name, make in router_factories.items():
        for sd in seeds[:2]:
            base = build_world(wcfg, seed=sd)
            perm = build_world(wcfg, seed=sd)
            remap = {0: 2, 1: 3, 2: 0, 3: 1}
            for e in perm.events:
                e.hidden_class = remap[e.hidden_class]
                e.ideal_action = remap[e.ideal_action]
            a = rollout(base, make(), ccfg, scfg, seed=sd, keep_decisions=True)
            b = rollout(perm, make(), ccfg, scfg, seed=sd, keep_decisions=True)
            l2.append({"router": name, "seed": sd,
                       "identical": [d.action for d in a.decisions] == [d.action for d in b.decisions]})
    checks.append({"id": "L2", "name": "hidden_label_permutation_invariance",
                   "passed": all(x["identical"] for x in l2), "detail": l2})

    # L4 -- truncating the future must not change any past decision
    l4 = []
    cut = wcfg.lifetime // 2
    for name, make in router_factories.items():
        for sd in seeds[:2]:
            full = build_world(wcfg, seed=sd)
            trunc = build_world(wcfg, seed=sd)
            trunc.events = [e for e in trunc.events if e.t < cut]
            a = rollout(full, make(), ccfg, scfg, seed=sd, keep_decisions=True)
            b = rollout(trunc, make(), ccfg, scfg, seed=sd, keep_decisions=True)
            l4.append({"router": name, "seed": sd,
                       "prefix_identical": [(d.t, d.action) for d in a.decisions if d.t < cut]
                                           == [(d.t, d.action) for d in b.decisions if d.t < cut]})
    checks.append({"id": "L4", "name": "causal_ordering_future_blindness",
                   "passed": all(x["prefix_identical"] for x in l4), "detail": l4})

    # L6 -- legal routers must never be handed privileged data
    l6 = []
    for name, make in router_factories.items():
        r = make()
        seen_priv: list = []
        original = r.act

        def spy(feats, rng, privileged=None, _o=original, _s=seen_priv):
            _s.append(privileged)
            return _o(feats, rng, privileged)

        r.act = spy  # type: ignore[method-assign]
        rollout(build_world(wcfg, seed=seeds[0]), r, ccfg, scfg, seed=seeds[0])
        l6.append({"router": name, "declares_privileged_fields": list(r.privileged_fields),
                   "always_none": bool(seen_priv) and all(p is None for p in seen_priv)})
    checks.append({"id": "L6", "name": "privilege_declaration",
                   "passed": all(not x["declares_privileged_fields"] and x["always_none"] for x in l6),
                   "detail": l6})

    # L3 -- first-encounter independence for every legal router
    l3 = {}
    for name, make in router_factories.items():
        stats = []
        for s in seeds:
            w = build_world(wcfg, seed=s)
            res = rollout(w, make(), ccfg, scfg, seed=s)
            stats.append(first_encounter_class_dependence(res.first_encounter_actions))
        l3[name] = {"min_p_value": min(x["p_value"] for x in stats),
                    "mean_mi_bits": sum(x["mutual_information_bits"] for x in stats) / len(stats),
                    "per_seed": stats}
    checks.append({"id": "L3", "name": "first_encounter_class_independence",
                   "passed": all(v["min_p_value"] > 0.05 for v in l3.values()),
                   "detail": l3})

    # L3-canary -- the test must be able to fail
    canary = []
    for s in seeds:
        w = build_world(wcfg, seed=s)
        res = rollout(w, OracleRouter(EXP001.oracle_mapping), ccfg, scfg, seed=s)
        canary.append(first_encounter_class_dependence(res.first_encounter_actions))
    checks.append({"id": "L3-canary", "name": "detector_catches_known_leak",
                   "passed": all(c["p_value"] < 0.05 for c in canary),
                   "detail": {"oracle_p_values": [c["p_value"] for c in canary],
                              "oracle_mi_bits": [c["mutual_information_bits"] for c in canary]}})

    # L7 -- audit probes must not perturb the run
    l7 = []
    for name, make in router_factories.items():
        for s in seeds[:1]:
            w = build_world(wcfg, seed=s)
            a = rollout(w, make(), ccfg, scfg, seed=s, audit_every=100, keep_decisions=True)
            b = rollout(w, make(), ccfg, scfg, seed=s, audit_every=0, keep_decisions=True)
            l7.append({"router": name,
                       "actions_identical": [d.action for d in a.decisions] == [d.action for d in b.decisions],
                       "utility_identical": abs(a.task_utility - b.task_utility) < 1e-12,
                       "writes_identical": a.ledger["write_elements"] == b.ledger["write_elements"]})
    checks.append({"id": "L7", "name": "probe_non_interference",
                   "passed": all(x["actions_identical"] and x["utility_identical"]
                                 and x["writes_identical"] for x in l7),
                   "detail": l7})

    return {"seeds": seeds, "checks": checks, "passed": all(c["passed"] for c in checks)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    ap.add_argument("--out", type=Path, default=Path("results/leakage_audit.json"))
    args = ap.parse_args()

    factories = {name: (lambda r=r: r) for name, r in constant_routers().items()}
    factories["HEURISTIC"] = HeuristicRouter
    factories["LEARNED_UNTRAINED"] = lambda: LearnedRouter(seed=0)

    out = audit(args.seeds, factories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")

    for c in out["checks"]:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']:<10} {c['name']}")
    print(f"\nleakage audit: {'PASSED' if out['passed'] else 'FAILED'} -> {args.out}")
    sys.exit(0 if out["passed"] else 1)


if __name__ == "__main__":
    main()
