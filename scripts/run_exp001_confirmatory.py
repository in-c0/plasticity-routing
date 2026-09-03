#!/usr/bin/env python3
"""EXP-001 confirmatory execution. One shot.

Amendment M. This runner exists because the only other one,
`run_exp000.py`, hard-codes `classification="DEV_CALIBRATION"` and takes seeds
from the command line: running the confirmatory arm through it would have
produced manifests describing themselves as non-evidential, on arbitrary seeds.

Guarantees enforced here, not by discipline:

*   seeds come **only** from the frozen protocol lock; there is no seed CLI;
*   development and audit seeds are refused outright;
*   the commit, source-tree hash, config hash, both policy hashes and the
    comparator hash must match the lock and `config.py` before anything runs;
*   manifests are written as `CONFIRMATORY`, to a directory separate from the
    development results;
*   a second execution is refused unless `--reproduce-only`, which recomputes
    and compares without writing;
*   the preregistered reveal order is enforced: comparative metrics are
    computed but **withheld** until leakage, manifest validation, resource
    ceilings, benchmark admissibility C1-C6, and the invalidating criteria
    K3-K6/K8 have all passed. If an invalidating gate fires, the run stops
    before the headline comparison is printed or aggregated.

Nothing scientific is decided here. Every configuration, checkpoint, comparator,
threshold, seed and statistic was frozen before this file existed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import (  # noqa: E402
    AUDIT_SEEDS, DEV_SEEDS, EXP001, MATCHED_HEURISTIC_SHA256, POLICY_SEEDS,
    SELECTED_POLICY_SHA256,
)
from plasticity_routing.manifest import (  # noqa: E402
    build_manifest, config_hash, git_sha, source_tree_sha256, write_manifest,
)
from plasticity_routing.metrics import (  # noqa: E402
    confusion_table, first_encounter_class_dependence, paired_bootstrap, summarize,
)
from plasticity_routing.routers import (  # noqa: E402
    ExtendedHeuristicRouter, HeuristicRouter, OracleRouter, PrivilegedTaskIdRouter,
    RandomMatchedRouter, constant_routers, is_bijective, search_best_mapping,
)
from plasticity_routing.substrates import ACTION_NAMES  # noqa: E402
from plasticity_routing.train import load_policy  # noqa: E402
from plasticity_routing.world import CLASS_NAMES, build_world, time_shuffled_world  # noqa: E402

from audit_leakage import audit as leakage_audit  # noqa: E402
from validate_runs import validate as validate_manifests  # noqa: E402

SINGLE_DEPTH = ("ALL_EPISODIC", "ALL_FAST", "ALL_SLOW")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Refusal(SystemExit):
    def __init__(self, msg: str):
        super().__init__(f"REFUSED: {msg}")


def preflight(lock_path: Path, out_dir: Path, reproduce_only: bool) -> dict:
    """Every check that must pass before a single rollout is executed."""
    if not lock_path.exists():
        raise Refusal(f"no protocol lock at {lock_path}; run scripts/freeze_protocol.py")
    lock = json.loads(lock_path.read_text())

    if not lock.get("frozen"):
        raise Refusal(f"protocol lock is not frozen; blockers: {lock.get('blockers')}")

    # The substantive pin is the source-tree hash. HEAD is allowed to be the
    # admissible commit *or a descendant of it*, because committing the lock
    # itself advances HEAD -- and neither `results/` nor the generated
    # `PROTOCOL-v*.md` is inside the fingerprint, so such a commit cannot change
    # what actually executes. Any real source change moves the tree hash and is
    # refused below.
    sha = git_sha(ROOT)
    admissible = lock.get("admissible_commit")
    if sha != admissible:
        import subprocess

        anc = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                              str(admissible), str(sha)], capture_output=True)
        if anc.returncode != 0:
            raise Refusal(f"HEAD {sha} is neither the admissible commit {admissible} "
                          "nor a descendant of it")

    tree = source_tree_sha256(ROOT)
    if tree != lock.get("source_tree_sha256"):
        raise Refusal("source tree has drifted from the frozen lock\n"
                      f"  locked {lock.get('source_tree_sha256')}\n  actual {tree}")

    cfg_hash = config_hash(EXP001.world, EXP001.substrate, EXP001.cost, EXP001.train)
    if cfg_hash != lock.get("config_hash"):
        raise Refusal(f"config hash {cfg_hash} != locked {lock.get('config_hash')}")

    seeds = list(lock["confirmatory_seeds"])
    if not seeds:
        raise Refusal("lock records no confirmatory seeds")
    bad_dev = sorted(set(seeds) & set(DEV_SEEDS))
    bad_audit = sorted(set(seeds) & set(AUDIT_SEEDS))
    if bad_dev or bad_audit:
        raise Refusal(f"confirmatory seeds overlap dev={bad_dev} audit={bad_audit}")

    pol_dir = ROOT / "results/policies"
    for name, expected in SELECTED_POLICY_SHA256.items():
        path = pol_dir / name
        if not path.exists():
            raise Refusal(f"missing frozen policy artefact {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise Refusal(f"{name} drifted from its pinned hash\n"
                          f"  pinned {expected}\n  actual {actual}")

    comp_path = ROOT / "results/heuristic_matched_search.json"
    if not comp_path.exists():
        raise Refusal(f"missing comparator artefact {comp_path}")
    comp_hash = sha256_file(comp_path)
    if comp_hash != MATCHED_HEURISTIC_SHA256:
        raise Refusal(f"comparator artefact drifted from its pinned hash\n"
                      f"  pinned {MATCHED_HEURISTIC_SHA256}\n  actual {comp_hash}")

    existing = sorted(out_dir.glob("run_*.json")) if out_dir.exists() else []
    if existing and not reproduce_only:
        raise Refusal(
            f"{len(existing)} confirmatory manifests already exist in {out_dir}.\n"
            "  EXP-001 confirmatory is one-shot. To recompute and compare without\n"
            "  writing, pass --reproduce-only.")

    return {"lock": lock, "seeds": seeds, "git_sha": sha, "source_tree_sha256": tree,
            "config_hash": cfg_hash, "comparator_sha256": comp_hash,
            "policy_sha256": dict(SELECTED_POLICY_SHA256)}


def matched_heuristic() -> ExtendedHeuristicRouter:
    prm = dict(json.loads((ROOT / "results/heuristic_matched_search.json").read_text())["best_params"])
    inv = {v: k for k, v in ACTION_NAMES.items()}
    prm["recurrent_default"] = inv[prm["recurrent_default"]]
    prm["novel_action"] = inv[prm["novel_action"]]
    return ExtendedHeuristicRouter(**prm)


def main() -> None:
    ap = argparse.ArgumentParser(description="EXP-001 confirmatory execution (one shot).")
    ap.add_argument("--lock", type=Path, default=ROOT / "results/protocol_v1.1_lock.json")
    ap.add_argument("--out", type=Path, default=ROOT / "results/confirmatory")
    ap.add_argument("--reproduce-only", action="store_true",
                    help="recompute and compare against existing manifests; writes nothing")
    args = ap.parse_args()
    # There is deliberately no --seeds: seeds come only from the lock.

    print("== EXP-001 CONFIRMATORY ==")
    pre = preflight(args.lock, args.out, args.reproduce_only)
    seeds = pre["seeds"]
    print(f"  lock            {args.lock.name} (protocol v{pre['lock']['protocol_version']})")
    print(f"  commit          {pre['git_sha']}")
    print(f"  source tree     {pre['source_tree_sha256'][:16]}...")
    print(f"  config hash     {pre['config_hash']}")
    print(f"  policies        pinned hashes verified")
    print(f"  comparator      pinned hash verified")
    print(f"  seeds           {seeds}  (from the lock; no seed CLI exists)")
    print(f"  mode            {'REPRODUCE-ONLY (no writes)' if args.reproduce_only else 'ONE-SHOT'}")

    wcfg, scfg, ccfg, tcfg = EXP001.world, EXP001.substrate, EXP001.cost, EXP001.train
    worlds = {s: build_world(wcfg, seed=s) for s in seeds}

    # ---- gate 1: leakage ------------------------------------------------
    print("\n-- gate 1: leakage audit --")
    factories = {n: (lambda r=r: r) for n, r in constant_routers().items()}
    factories["HEURISTIC"] = HeuristicRouter
    audit = leakage_audit(list(DEV_SEEDS), factories)
    for c in audit["checks"]:
        if c.get("gating") is False:
            print(f"  [ -- ] {c['id']:<10} {c['name']}  ({c.get('status', 'non-gating')})")
        else:
            print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['id']:<10} {c['name']}")
    if not audit["passed"]:
        raise Refusal("leakage audit failed; K3 fires. Stopping before any metric is inspected.")

    # ---- execute all arms (metrics withheld) ----------------------------
    print("\n-- executing arms (comparative metrics withheld until gates pass) --")
    learned, meta_R = load_policy(ROOT / "results/policies/real_selected.json")
    shuffle_trained, meta_S = load_policy(ROOT / "results/policies/shuffled_selected.json")
    learned.greedy = shuffle_trained.greedy = True
    shuffle_trained.name = "SHUFFLE_TRAINED"

    arms: dict[str, object] = dict(constant_routers())
    arms["HEURISTIC"] = HeuristicRouter()
    arms["HEURISTIC_EXT"] = matched_heuristic()
    arms["LEARNED"] = learned
    arms["SHUFFLE_TRAINED"] = shuffle_trained

    results: dict[str, list] = {}
    for name, router in arms.items():
        results[name] = [rollout(worlds[s], router, ccfg, scfg, seed=s) for s in seeds]

    probs = np.mean([r.action_probs for r in results["LEARNED"]], axis=0)
    arms["RANDOM_MATCHED"] = RandomMatchedRouter(probs)
    results["RANDOM_MATCHED"] = [rollout(worlds[s], arms["RANDOM_MATCHED"], ccfg, scfg, seed=s)
                                 for s in seeds]

    arms["ORACLE"] = OracleRouter(EXP001.oracle_mapping)
    arms["DESIGNED_MAPPING"] = OracleRouter(EXP001.designed_mapping)
    arms["PRIVILEGED_TASKID"] = PrivilegedTaskIdRouter(HeuristicRouter())
    for name in ("ORACLE", "DESIGNED_MAPPING", "PRIVILEGED_TASKID"):
        results[name] = [rollout(worlds[s], arms[name], ccfg, scfg, seed=s) for s in seeds]

    entry = scfg.key_dim + scfg.value_dim
    cap_scfg = replace(scfg, episodic_capacity=scfg.episodic_capacity
                       + (2 * scfg.key_dim * scfg.value_dim) // entry)
    results["CAPACITY_MATCHED_EPISODIC"] = [
        rollout(worlds[s], constant_routers()["ALL_EPISODIC"], ccfg, cap_scfg, seed=s) for s in seeds
    ]
    print(f"  {len(results)} arms x {len(seeds)} seeds executed")

    # ---- manifests ------------------------------------------------------
    manifests = []
    for name, rs in results.items():
        for r in rs:
            m = build_manifest(root=ROOT, classification="CONFIRMATORY", arm=name, result=r,
                               world_cfg=wcfg, sub_cfg=scfg, cost_cfg=ccfg, train_cfg=tcfg,
                               dev_seeds=list(DEV_SEEDS), leakage=audit,
                               notes="EXP-001 confirmatory, protocol "
                                     f"v{pre['lock']['protocol_version']}, one shot.")
            m["first_encounter_leakage"] = first_encounter_class_dependence(r.first_encounter_actions)
            m["diagnostics"]["confusion_hidden_class_by_action"] = confusion_table(r.confusion)
            m["protocol"] = {"version": pre["lock"]["protocol_version"],
                             "lock_commit": pre["lock"]["admissible_commit"],
                             "policy_sha256": pre["policy_sha256"],
                             "comparator_sha256": pre["comparator_sha256"],
                             "policy_seeds": list(POLICY_SEEDS)}
            manifests.append((name, r.seed, m))

    if not args.reproduce_only:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, seed, m in manifests:
            write_manifest(args.out / f"run_{name}_seed{seed}.json", m)

    # ---- gate 2: manifest validation ------------------------------------
    print("\n-- gate 2: manifest validation --")
    v = validate_manifests([m for _, _, m in manifests])
    print(f"  arms: {', '.join(v['arms'])}")
    print(f"  {'PASSED' if v['passed'] else 'FAILED'}")
    if not v["passed"]:
        for r in v["reasons"]:
            print(f"    - {r}")
        raise Refusal("manifest validation failed. Stopping before any metric is inspected.")

    # ---- gate 3: resources ----------------------------------------------
    print("\n-- gate 3: write ceilings and resource accounting --")
    ceiling = ccfg.write_element_ceiling
    ok = True
    for name, rs in results.items():
        w = max(r.ledger["write_elements"] for r in rs)
        fi = max(r.ledger["forced_ignores_budget"] for r in rs)
        breach = w > ceiling
        ok &= not breach
        print(f"  {'ok ' if not breach else 'BAD'} {name:<26} max writes {w:>9,} / {ceiling:,} "
              f"({w / ceiling:5.1%})  forced-ignores {fi:>5}")
    if not ok:
        raise Refusal("write ceiling breached. Stopping before any metric is inspected.")

    # ---- gate 4: benchmark admissibility C1-C6 --------------------------
    print("\n-- gate 4: benchmark admissibility C1-C6 --")
    obj = {k: float(np.mean([r.objective for r in rs])) for k, rs in results.items()}
    util = {k: float(np.mean([r.task_utility for r in rs])) for k, rs in results.items()}
    best_sd = max(SINGLE_DEPTH, key=lambda k: obj[k])
    c1 = obj["ORACLE"] >= 0.45
    c2 = max(util.values()) <= 0.95
    c3 = (obj["ORACLE"] - obj[best_sd]) >= 0.06
    c4 = (obj["ORACLE"] - obj["HEURISTIC_EXT"]) >= 0.06
    best_map, _, _ = search_best_mapping(worlds, ccfg, scfg, seeds, method="exhaustive")
    c5 = is_bijective(best_map) and tuple(best_map) == tuple(EXP001.oracle_mapping)
    print(f"  C1 ORACLE >= 0.45                      {obj['ORACLE']:.4f}   {'PASS' if c1 else 'FAIL'}")
    print(f"  C2 max task utility <= 0.95            {max(util.values()):.4f}   {'PASS' if c2 else 'FAIL'}")
    print(f"  C3 ORACLE - best single depth >= 0.06  {obj['ORACLE'] - obj[best_sd]:.4f}   "
          f"{'PASS' if c3 else 'FAIL'}  ({best_sd})")
    print(f"  C4 ORACLE - HEURISTIC_EXT >= 0.06      {obj['ORACLE'] - obj['HEURISTIC_EXT']:.4f}   "
          f"{'PASS' if c4 else 'FAIL'}")
    print(f"  C5 optimal mapping bijective & == frozen  {list(best_map)}   {'PASS' if c5 else 'FAIL'}")
    print("  C6 local stability of the optimum      inherited from the frozen calibration "
          "(15/16 neighbourhood points); not recomputed on confirmatory seeds")
    c_all = c1 and c2 and c3 and c4 and c5

    # ---- gate 5: invalidating criteria K3-K6, K8 ------------------------
    print("\n-- gate 5: benchmark-invalidating criteria --")
    k3 = not audit["passed"]
    k4 = obj[best_sd] >= obj["LEARNED"]
    k5 = min(util.values()) > 0.95
    k6 = obj["ORACLE"] < 0.45
    k8 = not is_bijective(best_map)
    for tag, fired, desc in (("K3", k3, "leakage failure"),
                             ("K4", k4, "a single-depth control matches LEARNED"),
                             ("K5", k5, "all arms > 0.95 utility"),
                             ("K6", k6, "ORACLE < 0.45"),
                             ("K8", k8, "optimal mapping not a bijection")):
        print(f"  {tag} {'FIRED' if fired else 'not fired'}   {desc}")
    if k3 or k4 or k5 or k6 or k8 or not c_all:
        raise Refusal("a benchmark-invalidating gate fired (or C1-C6 failed). "
                      "Stopping before the headline comparison is interpreted.")

    # ---- only now: comparative metrics ----------------------------------
    print("\n== gates passed; comparative metrics ==")
    summaries = {k: summarize(rs) for k, rs in results.items()}
    print(f"{'arm':<26} {'legal':>5} {'obj':>8} {'util':>6} {'forget':>7} "
          f"{'IGNORE':>7} {'EPIS':>6} {'FAST':>6} {'SLOW':>6}")
    for k, sm in sorted(summaries.items(), key=lambda x: -x[1]["objective"]["mean"]):
        a = sm["action_probs_mean"]
        print(f"{k:<26} {'Y' if sm['legal'] else 'N':>5} {sm['objective']['mean']:>8.4f} "
              f"{sm['task_utility']['mean']:>6.3f} {sm['forgetting']['mean']:>7.3f} "
              f"{a[0]:>7.3f} {a[1]:>6.3f} {a[2]:>6.3f} {a[3]:>6.3f}")

    print("\nper-class recall utility:")
    print(f"{'arm':<26} " + " ".join(f"{c:>9}" for c in CLASS_NAMES.values()))
    for k, sm in sorted(summaries.items(), key=lambda x: -x[1]["objective"]["mean"]):
        row = sm["utility_by_class_mean"]
        print(f"{k:<26} " + " ".join(
            f"{row[c]:>9.3f}" if row[c] == row[c] else f"{'n/a':>9}" for c in CLASS_NAMES.values()))

    print("\n== preregistered contrasts (paired bootstrap, 95% CI) ==")
    contrasts = {}

    def contrast(a: str, b: str, label: str):
        st = paired_bootstrap(summaries[a]["objective"]["values"],
                              summaries[b]["objective"]["values"])
        contrasts[label] = st
        print(f"  {label:<48} {st['mean_diff']:+.4f}  "
              f"[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]  "
              f"{'excludes 0' if st['excludes_zero'] else 'includes 0'}")
        return st

    k1 = contrast("LEARNED", "HEURISTIC_EXT", "K1  LEARNED - HEURISTIC_EXT")
    k2 = contrast("LEARNED", "SHUFFLE_TRAINED", "K2  LEARNED - SHUFFLE_TRAINED")
    k7 = contrast("LEARNED", "CAPACITY_MATCHED_EPISODIC", "K7  LEARNED - CAPACITY_MATCHED")
    k9_priv = contrast("PRIVILEGED_TASKID", "HEURISTIC", "K9  PRIVILEGED_TASKID - HEURISTIC")
    contrast("LEARNED", "RANDOM_MATCHED", "    LEARNED - RANDOM_MATCHED (secondary)")
    contrast("ORACLE", "LEARNED", "    ORACLE - LEARNED (headroom)")

    k1_pass = k1["mean_diff"] > 0 and k1["ci95"][0] > 0
    k2_pass = k2["mean_diff"] > 0 and k2["ci95"][0] > 0
    k7_pass = k7["mean_diff"] > 0 and k7["ci95"][0] > 0
    k9_fires = k9_priv["mean_diff"] >= k1["mean_diff"]
    l5b = json.loads((ROOT / "results/l5b_cross_world.json").read_text())
    k10_pass = bool(l5b.get("passed"))

    print("\n== H1 success rule ==")
    print(f"  K1  LEARNED > HEURISTIC_EXT, CI excludes 0     {'PASS' if k1_pass else 'FAIL'}")
    print(f"  K2  LEARNED > SHUFFLE_TRAINED, CI excludes 0   {'PASS' if k2_pass else 'FAIL'}")
    print(f"  K7  survives capacity matching                 {'PASS' if k7_pass else 'FAIL'}")
    print(f"  K9  privileged task ID explains less           {'PASS' if not k9_fires else 'FAIL'}")
    print(f"  K10 cross-world attribution (L5b)              {'PASS' if k10_pass else 'FAIL'}")
    h1 = k1_pass and k2_pass and k7_pass and not k9_fires and k10_pass
    print(f"\n  H1: {'SUPPORTED' if h1 else 'NOT SUPPORTED'}")

    summary = {
        "classification": "CONFIRMATORY",
        "protocol_version": pre["lock"]["protocol_version"],
        "lock_commit": pre["lock"]["admissible_commit"],
        "git_sha": pre["git_sha"], "source_tree_sha256": pre["source_tree_sha256"],
        "config_hash": pre["config_hash"], "policy_sha256": pre["policy_sha256"],
        "comparator_sha256": pre["comparator_sha256"],
        "confirmatory_seeds": seeds,
        "reproduce_only": args.reproduce_only,
        "gates": {"leakage_passed": audit["passed"], "manifests_validated": v["passed"],
                  "resources_ok": ok,
                  "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5,
                  "C6": "inherited_from_frozen_calibration",
                  "K3": k3, "K4": k4, "K5": k5, "K6": k6, "K8": k8,
                  "confirmatory_optimal_mapping": list(best_map)},
        "summaries": summaries, "contrasts": contrasts,
        "h1": {"K1": k1_pass, "K2": k2_pass, "K7": k7_pass,
               "K9_fires": k9_fires, "K10": k10_pass, "supported": h1},
    }
    if not args.reproduce_only:
        (args.out / "exp001_confirmatory_summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n")
        print(f"\nwrote {len(manifests)} CONFIRMATORY manifests + summary to {args.out}/")
    else:
        print("\nreproduce-only: nothing written")


if __name__ == "__main__":
    main()
