#!/usr/bin/env python3
"""L5b -- cross-world utility-shuffle negative control. Preregistered in
`experiments/EXP-001-PREREG.md` §8a (Amendment L), committed before any
cross-world number was inspected.

`R` is the policy trained on the real development worlds; `S` is the identically
specified policy trained on the time-shuffled development worlds -- same
network, same legal feature whitelist, same ES budget, same policy seeds, same
selection rule. The only difference is whether training preserved the
prefix->future-utility relationship, which is exactly the variable to isolate.
The null therefore retains every nuisance mechanism and removes only the
hypothesised informative relationship.

Both policies are frozen artefacts pinned by SHA-256 in
`config.SELECTED_POLICY_SHA256`. This script **loads** them. It must never
retrain: retraining inside the audit would let the compared objects drift with
the source tree.

Evaluated once, on `config.AUDIT_SEEDS` -- neither a training nor a confirmatory
set. The cross is deliberately *not* evaluated on development seeds, because `R`
was selected for real-dev performance and `S` for shuffled-dev performance, so
that comparison is selection-biased in `R`'s favour.

    Delta_real = J_RR - J_SR                      (primary)
    I          = (J_RR - J_SR) - (J_RS - J_SS)    (crossover interaction)

Pass iff the 95% paired-bootstrap CI for **both** excludes zero on the positive
side. Zero is the null; no ratio and no minimum effect size are introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plasticity_routing.agent import rollout  # noqa: E402
from plasticity_routing.config import (  # noqa: E402
    AUDIT_SEEDS, CONFIRMATORY_SEEDS, DEV_SEEDS, EXP001, SELECTED_POLICY_SHA256,
)
from plasticity_routing.manifest import config_hash, environment, git_sha, source_tree_sha256  # noqa: E402
from plasticity_routing.metrics import paired_bootstrap  # noqa: E402
from plasticity_routing.train import load_policy  # noqa: E402
from plasticity_routing.world import build_world, time_shuffled_world  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", type=Path, default=ROOT / "results/policies")
    ap.add_argument("--out", type=Path, default=ROOT / "results/l5b_cross_world.json")
    ap.add_argument("--allow-unpinned-policies", action="store_true",
                    help="development convenience; a real audit must run on the pinned artefacts")
    args = ap.parse_args()

    seeds = list(AUDIT_SEEDS)

    # -- preconditions ----------------------------------------------------
    overlap_dev = sorted(set(seeds) & set(DEV_SEEDS))
    overlap_conf = sorted(set(seeds) & set(CONFIRMATORY_SEEDS))
    if overlap_dev or overlap_conf:
        raise SystemExit(f"audit seeds overlap dev={overlap_dev} confirmatory={overlap_conf}")

    paths = {"R": args.policy_dir / "real_selected.json",
             "S": args.policy_dir / "shuffled_selected.json"}
    hashes = {}
    for label, path in paths.items():
        if not path.exists():
            raise SystemExit(f"missing frozen policy {path}")
        h = sha256_file(path)
        hashes[path.name] = h
        expected = SELECTED_POLICY_SHA256.get(path.name)
        if expected != h:
            msg = (f"{path.name} does not match the hash pinned by Amendment L\n"
                   f"  pinned : {expected}\n  actual : {h}")
            if not args.allow_unpinned_policies:
                raise SystemExit(msg)
            print("WARNING: " + msg)

    R, meta_R = load_policy(paths["R"])
    S, meta_S = load_policy(paths["S"])
    R.greedy = S.greedy = True

    # Both arms must be decision-time legal and identically specified.
    for label, pol in (("R", R), ("S", S)):
        assert pol.legal and pol.privileged_fields == (), f"{label} is not decision-time legal"
    assert R.p.W1.shape == S.p.W1.shape, "policies are not identically specified"
    assert R.n_params == S.n_params

    cfg_hash = config_hash(EXP001.world, EXP001.substrate, EXP001.cost)

    print("== L5b cross-world utility-shuffle negative control ==")
    print(f"R: {paths['R'].name} seed {meta_R['policy_seed']}, dev {meta_R['dev_objective']:.4f}")
    print(f"S: {paths['S'].name} seed {meta_S['policy_seed']}, dev {meta_S['dev_objective']:.4f}")
    print(f"audit seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)}), one-shot")
    print(f"shared config hash for all four cells: {cfg_hash}\n")

    cells = {"J_RR": [], "J_SR": [], "J_RS": [], "J_SS": []}
    per_seed = []
    for s in seeds:
        real = build_world(EXP001.world, seed=s)
        shuf = time_shuffled_world(real, seed=s)
        j_rr = rollout(real, R, EXP001.cost, EXP001.substrate, seed=s).objective
        j_sr = rollout(real, S, EXP001.cost, EXP001.substrate, seed=s).objective
        j_rs = rollout(shuf, R, EXP001.cost, EXP001.substrate, seed=s).objective
        j_ss = rollout(shuf, S, EXP001.cost, EXP001.substrate, seed=s).objective
        for k, v in zip(cells, (j_rr, j_sr, j_rs, j_ss)):
            cells[k].append(v)
        per_seed.append({"seed": s, "J_RR": j_rr, "J_SR": j_sr, "J_RS": j_rs, "J_SS": j_ss,
                         "delta_real": j_rr - j_sr,
                         "interaction": (j_rr - j_sr) - (j_rs - j_ss)})

    # Delta_real = J_RR - J_SR
    delta = paired_bootstrap(cells["J_RR"], cells["J_SR"], n=20000, seed=0)
    # I = (J_RR - J_SR) - (J_RS - J_SS) = (J_RR - J_RS) - (J_SR - J_SS)
    a = [rr - rs for rr, rs in zip(cells["J_RR"], cells["J_RS"])]
    b = [sr - ss for sr, ss in zip(cells["J_SR"], cells["J_SS"])]
    inter = paired_bootstrap(a, b, n=20000, seed=0)

    delta_pos = delta["ci95"][0] > 0
    inter_pos = inter["ci95"][0] > 0
    passed = bool(delta_pos and inter_pos)

    means = {k: float(np.mean(v)) for k, v in cells.items()}
    print(f"{'cell':<8} {'mean J':>9}")
    for k in ("J_RR", "J_SR", "J_RS", "J_SS"):
        print(f"{k:<8} {means[k]:>9.4f}")
    print()
    print(f"Delta_real = J_RR - J_SR            {delta['mean_diff']:+.4f}  "
          f"[{delta['ci95'][0]:+.4f}, {delta['ci95'][1]:+.4f}]  "
          f"{'excludes 0 (positive)' if delta_pos else 'DOES NOT exclude 0 positively'}")
    print(f"I = (J_RR-J_SR) - (J_RS-J_SS)       {inter['mean_diff']:+.4f}  "
          f"[{inter['ci95'][0]:+.4f}, {inter['ci95'][1]:+.4f}]  "
          f"{'excludes 0 (positive)' if inter_pos else 'DOES NOT exclude 0 positively'}")
    print(f"\nL5b: {'PASSED' if passed else 'FAILED'}")

    out = {
        "check": "L5b",
        "name": "cross_world_utility_shuffle_negative_control",
        "preregistered_in": "experiments/EXP-001-PREREG.md #8a (Amendment L)",
        "pass_rule": "95% paired-bootstrap CI for both Delta_real and I excludes zero on the "
                     "positive side",
        "audit_seeds": seeds,
        "n_audit_seeds": len(seeds),
        "policies": {"R": {"file": paths["R"].name, "sha256": hashes[paths["R"].name],
                           "meta": meta_R},
                     "S": {"file": paths["S"].name, "sha256": hashes[paths["S"].name],
                           "meta": meta_S}},
        "policy_hashes_match_amendment_L": all(
            SELECTED_POLICY_SHA256.get(n) == h for n, h in hashes.items()),
        "shared_config_hash_all_cells": cfg_hash,
        "world_config": asdict(EXP001.world),
        "substrate_config": asdict(EXP001.substrate),
        "cost_config": asdict(EXP001.cost),
        "cell_means": means,
        "per_seed": per_seed,
        "delta_real": delta,
        "interaction": inter,
        "delta_real_excludes_zero_positive": delta_pos,
        "interaction_excludes_zero_positive": inter_pos,
        "passed": passed,
        "git_sha": git_sha(ROOT),
        "source_tree_sha256": source_tree_sha256(ROOT),
        "environment": environment(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(f"wrote {args.out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
