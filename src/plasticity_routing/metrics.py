"""Metrics, paired statistics, and diagnostics.

Anything named `*_diagnostic` uses evaluator-only information (the hidden
class). Diagnostics explain a result; they never define success, and they are
never available to a policy.
"""

from __future__ import annotations

import numpy as np

from .substrates import ACTION_NAMES, N_ACTIONS
from .world import CLASS_NAMES


def paired_bootstrap(a: list[float], b: list[float], n: int = 20000, seed: int = 0) -> dict:
    """Paired bootstrap CI for mean(a) - mean(b) over matched seeds."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        raise ValueError("paired_bootstrap needs equal, non-empty samples")
    d = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean_diff": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_pairs": int(d.size),
    }


def routing_agreement_diagnostic(confusion: np.ndarray) -> float:
    """Fraction of write decisions matching the hidden ideal action.

    DIAGNOSTIC ONLY. Never a training signal, never a success criterion: the
    ideal action is unobservable at decision time, and a policy that maximises
    this quantity on first encounters is leaking.
    """
    total = confusion.sum()
    if total == 0:
        return float("nan")
    return float(np.trace(confusion) / total)


def first_encounter_class_dependence(pairs: list[tuple[int, int]]) -> dict:
    """Leakage statistic L3.

    On a *first* encounter, SDW-1 makes every hidden class observationally
    identical by construction. A legal policy's first-encounter action must
    therefore be independent of the hidden class. This reports the empirical
    mutual information I(class ; action) in bits, plus a permutation p-value.
    """
    if not pairs:
        return {"mutual_information_bits": 0.0, "p_value": 1.0, "n": 0}
    cls = np.array([c for c, _ in pairs])
    act = np.array([a for _, a in pairs])

    def mi(c: np.ndarray, a: np.ndarray) -> float:
        joint = np.zeros((4, N_ACTIONS))
        for ci, ai in zip(c, a):
            joint[ci, ai] += 1
        joint /= joint.sum()
        pc = joint.sum(axis=1, keepdims=True)
        pa = joint.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = joint * np.log2(joint / (pc * pa))
        return float(np.nansum(term))

    observed = mi(cls, act)
    rng = np.random.default_rng(0)
    null = np.array([mi(cls, rng.permutation(act)) for _ in range(500)])
    return {
        "mutual_information_bits": observed,
        "null_mean_bits": float(null.mean()),
        "null_p95_bits": float(np.percentile(null, 95)),
        "p_value": float((null >= observed).mean()),
        "n": int(len(pairs)),
    }


def confusion_table(confusion: np.ndarray) -> dict:
    return {
        CLASS_NAMES[c]: {ACTION_NAMES[a]: int(confusion[c, a]) for a in range(N_ACTIONS)}
        for c in range(4)
    }


def summarize(results: list) -> dict:
    """Aggregate matched-seed rollouts for one arm."""
    if not results:
        return {}
    keys = ("task_utility", "forgetting", "objective")
    out: dict = {"router": results[0].router_name, "legal": results[0].legal, "n_seeds": len(results)}
    for k in keys:
        vals = [getattr(r, k) for r in results]
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                  "values": [float(v) for v in vals]}
    out["write_elements_mean"] = float(np.mean([r.ledger["write_elements"] for r in results]))
    out["storage_norm_mean"] = float(np.mean([r.ledger["normalized"]["storage"] for r in results]))
    out["compute_norm_mean"] = float(np.mean([r.ledger["normalized"]["compute"] for r in results]))
    out["router_compute_mean"] = float(np.mean([r.ledger["router_compute_elements"] for r in results]))
    out["action_probs_mean"] = np.mean([r.action_probs for r in results], axis=0).tolist()
    out["utility_by_class_mean"] = {
        c: float(np.nanmean([r.utility_by_class[c] for r in results])) for c in CLASS_NAMES.values()
    }
    out["routing_agreement_diagnostic"] = float(
        np.mean([routing_agreement_diagnostic(r.confusion) for r in results])
    )
    return out
