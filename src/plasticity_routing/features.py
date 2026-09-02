"""Decision-time feature extraction -- the leakage firewall.

This module is the *only* path by which information reaches a routing policy.
It is deliberately small so that it can be audited by reading it.

The single invariant, tested in `tests/test_leakage.py` and audited at runtime
by `scripts/audit_leakage.py`:

    Every feature is a function of the stream prefix up to and including the
    current write event, plus the agent's own internal state. No feature may
    depend on the hidden class, the regime identity, the future query schedule,
    evaluator audit probes, or any future utility.

`FEATURE_NAMES` is the whitelist. Adding a feature requires adding its name
here, and the leakage tests re-derive the whole vector from a truncated stream
to prove causality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .substrates import SubstrateBank, utility

FEATURE_NAMES = (
    "bias",
    "is_novel",
    "log_times_seen",
    "presentation_recency",
    "log_times_queried",
    "query_recency",
    "pred_error",
    "value_revised",
    "value_agreement",
    "episodic_occupancy",
    "fast_norm",
    "slow_norm",
    "past_query_hit_rate",
    "write_budget_remaining",
    "t_frac",
)
N_FEATURES = len(FEATURE_NAMES)

#: Names that must never appear as a feature. Asserted by the leakage tests.
FORBIDDEN_SOURCES = (
    "hidden_class",
    "regime_id",
    "ideal_action",
    "is_evaluator_probe",
    "future_utility",
    "future_query_schedule",
    "audit_probe",
    "true_value_at_future_time",
)


@dataclass
class KeyTrace:
    """Per-key statistics accumulated strictly from the observed prefix."""

    times_seen: int = 0
    last_seen_t: int = -1
    times_queried: int = 0
    last_query_t: int = -1
    query_hits: float = 0.0
    last_observed_value: np.ndarray | None = None


@dataclass
class ObserverState:
    """Everything the agent has legitimately observed so far."""

    lifetime: int
    traces: dict[int, KeyTrace] = field(default_factory=dict)

    def trace(self, key_id: int) -> KeyTrace:
        return self.traces.setdefault(key_id, KeyTrace())

    # -- prefix updates ---------------------------------------------------
    def observe_write(self, key_id: int, value: np.ndarray, t: int) -> None:
        tr = self.trace(key_id)
        tr.times_seen += 1
        tr.last_seen_t = t
        tr.last_observed_value = value.copy()

    def observe_query(self, key_id: int, t: int, hit_utility: float) -> None:
        """Record that a query happened and how well our own memory answered it.

        This is legal: it is the agent's own past performance, not a label about
        the future. It is what makes utility-driven credit assignment possible
        without privileged information.
        """
        tr = self.trace(key_id)
        tr.times_queried += 1
        tr.last_query_t = t
        tr.query_hits += hit_utility


def extract(
    *,
    key_id: int,
    key: np.ndarray,
    value: np.ndarray,
    t: int,
    obs: ObserverState,
    bank: SubstrateBank,
    write_budget_remaining: float,
) -> np.ndarray:
    """Build the decision-time feature vector for a write event.

    Arguments are named explicitly so that no `Event` object -- which carries
    hidden fields -- can be passed in by accident.
    """
    tr = obs.traces.get(key_id)
    horizon = max(1, obs.lifetime)

    if tr is None:
        is_novel = 1.0
        log_seen = 0.0
        recency = 1.0
        log_queried = 0.0
        q_recency = 1.0
        hit_rate = 0.0
        value_revised = 0.0
        value_agreement = 0.0
    else:
        is_novel = 0.0 if tr.times_seen > 0 else 1.0
        log_seen = float(np.log1p(tr.times_seen) / np.log(50.0))
        recency = 1.0 if tr.last_seen_t < 0 else min(1.0, (t - tr.last_seen_t) / horizon)
        log_queried = float(np.log1p(tr.times_queried) / np.log(50.0))
        q_recency = 1.0 if tr.last_query_t < 0 else min(1.0, (t - tr.last_query_t) / horizon)
        hit_rate = tr.query_hits / tr.times_queried if tr.times_queried else 0.0
        if tr.last_observed_value is None:
            value_revised, value_agreement = 0.0, 0.0
        else:
            agree = utility(tr.last_observed_value, value)
            value_agreement = agree
            value_revised = 1.0 if agree < 0.8 else 0.0

    answer, _ = bank.recall(key_id, key, t)
    pred_error = 1.0 - utility(answer, value)

    feats = np.array(
        [
            1.0,
            is_novel,
            log_seen,
            recency,
            log_queried,
            q_recency,
            pred_error,
            value_revised,
            value_agreement,
            len(bank.episodic) / max(1, bank.episodic.capacity),
            float(np.tanh(bank.fast.norm)),
            float(np.tanh(bank.slow.norm)),
            hit_rate,
            float(np.clip(write_budget_remaining, 0.0, 1.0)),
            t / horizon,
        ],
        dtype=float,
    )
    assert feats.shape == (N_FEATURES,), "feature vector desynchronised from FEATURE_NAMES"
    return feats
