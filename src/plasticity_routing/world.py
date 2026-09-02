"""SDW-1: Stratified Depth World, version 1.

A synthetic continual stream in which different observations genuinely deserve
different *storage depths*. The hidden class that determines the ideal depth is
never observable at decision time; it is recoverable only by integrating
evidence over the stream prefix.

Design contract (enforced by tests/test_world.py):

1. Every key vector and value vector is drawn from the same distribution
   regardless of hidden class, so the *first* encounter with a key is
   statistically uninformative about its class.
2. Class identity becomes inferable only through recurrence, value revision,
   and past query traffic -- all of which are prefix functions.
3. Each hidden class has a different ideal substrate for mechanistically
   distinct reasons (see docs/BENCHMARK.md), so no single-depth policy wins.

Nothing in this module may be exposed to a router. `HIDDEN_FIELDS` names the
attributes that are evaluator-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

import numpy as np

# Hidden generative classes. Evaluator-only.
NOISE = 0
ONE_OFF = 1
LOCAL = 2
STABLE = 3

CLASS_NAMES = {NOISE: "NOISE", ONE_OFF: "ONE_OFF", LOCAL: "LOCAL", STABLE: "STABLE"}

#: Attributes of an event that must never reach a routing policy.
HIDDEN_FIELDS = ("hidden_class", "regime_id", "ideal_action", "is_evaluator_probe")

EventKind = Literal["WRITE", "QUERY"]


@dataclass(frozen=True)
class WorldConfig:
    """Generative parameters for SDW-1.

    All fields are development-tunable only. Confirmatory runs must use a
    frozen config whose hash is recorded in the run manifest.
    """

    key_dim: int = 48
    value_dim: int = 16

    lifetime: int = 3000
    regime_len: int = 300

    n_stable_keys: int = 20
    n_local_slots: int = 12

    # Per-write-event class prior. Order: NOISE, ONE_OFF, LOCAL, STABLE.
    class_prior: tuple[float, float, float, float] = (0.30, 0.30, 0.26, 0.14)

    # ONE_OFF items are queried exactly once, this many steps later.
    one_off_delay_min: int = 4
    one_off_delay_max: int = 45

    # A LOCAL key is queried this many times while its regime is active.
    local_queries_per_presentation: int = 3
    # A STABLE key is queried this many times, spread across the whole lifetime.
    stable_queries_per_presentation: int = 3
    # How far ahead a STABLE query may land (tests durability).
    stable_query_horizon: int = 1400

    # Fraction of timesteps that are write events (rest are query slots).
    write_fraction: float = 0.55

    # Recurrent keys (LOCAL slots, STABLE keys) are *introduced* at times drawn
    # uniformly over this fraction of the lifetime, and each is force-presented
    # at its introduction time. Without this, the small recurrent pools are all
    # exhausted early and a key's arrival time predicts its class -- which
    # leakage test L3 correctly flags, because it lets a router identify a
    # recurrent key on its first encounter without any recurrence evidence.
    # The window must be the *whole* lifetime: any narrower window leaves the
    # mean first-encounter time of recurrent classes earlier than that of the
    # single-use classes, which is the same cue in weaker form.
    activation_window: float = 1.0

    value_noise: float = 0.05


@dataclass
class Event:
    """One stream event.

    Router-visible fields: `kind`, `t`, `key_id`, `key`, `value` (WRITE only).
    Everything listed in `HIDDEN_FIELDS` is evaluator-only bookkeeping and is
    stripped by `plasticity_routing.features` before a policy sees anything.
    """

    kind: EventKind
    t: int
    key_id: int
    key: np.ndarray
    value: np.ndarray | None = None      # observed value (WRITE) / ground truth (QUERY)
    hidden_class: int = NOISE            # HIDDEN
    regime_id: int = -1                  # HIDDEN
    ideal_action: int = 0                # HIDDEN (oracle only)
    is_evaluator_probe: bool = False     # HIDDEN


@dataclass
class World:
    config: WorldConfig
    seed: int
    events: list[Event] = field(default_factory=list)
    keys: dict[int, np.ndarray] = field(default_factory=dict)
    #: key_id -> list of (t_from, value) giving the currently-true value over time.
    truth_timeline: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)
    class_of_key: dict[int, int] = field(default_factory=dict)
    #: Evaluator-only: recurrent keys eligible for retention audit probes.
    _probe_candidates: list[int] = field(default_factory=list)

    def config_dict(self) -> dict:
        return asdict(self.config)

    def true_value(self, key_id: int, t: int) -> np.ndarray | None:
        """Evaluator-only: the value that is correct for `key_id` at time `t`."""
        tl = self.truth_timeline.get(key_id)
        if not tl:
            return None
        cur = None
        for t_from, v in tl:
            if t_from <= t:
                cur = v
            else:
                break
        return cur

    def live_keys(self, t: int) -> list[int]:
        """Evaluator-only: keys whose value is currently well-defined and still probed.

        Only recurrent classes are audited; one-off and noise keys have no
        retention semantics. `_probe_candidates` is precomputed at build time so
        that auditing does not scan the whole (mostly single-use) key inventory.
        """
        return [kid for kid in self._probe_candidates if self.true_value(kid, t) is not None]


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.standard_normal((n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def build_world(config: WorldConfig, seed: int) -> World:
    """Generate one lifetime of SDW-1.

    Key and value vectors are drawn i.i.d. from the same distribution for every
    hidden class, which is what makes first encounters genuinely ambiguous.
    """
    rng = np.random.default_rng(seed)
    cfg = config
    w = World(config=cfg, seed=seed)

    n_regimes = max(1, cfg.lifetime // cfg.regime_len)

    # ---- persistent key inventories -------------------------------------
    next_key_id = 0

    stable_ids = list(range(next_key_id, next_key_id + cfg.n_stable_keys))
    next_key_id += cfg.n_stable_keys
    for kid in stable_ids:
        w.keys[kid] = _unit(rng, 1, cfg.key_dim)[0]
        w.class_of_key[kid] = STABLE
        v = _unit(rng, 1, cfg.value_dim)[0]
        w.truth_timeline[kid] = [(0, v)]

    # LOCAL slots are *reused* across regimes with a different value each time.
    # That reuse is what makes durable storage of a local rule actively harmful.
    local_ids = list(range(next_key_id, next_key_id + cfg.n_local_slots))
    next_key_id += cfg.n_local_slots
    for kid in local_ids:
        w.keys[kid] = _unit(rng, 1, cfg.key_dim)[0]
        w.class_of_key[kid] = LOCAL
        w.truth_timeline[kid] = []
    local_regime_values: dict[int, dict[int, np.ndarray]] = {}
    for r in range(n_regimes):
        local_regime_values[r] = {}
        for kid in local_ids:
            v = _unit(rng, 1, cfg.value_dim)[0]
            local_regime_values[r][kid] = v
            w.truth_timeline[kid].append((r * cfg.regime_len, v))

    # ---- event schedule --------------------------------------------------
    prior = np.asarray(cfg.class_prior, dtype=float)
    prior = prior / prior.sum()
    # Per-step, per-class emission rates. Classes are emitted by *independent*
    # Bernoulli draws rather than by sampling one class per write slot. With a
    # single categorical draw, a step at which no recurrent key is yet active
    # would be reassigned to NOISE/ONE_OFF, making the single-use classes more
    # frequent early and reintroducing the arrival-time cue that activation
    # times exist to remove.
    rates = prior * cfg.write_fraction

    activation_hi = max(1, int(cfg.lifetime * cfg.activation_window))
    activation_t = {kid: int(rng.integers(0, activation_hi)) for kid in stable_ids + local_ids}
    activates_at: dict[int, list[int]] = {}
    for kid, at in activation_t.items():
        activates_at.setdefault(at, []).append(kid)

    pending_queries: list[tuple[int, int, int]] = []  # (t, key_id, hidden_class)
    writes: list[Event] = []

    def schedule_local(kid: int, t: int, regime: int) -> None:
        regime_end = min((regime + 1) * cfg.regime_len, cfg.lifetime)
        for _ in range(cfg.local_queries_per_presentation):
            if t + 2 < regime_end:
                pending_queries.append((int(rng.integers(t + 2, regime_end)), kid, LOCAL))

    def schedule_stable(kid: int, t: int) -> None:
        for _ in range(cfg.stable_queries_per_presentation):
            hi = min(t + cfg.stable_query_horizon, cfg.lifetime - 1)
            if hi > t + 2:
                pending_queries.append((int(rng.integers(t + 2, hi)), kid, STABLE))

    for t in range(cfg.lifetime):
        regime = min(t // cfg.regime_len, n_regimes - 1)

        # Forced first presentation at the key's introduction time. This makes
        # the distribution of *first-encounter times* uniform for recurrent
        # classes, matching the single-use classes.
        for kid in activates_at.get(t, []):
            cls = w.class_of_key[kid]
            if cls == STABLE:
                writes.append(Event("WRITE", t, kid, w.keys[kid], w.true_value(kid, t), STABLE, regime,
                                    ideal_action=3))
                schedule_stable(kid, t)
            else:
                val = local_regime_values[regime][kid]
                writes.append(Event("WRITE", t, kid, w.keys[kid], val, LOCAL, regime, ideal_action=2))
                schedule_local(kid, t, regime)

        active_local = [k for k in local_ids if activation_t[k] < t]
        active_stable = [k for k in stable_ids if activation_t[k] < t]

        if rng.random() < rates[NOISE]:
            kid = next_key_id
            next_key_id += 1
            w.keys[kid] = _unit(rng, 1, cfg.key_dim)[0]
            w.class_of_key[kid] = NOISE
            val = _unit(rng, 1, cfg.value_dim)[0]
            # Never entered into truth_timeline: NOISE is never queried, so any
            # capacity or interference it consumes is pure waste.
            writes.append(Event("WRITE", t, kid, w.keys[kid], val, NOISE, regime, ideal_action=0))

        if rng.random() < rates[ONE_OFF]:
            kid = next_key_id
            next_key_id += 1
            w.keys[kid] = _unit(rng, 1, cfg.key_dim)[0]
            w.class_of_key[kid] = ONE_OFF
            val = _unit(rng, 1, cfg.value_dim)[0]
            w.truth_timeline[kid] = [(t, val)]
            writes.append(Event("WRITE", t, kid, w.keys[kid], val, ONE_OFF, regime, ideal_action=1))
            dt = int(rng.integers(cfg.one_off_delay_min, cfg.one_off_delay_max + 1))
            if t + dt < cfg.lifetime:
                pending_queries.append((t + dt, kid, ONE_OFF))

        if active_local and rng.random() < rates[LOCAL]:
            kid = int(rng.choice(active_local))
            val = local_regime_values[regime][kid]
            writes.append(Event("WRITE", t, kid, w.keys[kid], val, LOCAL, regime, ideal_action=2))
            schedule_local(kid, t, regime)

        if active_stable and rng.random() < rates[STABLE]:
            kid = int(rng.choice(active_stable))
            writes.append(Event("WRITE", t, kid, w.keys[kid], w.true_value(kid, t), STABLE, regime,
                                ideal_action=3))
            schedule_stable(kid, t)

    # ---- interleave ------------------------------------------------------
    by_t: dict[int, list[Event]] = {}
    for e in writes:
        by_t.setdefault(e.t, []).append(e)
    for (qt, kid, cls) in pending_queries:
        truth = w.true_value(kid, qt)
        if truth is None:
            continue
        # A LOCAL query is only meaningful inside the regime that defined it.
        by_t.setdefault(qt, []).append(
            Event("QUERY", qt, kid, w.keys[kid], truth, cls, min(qt // cfg.regime_len, n_regimes - 1))
        )

    ordered: list[Event] = []
    for t in range(cfg.lifetime):
        for e in by_t.get(t, []):
            ordered.append(e)
    w.events = ordered
    w._probe_candidates = [k for k, c in w.class_of_key.items() if c in (STABLE, LOCAL)]
    return w


def class_histogram(world: World) -> dict[str, int]:
    out = {name: 0 for name in CLASS_NAMES.values()}
    for e in world.events:
        if e.kind == "WRITE":
            out[CLASS_NAMES[e.hidden_class]] += 1
    return out
