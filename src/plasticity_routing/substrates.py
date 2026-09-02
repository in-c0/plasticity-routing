"""The three write depths plus the abstain action.

The four actions of EXP-001 are the smallest set that makes *write depth* a
real axis rather than a relabelling of expert choice:

    0 IGNORE    no write, no storage, no interference
    1 EPISODIC  exact non-parametric entry; bounded capacity; evictable;
                zero interference; search cost at read time; grows footprint
    2 FAST      parametric associative write that decays; cheap; interferes
                with other parametric content, but transiently
    3 SLOW      parametric associative write that persists; expensive;
                interferes durably -- a wrong slow write is lasting damage

Module-spawning actions (UPDATE_EXISTING_MODULE_i, SPAWN_NEW_MODULE) are
deliberately excluded from EXP-001; see docs/ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE, EPISODIC, FAST, SLOW = 0, 1, 2, 3
ACTIONS = (IGNORE, EPISODIC, FAST, SLOW)
ACTION_NAMES = {IGNORE: "IGNORE", EPISODIC: "EPISODIC", FAST: "FAST", SLOW: "SLOW"}
N_ACTIONS = 4


@dataclass(frozen=True)
class SubstrateConfig:
    key_dim: int = 48
    value_dim: int = 16

    episodic_capacity: int = 60
    fast_decay: float = 0.996
    fast_lr: float = 0.55
    slow_lr: float = 0.30
    slow_consolidation_steps: int = 4


class EpisodicStore:
    """Exact key->value store with bounded capacity and LRU eviction.

    Exactness is its advantage; footprint growth and eviction pressure are its
    price. A stale entry returns a confidently *wrong* answer, which is why
    storing a regime-local rule here is not free.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: dict[int, np.ndarray] = {}
        self.last_touch: dict[int, int] = {}
        self.evictions = 0

    def write(self, key_id: int, value: np.ndarray, t: int) -> None:
        if key_id not in self.data and len(self.data) >= self.capacity:
            victim = min(self.last_touch, key=self.last_touch.get)
            del self.data[victim]
            del self.last_touch[victim]
            self.evictions += 1
        self.data[key_id] = value.copy()
        self.last_touch[key_id] = t

    def read(self, key_id: int, t: int) -> np.ndarray | None:
        v = self.data.get(key_id)
        if v is not None:
            self.last_touch[key_id] = t
        return v

    def __len__(self) -> int:
        return len(self.data)


class AssociativeMatrix:
    """Delta-rule associative memory  v_hat = W k_hat.

    Non-orthogonal keys make writes interfere. `decay` < 1 makes the substrate
    forget on its own, which is exactly what a short-lived local regularity
    wants and what a stable fact does not.
    """

    def __init__(self, key_dim: int, value_dim: int, lr: float, decay: float):
        self.W = np.zeros((value_dim, key_dim))
        self.lr = lr
        self.decay = decay

    def step(self) -> None:
        if self.decay < 1.0:
            self.W *= self.decay

    def write(self, key: np.ndarray, value: np.ndarray, repeats: int = 1) -> None:
        k = key / (np.linalg.norm(key) + 1e-12)
        for _ in range(repeats):
            self.W += self.lr * np.outer(value - self.W @ k, k)

    def read(self, key: np.ndarray) -> np.ndarray:
        k = key / (np.linalg.norm(key) + 1e-12)
        return self.W @ k

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.W))


class SubstrateBank:
    """The agent's whole cognitive substrate: episodic + fast + slow."""

    def __init__(self, cfg: SubstrateConfig):
        self.cfg = cfg
        self.episodic = EpisodicStore(cfg.episodic_capacity)
        self.fast = AssociativeMatrix(cfg.key_dim, cfg.value_dim, cfg.fast_lr, cfg.fast_decay)
        self.slow = AssociativeMatrix(cfg.key_dim, cfg.value_dim, cfg.slow_lr, 1.0)

    def step(self) -> None:
        self.fast.step()

    def apply(self, action: int, key_id: int, key: np.ndarray, value: np.ndarray, t: int) -> None:
        if action == IGNORE:
            return
        if action == EPISODIC:
            self.episodic.write(key_id, value, t)
        elif action == FAST:
            self.fast.write(key, value)
        elif action == SLOW:
            self.slow.write(key, value, repeats=self.cfg.slow_consolidation_steps)
        else:
            raise ValueError(f"unknown action {action}")

    def recall(self, key_id: int, key: np.ndarray, t: int) -> tuple[np.ndarray, str]:
        """Answer a query. Episodic hits win; otherwise read the parametric sum."""
        hit = self.episodic.read(key_id, t)
        if hit is not None:
            return hit, "episodic"
        return self.fast.read(key) + self.slow.read(key), "parametric"

    def probe(self, key_id: int, key: np.ndarray) -> np.ndarray:
        """Read WITHOUT side effects, for evaluator-only retention audits.

        `recall` refreshes episodic LRU order. If audit probes used it, the
        evaluator would silently change the agent's eviction behaviour and
        contaminate the very quantity it is measuring. This path must be used
        by every evaluator probe.
        """
        hit = self.episodic.data.get(key_id)
        if hit is not None:
            return hit
        return self.fast.read(key) + self.slow.read(key)


def utility(answer: np.ndarray, truth: np.ndarray) -> float:
    """Bounded recall utility in [0, 1]: clipped cosine similarity."""
    na = float(np.linalg.norm(answer))
    nt = float(np.linalg.norm(truth))
    if na < 1e-9 or nt < 1e-9:
        return 0.0
    return float(np.clip(np.dot(answer, truth) / (na * nt), 0.0, 1.0))
