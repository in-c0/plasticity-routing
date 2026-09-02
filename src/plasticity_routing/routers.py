"""Routing policies: the controls and the method under test.

Every router exposes the same interface. The distinction that matters
scientifically is `legal`:

*   `legal = True`  -- the router sees only `features.extract` output, i.e. a
    function of the stream prefix. These are the only arms that may support a
    claim.
*   `legal = False` -- the router receives privileged information (the hidden
    class, or the regime id). ORACLE is an *upper bound only* and is never a
    claim; PRIVILEGED_TASKID exists to quantify how much of any gap is simply
    access to context identity rather than learned inference.

`n_params` is the router's decision-time compute charge. It is billed on every
decision and reported separately from substrate compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import N_FEATURES
from .substrates import EPISODIC, FAST, IGNORE, N_ACTIONS, SLOW


class Router:
    name = "base"
    legal = True
    privileged_fields: tuple[str, ...] = ()
    n_params = 0

    def act(self, feats: np.ndarray, rng: np.random.Generator, privileged: dict | None = None) -> int:
        raise NotImplementedError

    def reset_episode(self) -> None:
        pass


class ConstantRouter(Router):
    """Depth-agnostic controls: everything to one substrate."""

    def __init__(self, action: int, name: str):
        self.action = action
        self.name = name

    def act(self, feats, rng, privileged=None) -> int:
        return self.action


class HeuristicRouter(Router):
    """Fixed hand-designed routing ladder.

    This is the control that matters most. Prior work (Yoon 2026,
    arXiv:2606.30067) reports that a simple similarity rule matches or beats a
    learned RL allocation controller under fixed capacity. If the learned
    router cannot beat this, the honest conclusion is that learning adds
    nothing here.

    Thresholds are the argmax over the predeclared grid in
    experiments/EXP-001-PREREG.md, selected on development seeds by
    `scripts/calibrate_heuristic.py` to make this comparator as strong as the
    grid allows (dev objective 0.4046, versus 0.3962 for the uncalibrated
    defaults). Under-tuning the primary comparator would flatter the method
    under test, so the calibration budget matches the learned router's.
    """

    name = "HEURISTIC"

    def __init__(self, seen_threshold: int = 1, revision_tolerance: float = 0.8,
                 error_floor: float = 0.25):
        self.seen_threshold = seen_threshold
        self.revision_tolerance = revision_tolerance
        self.error_floor = error_floor
        # Charged as if it were a tiny linear probe, so it is not free.
        self.n_params = N_FEATURES

    def act(self, feats, rng, privileged=None) -> int:
        from .features import FEATURE_NAMES

        f = dict(zip(FEATURE_NAMES, feats))
        seen_norm = f["log_times_seen"] * np.log(50.0)
        times_seen = float(np.expm1(seen_norm))
        if f["pred_error"] < self.error_floor and times_seen >= 1:
            return IGNORE                       # already known well enough
        if times_seen < self.seen_threshold:
            return EPISODIC                     # unproven: cheap, reversible
        # Compare against the router's own tolerance rather than the
        # pre-thresholded `value_revised` flag, whose cut is fixed in
        # features.py. Reading the flag made `revision_tolerance` a dead
        # parameter and silently shrank the calibration grid from 27 points
        # to 9.
        if f["value_agreement"] < self.revision_tolerance:
            return FAST                         # value keeps changing -> let it decay
        return SLOW                             # recurrent and stable -> consolidate


class RandomMatchedRouter(Router):
    """Random routing with an action distribution matched to a target arm.

    Matching is done post-hoc on the target's realised action histogram, so the
    write and storage budgets are comparable. This isolates "did routing decide
    *which* item goes where" from "did the arm simply spend its budget
    differently".
    """

    name = "RANDOM_MATCHED"

    def __init__(self, probs: np.ndarray):
        p = np.asarray(probs, dtype=float)
        self.probs = p / p.sum()

    def act(self, feats, rng, privileged=None) -> int:
        return int(rng.choice(N_ACTIONS, p=self.probs))


#: The mapping the benchmark was *designed* around. It is a hypothesis about
#: SDW-1, not a fact: whether it is actually optimal is checked by
#: `search_best_mapping` during development calibration.
INTENDED_MAPPING = (IGNORE, EPISODIC, FAST, SLOW)


class OracleRouter(Router):
    """Class-conditional upper bound. ILLEGAL: reads the hidden generative class.

    An "oracle" that merely applies the designer's assumed mapping is not an
    upper bound -- it is an assumption. During the first calibration sweep the
    assumed mapping was in fact *beaten by the fixed heuristic*, which showed
    the assumption was wrong rather than that the heuristic was superhuman.

    The oracle used for reporting is therefore the argmax over all
    ``len(ACTIONS) ** 4`` deterministic class->action mappings, found by
    `search_best_mapping` on development seeds. It upper-bounds the entire
    class-conditional policy family, which is the family a decision-time-legal
    router is trying to approximate without seeing the class.

    ORACLE is a ceiling for calibration only. Any run that lists ORACLE among
    its claim-supporting arms is invalid.
    """

    name = "ORACLE"
    legal = False
    privileged_fields = ("hidden_class",)

    def __init__(self, mapping: tuple[int, ...] = INTENDED_MAPPING):
        self.mapping = tuple(mapping)

    def act(self, feats, rng, privileged=None) -> int:
        if privileged is None or "hidden_class" not in privileged:
            raise RuntimeError("OracleRouter requires privileged hidden_class")
        return self.mapping[privileged["hidden_class"]]


def _mapping_objective(mapping, worlds, cost_cfg, sub_cfg, seeds) -> float:
    from .agent import rollout

    return float(np.mean([
        rollout(worlds[s], OracleRouter(mapping), cost_cfg, sub_cfg, seed=s).objective for s in seeds
    ]))


def search_best_mapping(
    worlds, cost_cfg, sub_cfg, seeds, method: str = "exhaustive"
) -> tuple[tuple[int, ...], float, list]:
    """Search class-conditional mappings for the true upper bound. Development only.

    `method="exhaustive"` evaluates all ``N_ACTIONS ** 4`` mappings.
    `method="coordinate"` runs coordinate ascent from INTENDED_MAPPING, which
    costs roughly an order of magnitude less and is used to prefilter a large
    configuration grid before exhaustive verification.

    Returns `(best_mapping, best_objective, ranked_rows)`.
    """
    import itertools

    if method == "exhaustive":
        rows = [
            (tuple(m), _mapping_objective(m, worlds, cost_cfg, sub_cfg, seeds))
            for m in itertools.product(range(N_ACTIONS), repeat=4)
        ]
        rows.sort(key=lambda x: -x[1])
        return rows[0][0], rows[0][1], rows

    if method != "coordinate":
        raise ValueError(f"unknown search method {method!r}")

    cache: dict[tuple[int, ...], float] = {}

    def obj(m):
        m = tuple(m)
        if m not in cache:
            cache[m] = _mapping_objective(m, worlds, cost_cfg, sub_cfg, seeds)
        return cache[m]

    cur = tuple(INTENDED_MAPPING)
    best = obj(cur)
    for _ in range(4):
        improved = False
        for cls in range(4):
            for a in range(N_ACTIONS):
                if a == cur[cls]:
                    continue
                cand = cur[:cls] + (a,) + cur[cls + 1:]
                v = obj(cand)
                if v > best + 1e-9:
                    cur, best, improved = cand, v, True
        if not improved:
            break
    rows = sorted(cache.items(), key=lambda x: -x[1])
    return cur, best, rows


def is_bijective(mapping) -> bool:
    """True when every action is uniquely optimal for exactly one hidden class.

    Bijectivity is what makes the four-action space *necessary*. If two classes
    share an optimal action, one action is redundant and the benchmark does not
    justify the action set it declares.
    """
    return len(set(mapping)) == N_ACTIONS


class PrivilegedTaskIdRouter(Router):
    """Heuristic router that additionally receives the regime id.

    Quantifies the value of privileged context identity so that it is not
    silently conflated with learned routing benefit.
    """

    name = "PRIVILEGED_TASKID"
    legal = False
    privileged_fields = ("regime_id",)

    def __init__(self, base: HeuristicRouter):
        self.base = base
        self.n_params = base.n_params + 1
        self._last_regime = -1

    def act(self, feats, rng, privileged=None) -> int:
        if privileged is None or "regime_id" not in privileged:
            raise RuntimeError("PrivilegedTaskIdRouter requires privileged regime_id")
        regime = privileged["regime_id"]
        changed = regime != self._last_regime
        self._last_regime = regime
        a = self.base.act(feats, rng)
        # Knowing a regime boundary just occurred, prefer the decaying substrate
        # for anything whose value may have been redefined by the new regime.
        if changed and a == SLOW:
            return FAST
        return a

    def reset_episode(self) -> None:
        self._last_regime = -1
        self.base.reset_episode()


@dataclass
class PolicyParams:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def copy(self) -> "PolicyParams":
        return PolicyParams(self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def flat(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_flat(self, v: np.ndarray) -> None:
        i = 0
        for name in ("W1", "b1", "W2", "b2"):
            arr = getattr(self, name)
            n = arr.size
            arr[...] = v[i:i + n].reshape(arr.shape)
            i += n
        if i != v.size:
            raise ValueError("parameter vector length mismatch")

    @property
    def size(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


class LearnedRouter(Router):
    """The method under test: a small stochastic policy over write depth.

    Input is `features.extract` output only. The *training signal* comes from
    future task utility and resource cost, which is the entire point of the
    track -- but that signal is applied offline by the trainer and is never an
    input to the policy at decision time. `tests/test_leakage.py` enforces the
    distinction.
    """

    name = "LEARNED"

    def __init__(self, hidden: int = 16, seed: int = 0, temperature: float = 1.0):
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(N_FEATURES)
        self.p = PolicyParams(
            W1=rng.normal(0, scale, (hidden, N_FEATURES)),
            b1=np.zeros(hidden),
            W2=rng.normal(0, 1.0 / np.sqrt(hidden), (N_ACTIONS, hidden)),
            b2=np.zeros(N_ACTIONS),
        )
        self.temperature = temperature
        self.n_params = self.p.size
        self.greedy = False
        self.trace: list[tuple[np.ndarray, np.ndarray, int]] = []
        self.record = False

    # -- forward -----------------------------------------------------------
    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.tanh(self.p.W1 @ x + self.p.b1)
        logits = (self.p.W2 @ h + self.p.b2) / self.temperature
        logits -= logits.max()
        e = np.exp(logits)
        return h, e / e.sum()

    def probs(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[1]

    def act(self, feats, rng, privileged=None) -> int:
        h, pi = self._forward(feats)
        a = int(np.argmax(pi)) if self.greedy else int(rng.choice(N_ACTIONS, p=pi))
        if self.record:
            self.trace.append((feats.copy(), h, a))
        return a

    def reset_episode(self) -> None:
        self.trace = []

    # -- REINFORCE gradient -------------------------------------------------
    def grad(self, advantages: np.ndarray, entropy_beta: float = 0.01) -> PolicyParams:
        g = PolicyParams(
            np.zeros_like(self.p.W1), np.zeros_like(self.p.b1),
            np.zeros_like(self.p.W2), np.zeros_like(self.p.b2),
        )
        if not self.trace:
            return g
        for (x, h, a), adv in zip(self.trace, advantages):
            logits = (self.p.W2 @ h + self.p.b2) / self.temperature
            logits -= logits.max()
            e = np.exp(logits)
            pi = e / e.sum()

            dlogits = -pi.copy()
            dlogits[a] += 1.0
            dlogits *= adv
            # entropy bonus: encourage exploration early, decayed by the caller
            dlogits += entropy_beta * pi * (-np.log(pi + 1e-12) - float(-(pi * np.log(pi + 1e-12)).sum()))
            dlogits /= self.temperature

            g.W2 += np.outer(dlogits, h)
            g.b2 += dlogits
            dh = (self.p.W2.T @ dlogits) * (1.0 - h ** 2)
            g.W1 += np.outer(dh, x)
            g.b1 += dh
        n = max(1, len(self.trace))
        for arr in (g.W1, g.b1, g.W2, g.b2):
            arr /= n
        return g

    def apply_grad(self, g: PolicyParams, lr: float, state: dict) -> None:
        """Adam."""
        state.setdefault("t", 0)
        state["t"] += 1
        t = state["t"]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name in ("W1", "b1", "W2", "b2"):
            grad = getattr(g, name)
            m = state.setdefault(f"m_{name}", np.zeros_like(grad))
            v = state.setdefault(f"v_{name}", np.zeros_like(grad))
            m *= b1
            m += (1 - b1) * grad
            v *= b2
            v += (1 - b2) * grad ** 2
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            cur = getattr(self.p, name)
            cur += lr * mhat / (np.sqrt(vhat) + eps)


def constant_routers() -> dict[str, Router]:
    return {
        "ALL_IGNORE": ConstantRouter(IGNORE, "ALL_IGNORE"),
        "ALL_EPISODIC": ConstantRouter(EPISODIC, "ALL_EPISODIC"),
        "ALL_FAST": ConstantRouter(FAST, "ALL_FAST"),
        "ALL_SLOW": ConstantRouter(SLOW, "ALL_SLOW"),
    }
