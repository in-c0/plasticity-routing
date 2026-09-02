"""Training the learned router from future task utility.

The learning signal is delayed future utility net of resource cost. That signal
is applied **offline by the trainer**; it is never an input to the policy at
decision time. Keeping those two things apart is the central methodological
requirement of this track, so it is worth stating precisely:

*   ``LearnedRouter.act`` reads ``features.extract`` output only -- a function
    of the stream prefix.
*   ``train_router`` reads outcomes that occur *after* a decision, because that
    is what "learn where writing pays off" means. This is legitimate offline
    credit assignment, exactly as a policy-gradient method is normally trained.
*   The trained policy is then frozen and evaluated on seeds it never saw.

Advantage is a convex blend of per-decision attributed utility (low variance,
but blind to interference the write caused elsewhere) and the lifetime
objective (captures interference, high variance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent import rollout
from .ledger import CostConfig, Ledger
from .routers import LearnedRouter, PolicyParams
from .substrates import SubstrateConfig
from .world import WorldConfig, build_world


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    lr: float = 0.02
    entropy_beta_start: float = 0.03
    entropy_beta_end: float = 0.002
    alpha_local: float = 0.7
    baseline_momentum: float = 0.9
    temperature: float = 1.0
    hidden: int = 16
    grad_clip: float = 5.0


def _decision_costs(decisions, cost_cfg: CostConfig, lifetime: int, episodic_capacity: int) -> np.ndarray:
    """Per-decision normalised resource charge, in the same units as utility."""
    led = Ledger(cost_cfg)
    norms = led.normalizers(lifetime, episodic_capacity)
    out = np.zeros(len(decisions))
    for i, d in enumerate(decisions):
        w = led.write_cost(d.action) / max(1.0, norms["write"])
        out[i] = cost_cfg.lam_write * w
    return out


def train_router(
    world_cfg: WorldConfig,
    sub_cfg: SubstrateConfig,
    cost_cfg: CostConfig,
    dev_seeds: list[int],
    train_cfg: TrainConfig = TrainConfig(),
    policy_seed: int = 0,
    verbose: bool = False,
) -> tuple[LearnedRouter, list[dict]]:
    """Train on development seeds only. Confirmatory seeds must be disjoint."""
    router = LearnedRouter(hidden=train_cfg.hidden, seed=policy_seed, temperature=train_cfg.temperature)
    router.record = True
    opt_state: dict = {}
    baseline: float | None = None
    history: list[dict] = []

    worlds = {s: build_world(world_cfg, seed=s) for s in dev_seeds}

    for epoch in range(train_cfg.epochs):
        frac = epoch / max(1, train_cfg.epochs - 1)
        beta = train_cfg.entropy_beta_start + frac * (train_cfg.entropy_beta_end - train_cfg.entropy_beta_start)
        epoch_objs = []

        for s in dev_seeds:
            res = rollout(worlds[s], router, cost_cfg, sub_cfg, seed=s, keep_decisions=True)
            epoch_objs.append(res.objective)
            baseline = res.objective if baseline is None else (
                train_cfg.baseline_momentum * baseline + (1 - train_cfg.baseline_momentum) * res.objective
            )

            local = np.array([
                d.attributed_utility / max(1, d.n_attributed_queries) if d.n_attributed_queries else 0.0
                for d in res.decisions
            ])
            local = local - _decision_costs(res.decisions, cost_cfg, world_cfg.lifetime, sub_cfg.episodic_capacity)
            if local.std() > 1e-8:
                local = (local - local.mean()) / (local.std() + 1e-8)
            else:
                local = local - local.mean()

            glob = res.objective - baseline
            adv = train_cfg.alpha_local * local + (1 - train_cfg.alpha_local) * glob

            g = router.grad(adv, entropy_beta=beta)
            gn = float(np.linalg.norm(g.flat()))
            if gn > train_cfg.grad_clip:
                scale = train_cfg.grad_clip / gn
                for name in ("W1", "b1", "W2", "b2"):
                    setattr(g, name, getattr(g, name) * scale)
            router.apply_grad(g, train_cfg.lr, opt_state)
            router.reset_episode()

        history.append({"epoch": epoch, "mean_objective": float(np.mean(epoch_objs)),
                        "entropy_beta": float(beta)})
        if verbose and (epoch % 10 == 0 or epoch == train_cfg.epochs - 1):
            print(f"  epoch {epoch:3d}  dev objective {np.mean(epoch_objs):.4f}")

    router.record = False
    router.reset_episode()
    return router, history


# ---------------------------------------------------------------------------
# Evolution strategies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ESConfig:
    """Hyperparameters for the ES trainer.

    Chosen on development seeds before any confirmatory run, and frozen.
    """

    generations: int = 60
    population: int = 24          # must be even; antithetic pairs
    sigma: float = 0.12
    lr: float = 0.06
    hidden: int = 16
    seeds_per_generation: int = 2
    weight_decay: float = 0.002


def evaluate_policy(
    router: LearnedRouter, worlds: dict, cost_cfg: CostConfig, sub_cfg: SubstrateConfig,
    seeds: list[int],
) -> float:
    """Mean preregistered objective over `seeds`, greedy policy."""
    prev = router.greedy
    router.greedy = True
    try:
        return float(np.mean([
            rollout(worlds[s], router, cost_cfg, sub_cfg, seed=s).objective for s in seeds
        ]))
    finally:
        router.greedy = prev


def train_router_es(
    world_cfg: WorldConfig,
    sub_cfg: SubstrateConfig,
    cost_cfg: CostConfig,
    dev_seeds: list[int],
    es_cfg: ESConfig = ESConfig(),
    policy_seed: int = 0,
    verbose: bool = False,
) -> tuple[LearnedRouter, list[dict]]:
    """Train the routing policy by evolution strategies on development seeds.

    Why ES rather than the policy-gradient trainer above: the preregistered
    objective is

        task utility - forgetting - storage - write - compute,

    and three of those five terms are **non-local**. A write's resource cost is
    four orders of magnitude smaller than its attributed utility, so it vanishes
    under advantage standardisation; interference caused in *other* keys is not
    attributable to the write that caused it; and budget exhaustion is a
    property of the whole trajectory. Per-decision credit assignment therefore
    cannot see most of the objective, and the REINFORCE trainer measurably did
    not: its development objective was flat across 60 epochs and the policy
    collapsed to writing durably until the budget ran out.

    ES optimises the actual objective as a black box. It needs no attribution
    assumption, no reward shaping, and no auxiliary target -- which also means
    it cannot smuggle in privileged information, since the only thing it ever
    reads is the same objective every arm is scored on.

    Uses antithetic sampling and rank normalisation for variance reduction.
    """
    if es_cfg.population % 2 != 0:
        raise ValueError("population must be even for antithetic sampling")

    router = LearnedRouter(hidden=es_cfg.hidden, seed=policy_seed)
    router.greedy = True
    theta = router.p.flat().copy()
    n = theta.size

    rng = np.random.default_rng(policy_seed + 9973)
    worlds = {s: build_world(world_cfg, seed=s) for s in dev_seeds}
    opt_state: dict = {}
    history: list[dict] = []
    best_theta, best_score = theta.copy(), -np.inf

    def score(vec: np.ndarray, seeds: list[int]) -> float:
        router.p.set_flat(vec)
        return evaluate_policy(router, worlds, cost_cfg, sub_cfg, seeds)

    for gen in range(es_cfg.generations):
        k = min(es_cfg.seeds_per_generation, len(dev_seeds))
        gen_seeds = list(rng.choice(dev_seeds, size=k, replace=False))

        half = es_cfg.population // 2
        eps = rng.standard_normal((half, n))
        returns = np.empty(es_cfg.population)
        for i in range(half):
            returns[2 * i] = score(theta + es_cfg.sigma * eps[i], gen_seeds)
            returns[2 * i + 1] = score(theta - es_cfg.sigma * eps[i], gen_seeds)

        # Rank normalisation: robust to the objective's scale and to outliers.
        ranks = np.empty(es_cfg.population)
        ranks[np.argsort(returns)] = np.arange(es_cfg.population)
        shaped = ranks / (es_cfg.population - 1) - 0.5

        grad = np.zeros(n)
        for i in range(half):
            grad += (shaped[2 * i] - shaped[2 * i + 1]) * eps[i]
        grad /= (half * es_cfg.sigma)
        grad -= es_cfg.weight_decay * theta

        router.p.set_flat(theta)
        g = PolicyParams(
            np.zeros_like(router.p.W1), np.zeros_like(router.p.b1),
            np.zeros_like(router.p.W2), np.zeros_like(router.p.b2),
        )
        g_holder = LearnedRouter(hidden=es_cfg.hidden, seed=policy_seed)
        g_holder.p.set_flat(grad)
        g = g_holder.p
        router.apply_grad(g, es_cfg.lr, opt_state)
        theta = router.p.flat().copy()

        full = score(theta, dev_seeds)
        if full > best_score:
            best_score, best_theta = full, theta.copy()
        history.append({"generation": gen, "dev_objective": full,
                        "population_mean": float(returns.mean()),
                        "population_max": float(returns.max()),
                        "seeds": [int(s) for s in gen_seeds]})
        if verbose and (gen % 10 == 0 or gen == es_cfg.generations - 1):
            print(f"  gen {gen:3d}  dev objective {full:.4f}  (pop mean {returns.mean():.4f})")

    # Select the best development checkpoint, not the last one.
    router.p.set_flat(best_theta)
    router.greedy = True
    history.append({"selected": "best_dev_checkpoint", "dev_objective": best_score})
    return router, history
