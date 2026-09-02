"""Rollout engine: one router, one lifetime, one ledger.

The engine enforces three invariants that the whole track depends on:

1.  A router receives privileged fields *only* if it declares them in
    `privileged_fields`. Legal routers are handed `None`.
2.  Evaluator retention probes use `SubstrateBank.probe`, which has no side
    effects, and are charged to no budget. They never reach `ObserverState`.
3.  The write-budget ceiling is hard. When an action is unaffordable it is
    downgraded to IGNORE and the downgrade is counted, rather than silently
    exceeding the budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import features as F
from .ledger import CostConfig, Ledger
from .routers import Router
from .substrates import ACTION_NAMES, IGNORE, N_ACTIONS, SubstrateBank, SubstrateConfig, utility
from .world import CLASS_NAMES, World


@dataclass
class Decision:
    t: int
    key_id: int
    action: int
    feats: np.ndarray
    hidden_class: int          # evaluator-only, for diagnostics and oracle
    ideal_action: int          # evaluator-only
    attributed_utility: float = 0.0
    n_attributed_queries: int = 0
    resident_steps: int = 0


@dataclass
class RolloutResult:
    router_name: str
    legal: bool
    seed: int
    task_utility: float
    forgetting: float
    objective: float
    n_queries: int
    utility_by_class: dict[str, float]
    action_histogram: dict[str, int]
    action_probs: np.ndarray
    confusion: np.ndarray                       # [hidden_class, action], diagnostic only
    ledger: dict
    decisions: list[Decision] = field(default_factory=list)
    audit_curve: list[tuple[int, float]] = field(default_factory=list)
    episodic_evictions: int = 0
    first_encounter_actions: list[tuple[int, int]] = field(default_factory=list)  # (hidden_class, action)


def rollout(
    world: World,
    router: Router,
    cost_cfg: CostConfig,
    sub_cfg: SubstrateConfig,
    seed: int,
    audit_every: int = 100,
    keep_decisions: bool = False,
    discount: float = 0.999,
) -> RolloutResult:
    rng = np.random.default_rng(seed)
    bank = SubstrateBank(sub_cfg)
    ledger = Ledger(cost_cfg)
    obs = F.ObserverState(lifetime=world.config.lifetime)
    router.reset_episode()

    decisions: list[Decision] = []
    last_decision_for_key: dict[int, list[int]] = {}

    total_u = 0.0
    n_q = 0
    u_by_class = {name: [0.0, 0] for name in CLASS_NAMES.values()}
    confusion = np.zeros((4, N_ACTIONS), dtype=int)
    hist = {name: 0 for name in ACTION_NAMES.values()}
    first_enc: list[tuple[int, int]] = []

    best_probe: dict[int, float] = {}
    last_probe: dict[int, float] = {}
    audit_curve: list[tuple[int, float]] = []

    ev_idx = 0
    events = world.events
    ceiling = cost_cfg.write_element_ceiling

    for t in range(world.config.lifetime):
        bank.step()
        ledger.charge_storage_step(len(bank.episodic))

        while ev_idx < len(events) and events[ev_idx].t == t:
            e = events[ev_idx]
            ev_idx += 1

            if e.kind == "WRITE":
                remaining = 1.0 if ceiling is None else max(0.0, 1.0 - ledger.write_elements / ceiling)
                feats = F.extract(
                    key_id=e.key_id, key=e.key, value=e.value, t=t,
                    obs=obs, bank=bank, write_budget_remaining=remaining,
                )
                ledger.charge_router(router.n_params)

                privileged = None
                if router.privileged_fields:
                    privileged = {k: getattr(e, k) for k in router.privileged_fields}

                is_first = e.key_id not in obs.traces
                action = router.act(feats, rng, privileged)
                if not ledger.can_afford(action):
                    action = IGNORE
                    ledger.forced_ignores_budget += 1

                ledger.charge_write(action)
                bank.apply(action, e.key_id, e.key, e.value, t)
                hist[ACTION_NAMES[action]] += 1
                confusion[e.hidden_class, action] += 1
                if is_first:
                    first_enc.append((e.hidden_class, action))

                d = Decision(t, e.key_id, action, feats, e.hidden_class, e.ideal_action)
                decisions.append(d)
                last_decision_for_key.setdefault(e.key_id, []).append(len(decisions) - 1)
                obs.observe_write(e.key_id, e.value, t)

            else:  # QUERY
                answer, path = bank.recall(e.key_id, e.key, t)
                u = utility(answer, e.value)
                ledger.charge_read(len(bank.episodic), path)
                total_u += u
                n_q += 1
                cname = CLASS_NAMES[e.hidden_class]
                u_by_class[cname][0] += u
                u_by_class[cname][1] += 1
                obs.observe_query(e.key_id, t, u)

                # Credit assignment: attribute this outcome back to the write
                # decisions that could have caused it, discounted by delay.
                for di in last_decision_for_key.get(e.key_id, []):
                    d = decisions[di]
                    if d.t <= t:
                        d.attributed_utility += (discount ** (t - d.t)) * u
                        d.n_attributed_queries += 1

        if audit_every and t > 0 and t % audit_every == 0:
            live = world.live_keys(t)
            vals = []
            for kid in live:
                truth = world.true_value(kid, t)
                if truth is None:
                    continue
                pu = utility(bank.probe(kid, world.keys[kid]), truth)
                vals.append(pu)
                best_probe[kid] = max(best_probe.get(kid, 0.0), pu)
                last_probe[kid] = pu
            if vals:
                audit_curve.append((t, float(np.mean(vals))))

    task_utility = total_u / max(1, n_q)
    regressions = [max(0.0, best_probe[k] - last_probe.get(k, 0.0)) for k in best_probe]
    forgetting = float(np.mean(regressions)) if regressions else 0.0

    obj = ledger.objective(task_utility, forgetting, world.config.lifetime, sub_cfg.episodic_capacity)
    total_actions = max(1, sum(hist.values()))
    probs = np.array([hist[ACTION_NAMES[a]] for a in range(N_ACTIONS)], dtype=float) / total_actions

    return RolloutResult(
        router_name=router.name,
        legal=router.legal,
        seed=seed,
        task_utility=task_utility,
        forgetting=forgetting,
        objective=obj,
        n_queries=n_q,
        utility_by_class={k: (v[0] / v[1] if v[1] else float("nan")) for k, v in u_by_class.items()},
        action_histogram=hist,
        action_probs=probs,
        confusion=confusion,
        ledger=ledger.to_dict(world.config.lifetime, sub_cfg.episodic_capacity),
        decisions=decisions if keep_decisions else [],
        audit_curve=audit_curve,
        episodic_evictions=bank.episodic.evictions,
        first_encounter_actions=first_enc,
    )
