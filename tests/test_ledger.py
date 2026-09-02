"""Resource accounting must be exact, shared, and enforced."""

from __future__ import annotations

from dataclasses import replace

import pytest

from plasticity_routing.agent import rollout
from plasticity_routing.ledger import CostConfig, Ledger
from plasticity_routing.routers import constant_routers
from plasticity_routing.substrates import EPISODIC, FAST, IGNORE, SLOW, SubstrateConfig
from plasticity_routing.world import WorldConfig, build_world


def test_write_costs_follow_the_declared_currency():
    cfg = CostConfig(key_dim=32, value_dim=8, slow_write_multiplier=4)
    led = Ledger(cfg)
    assert led.write_cost(IGNORE) == 0
    assert led.write_cost(EPISODIC) == 32 + 8   # an entry stores its key and its value
    assert led.write_cost(FAST) == 32 * 8
    assert led.write_cost(SLOW) == 32 * 8 * 4


def test_depth_ordering_of_write_cost():
    led = Ledger(CostConfig())
    assert led.write_cost(IGNORE) < led.write_cost(EPISODIC) < led.write_cost(FAST) < led.write_cost(SLOW)


def test_ceiling_is_enforced_and_downgrades_are_counted():
    wcfg = replace(WorldConfig(), lifetime=800)
    ccfg = replace(CostConfig(), write_element_ceiling=50_000)
    scfg = SubstrateConfig()
    w = build_world(wcfg, seed=1)
    res = rollout(w, constant_routers()["ALL_SLOW"], ccfg, scfg, seed=1)
    assert res.ledger["write_elements"] <= ccfg.write_element_ceiling
    assert res.ledger["write_ceiling_exceeded"] is False
    assert res.ledger["forced_ignores_budget"] > 0, "ceiling never bound; test is vacuous"


def test_no_arm_exceeds_the_shared_ceiling():
    wcfg, scfg, ccfg = WorldConfig(), SubstrateConfig(), CostConfig()
    w = build_world(wcfg, seed=2)
    for name, router in constant_routers().items():
        res = rollout(w, router, ccfg, scfg, seed=2)
        assert res.ledger["write_elements"] <= ccfg.write_element_ceiling, name


def test_ignore_arm_spends_nothing_on_writes():
    w = build_world(WorldConfig(), seed=3)
    res = rollout(w, constant_routers()["ALL_IGNORE"], CostConfig(), SubstrateConfig(), seed=3)
    assert res.ledger["write_elements"] == 0
    assert res.ledger["writes_by_action"]["IGNORE"] > 0


def test_episodic_read_is_a_similarity_search():
    """Charging O(|E|) instead of O(|E| * key_dim) makes a huge exact store
    look cheap to query, which trivialises the capacity confound."""
    cfg = CostConfig(key_dim=32, value_dim=8)
    led = Ledger(cfg)
    led.charge_read(episodic_len=100, path="episodic")
    assert led.read_compute_elements == 100 * 32


def test_parametric_read_costs_search_plus_matrix():
    cfg = CostConfig(key_dim=32, value_dim=8)
    led = Ledger(cfg)
    led.charge_read(episodic_len=10, path="parametric")
    assert led.read_compute_elements == 10 * 32 + 2 * 32 * 8


def test_storage_is_charged_as_an_occupancy_integral():
    """Episodic storage must grow with residency; constants must not differentiate."""
    w = build_world(replace(WorldConfig(), lifetime=1000), seed=4)
    ccfg, scfg = CostConfig(), SubstrateConfig()
    epi = rollout(w, constant_routers()["ALL_EPISODIC"], ccfg, scfg, seed=4)
    ign = rollout(w, constant_routers()["ALL_IGNORE"], ccfg, scfg, seed=4)
    assert epi.ledger["storage_element_steps"] > ign.ledger["storage_element_steps"]


def test_router_compute_is_reported_separately_and_included_in_total():
    from plasticity_routing.routers import HeuristicRouter

    w = build_world(WorldConfig(), seed=5)
    res = rollout(w, HeuristicRouter(), CostConfig(), SubstrateConfig(), seed=5)
    led = res.ledger
    assert led["router_compute_elements"] > 0
    assert led["total_compute_elements"] == led["read_compute_elements"] + led["router_compute_elements"]


def test_zero_param_router_still_reports_a_router_compute_field():
    w = build_world(replace(WorldConfig(), lifetime=400), seed=6)
    res = rollout(w, constant_routers()["ALL_FAST"], CostConfig(), SubstrateConfig(), seed=6)
    assert res.ledger["router_compute_elements"] == 0


def test_objective_decomposes_as_declared():
    w = build_world(replace(WorldConfig(), lifetime=600), seed=7)
    ccfg, scfg = CostConfig(), SubstrateConfig()
    res = rollout(w, constant_routers()["ALL_FAST"], ccfg, scfg, seed=7)
    p = res.ledger["penalties"]
    expected = (res.task_utility - ccfg.lam_forget * res.forgetting
                - p["storage"] - p["write"] - p["compute"])
    assert res.objective == pytest.approx(expected)


def test_normalized_terms_are_nonnegative():
    w = build_world(replace(WorldConfig(), lifetime=600), seed=8)
    for name, router in constant_routers().items():
        res = rollout(w, router, CostConfig(), SubstrateConfig(), seed=8)
        for k, v in res.ledger["normalized"].items():
            assert v >= 0.0, (name, k, v)


def test_every_arm_shares_one_cost_table():
    """There must be no per-arm cost configuration path."""
    import inspect

    from plasticity_routing import agent

    src = inspect.getsource(agent.rollout)
    assert "cost_cfg" in src
    assert src.count("Ledger(") == 1
