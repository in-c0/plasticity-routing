"""Resource accounting.

Every arm shares one cost table and one hard ceiling. Resource accounting is
mandatory: a run that cannot report its write / storage / compute consumption,
or that exceeds a ceiling, is invalid and may not be compared.

Costing conventions (see docs/RESOURCE-NORMALIZATION.md):

*   Writes are counted in **parameter/value elements exposed to a write**, the
    same currency used by the sibling `state-promotion` track. An episodic
    entry must store its *key* as well as its value, or it could never be
    retrieved -- so it costs `key_dim + value_dim`, not `value_dim`.
*   Storage is charged as an **occupancy integral**: footprint summed over
    timesteps. Parametric substrates have a constant footprint; the episodic
    store's grows, which is precisely the trade-off under test.
*   Reading the episodic store is a similarity search over its contents and
    therefore costs `|E| * key_dim`, not `|E|`. Charging O(|E|) understates a
    large exact store so badly that "just make the episodic store bigger" wins
    trivially, which is precisely the capacity confound this track must avoid.
*   Router decision compute is counted **separately and additionally**, never
    hidden inside "compute matched" claims.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

from .substrates import IGNORE, EPISODIC, FAST, SLOW


@dataclass(frozen=True)
class CostConfig:
    key_dim: int = 48
    value_dim: int = 16
    slow_write_multiplier: int = 4

    # Hard ceilings for one lifetime. None disables the ceiling (dev only).
    write_element_ceiling: int | None = 1_400_000
    storage_element_ceiling: int | None = None

    # Objective weights. Applied to *normalised* resource terms in [0, 1] so
    # that every term is commensurate with mean per-query utility in [0, 1].
    lam_forget: float = 0.50
    lam_storage: float = 0.10
    lam_write: float = 0.10
    lam_compute: float = 0.05


@dataclass
class Ledger:
    cfg: CostConfig
    write_elements: int = 0
    writes_by_action: dict[str, int] = field(default_factory=lambda: {"IGNORE": 0, "EPISODIC": 0, "FAST": 0, "SLOW": 0})
    storage_element_steps: int = 0
    read_compute_elements: int = 0
    router_compute_elements: int = 0
    forced_ignores_budget: int = 0

    # ---- write side ------------------------------------------------------
    def write_cost(self, action: int) -> int:
        c = self.cfg
        if action == IGNORE:
            return 0
        if action == EPISODIC:
            return c.key_dim + c.value_dim
        if action == FAST:
            return c.key_dim * c.value_dim
        if action == SLOW:
            return c.key_dim * c.value_dim * c.slow_write_multiplier
        raise ValueError(action)

    def can_afford(self, action: int) -> bool:
        if self.cfg.write_element_ceiling is None:
            return True
        return self.write_elements + self.write_cost(action) <= self.cfg.write_element_ceiling

    def charge_write(self, action: int) -> None:
        from .substrates import ACTION_NAMES

        self.write_elements += self.write_cost(action)
        self.writes_by_action[ACTION_NAMES[action]] += 1

    # ---- storage / compute ----------------------------------------------
    def charge_storage_step(self, episodic_len: int) -> None:
        c = self.cfg
        # An episodic entry occupies key + value; the two parametric matrices
        # occupy a constant footprint regardless of how much they hold.
        self.storage_element_steps += (
            episodic_len * (c.key_dim + c.value_dim) + 2 * c.key_dim * c.value_dim
        )

    def charge_read(self, episodic_len: int, path: str) -> None:
        c = self.cfg
        # Retrieval scans the store: |E| key comparisons of width key_dim.
        search = episodic_len * c.key_dim
        if path == "episodic":
            self.read_compute_elements += search
        else:
            self.read_compute_elements += search + 2 * c.key_dim * c.value_dim

    def charge_router(self, n_params: int) -> None:
        self.router_compute_elements += n_params

    # ---- totals ----------------------------------------------------------
    @property
    def total_compute_elements(self) -> int:
        return self.read_compute_elements + self.router_compute_elements

    def normalizers(self, lifetime: int, episodic_capacity: int) -> dict[str, float]:
        """Reference denominators shared by every arm.

        The references describe the *most expensive admissible* behaviour, so
        each normalised term lands in roughly [0, 1] and the objective weights
        are directly interpretable.
        """
        c = self.cfg
        param_footprint = 2 * c.key_dim * c.value_dim
        entry = c.key_dim + c.value_dim
        return {
            "storage": float(lifetime * (episodic_capacity * entry + param_footprint)),
            "write": float(c.write_element_ceiling) if c.write_element_ceiling else 1.0,
            "compute": float(lifetime * (episodic_capacity * c.key_dim + param_footprint)),
        }

    def normalized(self, lifetime: int, episodic_capacity: int) -> dict[str, float]:
        n = self.normalizers(lifetime, episodic_capacity)
        return {
            "storage": self.storage_element_steps / max(1.0, n["storage"]),
            "write": self.write_elements / max(1.0, n["write"]),
            "compute": self.total_compute_elements / max(1.0, n["compute"]),
        }

    def penalties(self, lifetime: int, episodic_capacity: int) -> dict[str, float]:
        c = self.cfg
        z = self.normalized(lifetime, episodic_capacity)
        return {
            "storage": c.lam_storage * z["storage"],
            "write": c.lam_write * z["write"],
            "compute": c.lam_compute * z["compute"],
        }

    def objective(self, task_utility: float, forgetting: float, lifetime: int, episodic_capacity: int) -> float:
        """`future task utility - forgetting penalty - storage cost - write cost - compute cost`."""
        p = self.penalties(lifetime, episodic_capacity)
        return task_utility - self.cfg.lam_forget * forgetting - p["storage"] - p["write"] - p["compute"]

    def to_dict(self, lifetime: int, episodic_capacity: int) -> dict:
        d = {
            "cost_config": asdict(self.cfg),
            "write_elements": self.write_elements,
            "writes_by_action": dict(self.writes_by_action),
            "storage_element_steps": self.storage_element_steps,
            "read_compute_elements": self.read_compute_elements,
            "router_compute_elements": self.router_compute_elements,
            "total_compute_elements": self.total_compute_elements,
            "forced_ignores_budget": self.forced_ignores_budget,
            "normalized": self.normalized(lifetime, episodic_capacity),
            "penalties": self.penalties(lifetime, episodic_capacity),
        }
        if self.cfg.write_element_ceiling is not None:
            d["write_ceiling_exceeded"] = self.write_elements > self.cfg.write_element_ceiling
            d["write_budget_utilization"] = self.write_elements / self.cfg.write_element_ceiling
        return d
