"""Adaptive Plasticity Routing -- the ALLOCATE stage of CCS.

Narrow question: can an agent learn a resource-aware write-depth policy from
future task utility, under interference and storage/compute constraints?

Status: pre-result. No empirical claim is made by this package.
"""

__version__ = "0.1.0"

from .ledger import CostConfig, Ledger
from .routers import (
    ExtendedHeuristicRouter,
    HeuristicRouter,
    LearnedRouter,
    OracleRouter,
    PrivilegedTaskIdRouter,
    RandomMatchedRouter,
    constant_routers,
)
from .substrates import ACTION_NAMES, SubstrateBank, SubstrateConfig
from .world import CLASS_NAMES, WorldConfig, build_world

__all__ = [
    "CostConfig", "Ledger", "SubstrateBank", "SubstrateConfig", "ACTION_NAMES",
    "WorldConfig", "build_world", "CLASS_NAMES", "HeuristicRouter", "LearnedRouter",
    "OracleRouter", "PrivilegedTaskIdRouter", "RandomMatchedRouter", "constant_routers",
    "ExtendedHeuristicRouter",
]
