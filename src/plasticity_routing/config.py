"""The frozen EXP-001 configuration.

Selected by `scripts/calibrate_world.py` on **development seeds 11, 12, 13**
using only the depth-agnostic controls, the fixed heuristic, and class-conditional
ORACLE mappings. `LearnedRouter` was never executed during calibration, so the
benchmark cannot have been tuned to favour the method under test.

Confirmatory seeds are disjoint from the development seeds and are listed in
`experiments/EXP-001-PREREG.md`. Changing anything in this module after protocol
freeze requires a logged pre-result amendment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .ledger import CostConfig
from .substrates import EPISODIC, FAST, IGNORE, SLOW
from .train import ESConfig, TrainConfig
from .world import WorldConfig

#: Development seeds. Used for calibration and for training the learned router.
DEV_SEEDS = (11, 12, 13)

#: Confirmatory seeds. Disjoint from DEV_SEEDS. Frozen before any confirmatory run.
CONFIRMATORY_SEEDS = (20260902, 20260903, 20260904, 20260905, 20260906)


@dataclass(frozen=True)
class ExperimentConfig:
    world: WorldConfig
    substrate: object
    cost: CostConfig
    train: ESConfig
    #: Empirically optimal class->action mapping, from exhaustive search on DEV_SEEDS.
    oracle_mapping: tuple[int, int, int, int]
    #: The mapping the benchmark was designed around, retained because it was
    #: falsified during calibration. See docs/ARCHITECTURE.md Amendment B.
    designed_mapping: tuple[int, int, int, int]


def _substrate():
    from .substrates import SubstrateConfig

    return replace(
        SubstrateConfig(),
        key_dim=96,
        value_dim=16,
        episodic_capacity=24,
        fast_lr=1.0,
        fast_decay=0.997,
        slow_lr=0.7,
        slow_consolidation_steps=4,
    )


EXP001 = ExperimentConfig(
    world=replace(
        WorldConfig(),
        key_dim=96,
        value_dim=16,
        lifetime=3000,
        regime_len=300,
        n_stable_keys=70,
        n_local_slots=44,
        class_prior=(0.26, 0.26, 0.28, 0.20),
        one_off_delay_min=4,
        one_off_delay_max=60,
        stable_query_horizon=2200,
    ),
    substrate=_substrate(),
    cost=replace(CostConfig(), key_dim=96, value_dim=16, write_element_ceiling=3_000_000),
    train=ESConfig(),
    # NOISE -> IGNORE, ONE_OFF -> FAST, LOCAL -> EPISODIC, STABLE -> SLOW
    oracle_mapping=(IGNORE, FAST, EPISODIC, SLOW),
    designed_mapping=(IGNORE, EPISODIC, FAST, SLOW),
)
