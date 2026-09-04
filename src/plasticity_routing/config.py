"""The frozen EXP-001 configuration.

Provenance, all on **development seeds 11, 12, 13**:

1. `scripts/calibrate_world.py` swept world/substrate parameters against the
   admissibility criteria, using only the depth-agnostic controls, the fixed
   heuristic, and class-conditional ORACLE mappings.
2. That sweep's optima were non-bijective, so the search was extended by hand
   over `n_stable_keys`, `n_local_slots`, `fast_lr` and `fast_decay` until an
   admissible bijective configuration was found. The decisive change was making
   the stable key population exceed episodic capacity, which is what gives the
   durable parametric substrate a role that exact storage cannot fill.
3. `routers.search_best_mapping(method="exhaustive")` derived the ORACLE mapping
   over all 256 class-conditional mappings.
4. `scripts/sensitivity.py` verified criterion C6 in the neighbourhood.
5. `scripts/calibrate_heuristic.py` froze the heuristic at its grid argmax.

`LearnedRouter` was **never executed** at any stage of 1-4, so the benchmark
cannot have been tuned to favour the method under test. Step 5 tunes the
*comparator*, in the direction that makes the test harder.

Confirmatory seeds are disjoint from the development seeds. Changing anything in
this module after protocol freeze requires a logged pre-result amendment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .ledger import CostConfig
from .substrates import EPISODIC, FAST, IGNORE, SLOW
from .train import ESConfig, TrainConfig
from .world import WorldConfig

#: Development seeds. Used for calibration and for training the learned router.
DEV_SEEDS = (11, 12, 13)

#: Independent policy initialisations trained on DEV_SEEDS. The policy carried
#: into a confirmatory run is the one with the best **development** objective
#: (`scripts/select_policy.py`). Policy-seed variance turned out to dominate the
#: ES budget -- at 60 generations three seeds spanned 0.386 to 0.502 -- so
#: reporting a single seed would have reported an accident of initialisation.
POLICY_SEEDS = (0, 1, 2)

#: Confirmatory seeds. Disjoint from DEV_SEEDS. Frozen before any confirmatory run.
CONFIRMATORY_SEEDS = (20260902, 20260903, 20260904, 20260905, 20260906)

#: One-shot audit seeds for the L5b cross-world negative control (Amendment L).
#: Deliberately neither training nor confirmatory. R was selected for real-dev
#: performance and S for shuffled-dev performance, so evaluating the cross on
#: DEV_SEEDS would be selection-biased in R's favour; and spending confirmatory
#: seeds on a validity gate would consume the held-out set before the experiment.
#: Frozen, with the L5b criterion, before any cross-world number was inspected.
AUDIT_SEEDS = tuple(range(91001, 91033))

#: SHA-256 of the two selected policies as of Amendment L. The L5b audit loads
#: these artefacts and must never retrain: retraining inside the audit would let
#: the policies drift with the source tree and silently change what is being
#: compared. `tests/test_l5b.py` asserts the files still hash to these values.
SELECTED_POLICY_SHA256 = {
    "real_selected.json": "7cdc29ecd7492a3c7a928d407cb87f39b04c19a9ce189ae81405d69adbaf9c40",
    "shuffled_selected.json": "cf81b58a36c77ccd799c4862b02f383db31a23476a132b70d4450b6636c28dbe",
}

#: SHA-256 of the matched-budget comparator artefact
#: (`results/heuristic_matched_search.json`), whose `best_params` define
#: `HEURISTIC_EXT`. Pinned at Amendment M for the same reason as the policies:
#: the comparator is an input to the confirmatory comparison, it lives outside
#: the source-tree fingerprint, and a silent change to it would alter what H1 is
#: measured against.
MATCHED_HEURISTIC_SHA256 = "3ef29dd377193e8fa873c5e3b130f749046f7b1dfb21dc099fc7bb6633bb3791"


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
    train=ESConfig(generations=100),
    # NOISE -> IGNORE, ONE_OFF -> FAST, LOCAL -> EPISODIC, STABLE -> SLOW
    oracle_mapping=(IGNORE, FAST, EPISODIC, SLOW),
    designed_mapping=(IGNORE, EPISODIC, FAST, SLOW),
)


#: Label from which the EXP-001R replication seeds are derived. Deriving them
#: from a fixed string rather than typing a list makes their provenance
#: machine-checkable: nobody can claim afterwards that the list was chosen to
#: suit a result. `tests/test_config_seeds.py` re-derives and compares.
REPLICATION_SEED_LABEL = "EXP-001R-replication-seeds-v1"
REPLICATION_SEED_COUNT = 32


def derive_seeds(label: str, count: int, lo: int = 40_000_000, hi: int = 49_999_999) -> tuple[int, ...]:
    """Deterministically derive `count` distinct seeds from `label`.

    The range is chosen to be disjoint from every seed set already in use
    (development, audit, confirmatory), and disjointness is asserted by tests
    rather than assumed.
    """
    import hashlib

    out: list[int] = []
    i = 0
    span = hi - lo + 1
    while len(out) < count:
        digest = hashlib.sha256(f"{label}:{i}".encode()).digest()
        v = lo + (int.from_bytes(digest[:8], "big") % span)
        if v not in out:
            out.append(v)
        i += 1
    return tuple(out)


#: EXP-001R replication seeds. Untouched: not development, not audit, not
#: confirmatory. Frozen before the replication runs.
REPLICATION_SEEDS = derive_seeds(REPLICATION_SEED_LABEL, REPLICATION_SEED_COUNT)


# ---------------------------------------------------------------------------
# EXP-002 (language model) -- design-stage seed provenance only.
#
# No EXP-002 code exists and LM execution is not authorized. These seeds are
# declared now so that the substrate-sufficiency preregistration can name them
# before any of it is implemented, and so their derivation is checkable rather
# than asserted.
# ---------------------------------------------------------------------------

EXP002_DEV_SEED_LABEL = "EXP-002-substrate-sufficiency-development-v1"
EXP002_DEV_SEED_COUNT = 5

#: Development seeds for the EXP-002 substrate-sufficiency gates. Permanently
#: excluded from any later EXP-002 comparative or confirmatory run: once a
#: sufficiency grid has been inspected on a seed, that seed can no longer
#: support an unbiased comparison.
EXP002_DEV_SEEDS = derive_seeds(
    EXP002_DEV_SEED_LABEL, EXP002_DEV_SEED_COUNT, lo=60_000_000, hi=69_999_999
)
