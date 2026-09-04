"""Seed-set provenance and disjointness.

Every seed set in this project has a distinct role, and a seed that leaks
between roles silently invalidates whatever it leaks into.
"""

from __future__ import annotations

from plasticity_routing.config import (
    AUDIT_SEEDS, CONFIRMATORY_SEEDS, DEV_SEEDS, EXP002_DEV_SEED_COUNT,
    EXP002_DEV_SEED_LABEL, EXP002_DEV_SEEDS, REPLICATION_SEED_COUNT,
    REPLICATION_SEED_LABEL, REPLICATION_SEEDS, derive_seeds,
)

ALL_SETS = {
    "dev": set(DEV_SEEDS),
    "audit": set(AUDIT_SEEDS),
    "confirmatory": set(CONFIRMATORY_SEEDS),
    "replication": set(REPLICATION_SEEDS),
    "exp002_dev": set(EXP002_DEV_SEEDS),
}


def test_every_seed_set_is_pairwise_disjoint():
    names = sorted(ALL_SETS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = ALL_SETS[a] & ALL_SETS[b]
            assert not overlap, f"{a} and {b} share seeds: {sorted(overlap)}"


def test_replication_seeds_are_reproducible_from_their_label():
    """Provenance must be checkable: the list is derived, not chosen."""
    assert derive_seeds(REPLICATION_SEED_LABEL, REPLICATION_SEED_COUNT) == REPLICATION_SEEDS


def test_replication_seed_set_is_the_declared_size_and_distinct():
    assert len(REPLICATION_SEEDS) == REPLICATION_SEED_COUNT == 32
    assert len(set(REPLICATION_SEEDS)) == 32


def test_derivation_is_label_sensitive():
    assert derive_seeds("a", 8) != derive_seeds("b", 8)
    assert derive_seeds("a", 8) == derive_seeds("a", 8)


def test_replication_seeds_sit_outside_every_other_range():
    lo, hi = min(REPLICATION_SEEDS), max(REPLICATION_SEEDS)
    for name in ("dev", "audit", "confirmatory"):
        other = ALL_SETS[name]
        assert not (lo <= min(other) <= hi or lo <= max(other) <= hi), \
            f"replication range overlaps {name}"


def test_exp002_development_seeds_are_reproducible_from_their_label():
    assert derive_seeds(
        EXP002_DEV_SEED_LABEL, EXP002_DEV_SEED_COUNT, lo=60_000_000, hi=69_999_999
    ) == EXP002_DEV_SEEDS


def test_exp002_development_seeds_are_declared_size_and_distinct():
    assert len(EXP002_DEV_SEEDS) == EXP002_DEV_SEED_COUNT == 5
    assert len(set(EXP002_DEV_SEEDS)) == 5


def test_derive_seeds_respects_its_range():
    s = derive_seeds("range-check", 40, lo=100, hi=999)
    assert all(100 <= x <= 999 for x in s)
    assert len(set(s)) == 40
