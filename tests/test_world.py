"""SDW-1 design-contract tests.

These test the *benchmark*, not any method. If they fail the benchmark is
broken and no result computed on it means anything.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plasticity_routing.world import (
    CLASS_NAMES, LOCAL, NOISE, ONE_OFF, STABLE, WorldConfig, build_world, class_histogram,
)


@pytest.fixture(scope="module")
def world():
    return build_world(WorldConfig(), seed=11)


def test_determinism():
    a = build_world(WorldConfig(), seed=7)
    b = build_world(WorldConfig(), seed=7)
    assert len(a.events) == len(b.events)
    for x, y in zip(a.events, b.events):
        assert (x.kind, x.t, x.key_id, x.hidden_class) == (y.kind, y.t, y.key_id, y.hidden_class)
        assert np.allclose(x.key, y.key)


def test_seeds_differ():
    a = build_world(WorldConfig(), seed=7)
    b = build_world(WorldConfig(), seed=8)
    assert [e.key_id for e in a.events] != [e.key_id for e in b.events]


def test_events_are_time_ordered(world):
    ts = [e.t for e in world.events]
    assert ts == sorted(ts)


def test_all_four_classes_present(world):
    hist = class_histogram(world)
    assert all(hist[name] > 0 for name in CLASS_NAMES.values()), hist


def test_noise_is_never_queried(world):
    """If noise were queried, IGNORE would not be its correct action."""
    assert not any(e.kind == "QUERY" and e.hidden_class == NOISE for e in world.events)


def test_one_off_queried_at_most_once(world):
    counts: dict[int, int] = {}
    for e in world.events:
        if e.kind == "QUERY" and e.hidden_class == ONE_OFF:
            counts[e.key_id] = counts.get(e.key_id, 0) + 1
    assert counts, "no ONE_OFF queries generated"
    assert max(counts.values()) == 1


def test_one_off_and_noise_keys_are_never_repeated(world):
    """Single-use classes must be genuinely single-use, or recurrence leaks class."""
    seen: dict[int, int] = {}
    for e in world.events:
        if e.kind == "WRITE" and e.hidden_class in (NOISE, ONE_OFF):
            seen[e.key_id] = seen.get(e.key_id, 0) + 1
    assert seen and max(seen.values()) == 1


def test_local_values_are_revised_across_regimes(world):
    """A LOCAL key must take a different value in a later regime.

    This is what makes durable storage of a local rule actively harmful, and it
    is the signal a router can legally use to tell LOCAL from STABLE.
    """
    revised = 0
    for kid, cls in world.class_of_key.items():
        if cls != LOCAL:
            continue
        tl = world.truth_timeline[kid]
        for (_, v0), (_, v1) in zip(tl, tl[1:]):
            if float(np.dot(v0, v1)) < 0.9:
                revised += 1
                break
    assert revised > 0


def test_stable_values_never_change(world):
    for kid, cls in world.class_of_key.items():
        if cls != STABLE:
            continue
        tl = world.truth_timeline[kid]
        assert len(tl) == 1, "a STABLE key must have exactly one lifetime value"


def test_stable_queries_reach_far_into_the_future(world):
    """STABLE must be probed long after its write, or durability is untested."""
    last_write = {}
    gaps = []
    for e in world.events:
        if e.hidden_class != STABLE:
            continue
        if e.kind == "WRITE":
            last_write[e.key_id] = e.t
        elif e.key_id in last_write:
            gaps.append(e.t - last_write[e.key_id])
    assert gaps and max(gaps) > 400, f"max STABLE write->query gap only {max(gaps) if gaps else 0}"


def test_query_ground_truth_matches_timeline(world):
    for e in world.events:
        if e.kind == "QUERY":
            truth = world.true_value(e.key_id, e.t)
            assert truth is not None
            assert np.allclose(truth, e.value)


def test_first_encounter_is_observationally_ambiguous():
    """The core design contract.

    Key and value vectors must be drawn from the same distribution for every
    class, so that a first encounter carries no class information. Without this
    the task degenerates into disguised supervised classification.
    """
    w = build_world(replace(WorldConfig(), lifetime=6000), seed=3)
    first_by_class: dict[int, list[np.ndarray]] = {c: [] for c in CLASS_NAMES}
    seen: set[int] = set()
    for e in w.events:
        if e.kind == "WRITE" and e.key_id not in seen:
            seen.add(e.key_id)
            first_by_class[e.hidden_class].append(e.key)

    for c, vecs in first_by_class.items():
        assert len(vecs) >= 5, f"too few first encounters for {CLASS_NAMES[c]}"
        arr = np.stack(vecs)
        assert np.allclose(np.linalg.norm(arr, axis=1), 1.0, atol=1e-9)

    # Permutation test: if class carried any information about the first-sight
    # key vector, the per-class mean vectors would separate more than they do
    # under randomly reassigned labels. A closed-form tolerance is easy to get
    # wrong (the norm of a mean of n unit vectors is ~1/sqrt(n), not a
    # per-coordinate quantity), so the null is generated by permutation.
    labels = np.concatenate([[c] * len(v) for c, v in first_by_class.items()])
    pooled = np.concatenate([np.stack(v) for v in first_by_class.values()])

    def max_class_mean_norm(lab: np.ndarray) -> float:
        return max(float(np.linalg.norm(pooled[lab == c].mean(axis=0))) for c in first_by_class)

    observed = max_class_mean_norm(labels)
    rng = np.random.default_rng(0)
    null = np.array([max_class_mean_norm(rng.permutation(labels)) for _ in range(400)])
    p_value = float((null >= observed).mean())
    assert p_value > 0.01, (
        f"first-encounter key vectors are class-dependent (permutation p={p_value:.4f}, "
        f"observed max class-mean norm {observed:.3f} vs null mean {null.mean():.3f})"
    )


def test_recurrence_separates_single_use_from_recurrent_classes():
    """The legal signal must actually exist: repeats imply LOCAL or STABLE."""
    w = build_world(WorldConfig(), seed=5)
    seen: set[int] = set()
    for e in w.events:
        if e.kind != "WRITE":
            continue
        if e.key_id in seen:
            assert e.hidden_class in (LOCAL, STABLE)
        seen.add(e.key_id)
