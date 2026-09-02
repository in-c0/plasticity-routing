"""Metrics, statistics, and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from plasticity_routing.metrics import (
    confusion_table, first_encounter_class_dependence, paired_bootstrap,
    routing_agreement_diagnostic,
)


def test_paired_bootstrap_detects_a_real_difference():
    a = [0.5, 0.52, 0.49, 0.51, 0.50]
    b = [0.3, 0.31, 0.29, 0.30, 0.32]
    st = paired_bootstrap(a, b, n=4000, seed=0)
    assert st["mean_diff"] > 0.15
    assert st["excludes_zero"]
    assert st["ci95"][0] < st["mean_diff"] < st["ci95"][1]


def test_paired_bootstrap_reports_no_difference_when_there_is_none():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0, 1, 30))
    b = list(rng.normal(0, 1, 30))
    st = paired_bootstrap(a, b, n=4000, seed=0)
    assert not st["excludes_zero"]


def test_paired_bootstrap_requires_matched_samples():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_routing_agreement_is_the_confusion_trace():
    c = np.array([[5, 1, 0, 0], [0, 4, 0, 0], [0, 0, 3, 1], [1, 0, 0, 5]])
    assert routing_agreement_diagnostic(c) == pytest.approx(17 / 20)
    assert np.isnan(routing_agreement_diagnostic(np.zeros((4, 4), dtype=int)))


def test_confusion_table_is_labelled():
    c = np.zeros((4, 4), dtype=int)
    c[3, 3] = 7
    t = confusion_table(c)
    assert t["STABLE"]["SLOW"] == 7
    assert set(t) == {"NOISE", "ONE_OFF", "LOCAL", "STABLE"}


def test_first_encounter_independence_on_independent_data():
    rng = np.random.default_rng(0)
    pairs = [(int(rng.integers(4)), int(rng.integers(4))) for _ in range(1200)]
    st = first_encounter_class_dependence(pairs)
    assert st["p_value"] > 0.05
    assert st["n"] == 1200


def test_first_encounter_independence_detects_a_deterministic_leak():
    pairs = [(c, c) for c in range(4) for _ in range(200)]
    st = first_encounter_class_dependence(pairs)
    assert st["p_value"] < 0.05
    assert st["mutual_information_bits"] == pytest.approx(2.0, abs=1e-6)


def test_first_encounter_independence_handles_empty_input():
    st = first_encounter_class_dependence([])
    assert st["n"] == 0 and st["p_value"] == 1.0
