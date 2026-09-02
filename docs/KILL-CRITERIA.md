# Falsification and kill criteria

Preregistered. A criterion that fires is reported, not worked around.

## Benchmark-invalidating criteria

If any of these fire, **no comparison on that configuration means anything**.
Recalibration is permitted only on development seeds and must be logged as a
pre-result amendment.

| ID | Fires when | Meaning |
|---|---|---|
| **K3** | any leakage test L1–L7 fails | the router had information it could not have at decision time; the run is void |
| **K4** | a single-depth control (`ALL_EPISODIC` / `ALL_FAST` / `ALL_SLOW`) matches or beats the learned router | the world is not depth-stratified; there is nothing for routing to decide |
| **K5** | every arm exceeds 0.95 task utility | ceiling effect |
| **K6** | ORACLE objective < 0.45 | floor effect; the world is unsolvable or dominated by noise |
| **K8** | the optimal class→action mapping is not a bijection | the declared four-action space is not justified by the benchmark |

## Hypothesis-falsifying criteria

These do **not** invalidate the experiment. They falsify the hypothesis, and a
negative result here is a publishable outcome, not a failure to be retried.

| ID | Fires when | Conclusion |
|---|---|---|
| **K1** | `LEARNED` does not exceed `HEURISTIC` by a paired-bootstrap 95% CI excluding zero | **learned routing adds nothing over a fixed rule.** This replicates Yoon (2026, arXiv:2606.30067) in a new setting and is reported as the headline result |
| **K2** | `LEARNED` ≈ `RANDOM_MATCHED` (CI includes zero) | any apparent gain is budget-allocation, not routing |
| **K7** | `LEARNED`'s advantage over `HEURISTIC` disappears under `CAPACITY_MATCHED` or under compute matching | the gain was capacity or compute, not routing |
| **K9** | `PRIVILEGED_TASKID` − `HEURISTIC` ≥ `LEARNED` − `HEURISTIC` | privileged context identity explains as much as learned inference; the interesting quantity was never routing |
| **K10** | the time-shuffled control (L5) still beats `RANDOM_MATCHED` | the router is exploiting structure other than genuine future utility |

## Prior

The prior on K1 *not* firing is deliberately set low. Yoon (2026) reports a
simple similarity rule matching or beating a learned RL allocation controller
under fixed capacity, and that is the closest published comparison to this one.
The primary comparator is therefore the fixed heuristic, calibrated on the same
grid budget as the learned router's hyperparameters. **Deliberately weakening
the heuristic would be the most direct way to manufacture a positive result and
is prohibited.**

## Success criterion for H1

`LEARNED` beats `HEURISTIC` on the objective with a paired-bootstrap 95% CI
excluding zero, on held-out confirmatory seeds, **and** survives K2, K7, K9, and
K10, **and** every leakage test passes.

Anything less is a negative or inconclusive result and is reported as such.

## Protocol commitments

*   Preregister before inspecting confirmatory results.
*   Development and confirmatory seeds are disjoint and frozen in
    `src/plasticity_routing/config.py`.
*   Calibration uses controls and the oracle only; the learned router is never
    run during benchmark calibration.
*   Do not tune thresholds, costs, learning rates, or the world to make the
    proposed method win.
*   Preserve all negative results, including the ones already recorded in
    [`BENCHMARK.md`](BENCHMARK.md) §6.
*   Inspect in order: leakage → mechanical validity → resource budgets →
    comparative metrics. Never the reverse.
