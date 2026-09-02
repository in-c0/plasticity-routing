# EXP-000 — development calibration result

**Classification: `DEV_CALIBRATION`. This is not evidence.**

The learned router was trained *and* evaluated on development seeds
`{11, 12, 13}`. Confirmatory seeds were not touched. EXP-000's purpose is to
show the harness is mechanically valid and to expose design defects before the
protocol is frozen — not to test H1.

Date: 2026-09-02. Config: `src/plasticity_routing/config.py`.
Manifests: `results/run_*.json` (33). Validator: `results/validation.json`
(**PASSED**).

## 1. Validity gates, in protocol order

| Gate | Outcome |
|---|---|
| L1 feature whitelist purity | PASS |
| L2 hidden-label permutation invariance | PASS |
| L3 first-encounter class independence | PASS (all legal routers, all dev seeds) |
| L3-canary — detector catches a deliberate oracle leak | PASS |
| L4 causal ordering / future blindness | PASS |
| L6 privilege declaration | PASS |
| L7 audit probes have no side effects | PASS |
| write ceiling respected by every arm | PASS (max 100.0% for ORACLE, no breaches) |
| manifest validation | PASS |
| K5 ceiling effect (all arms > 0.95 utility) | not fired (max 0.905, ORACLE) |
| K6 floor effect (ORACLE < 0.45) | not fired (ORACLE 0.563) |
| K8 optimal mapping is a bijection | not fired (`[IGNORE, FAST, EPISODIC, SLOW]`) |
| C6 local stability of the optimum | holds at 15/16 neighbourhood points; single boundary at `fast_decay=0.999`, two grid steps from the frozen 0.997 |

## 2. Arms

Mean over 3 development seeds. `routing agreement` is the fraction of decisions
matching the hidden ideal action — an evaluator-only **diagnostic**, never a
training signal or a success criterion.

| arm | legal | objective | utility | forgetting | writes |
|---|---|---|---|---|---|
| `ORACLE` | **no** | 0.563 ± 0.016 | 0.905 | 0.246 | 2,999,984 |
| `DESIGNED_MAPPING` | **no** | 0.430 ± 0.036 | 0.822 | 0.295 | 2,999,952 |
| `PRIVILEGED_TASKID` | **no** | 0.405 ± 0.024 | 0.798 | 0.385 | 1,735,205 |
| `HEURISTIC` | yes | 0.405 ± 0.024 | 0.798 | 0.385 | 1,736,741 |
| `LEARNED` | yes | 0.386 ± 0.019 | 0.808 | 0.474 | 1,240,976 |
| `RANDOM_MATCHED` | yes | 0.275 ± 0.036 | 0.733 | 0.562 | 1,226,736 |
| `CAPACITY_MATCHED_EPISODIC` | yes | 0.252 ± 0.035 | 0.712 | 0.637 | 193,387 |
| `ALL_FAST` | yes | 0.230 ± 0.024 | 0.679 | 0.556 | 2,652,160 |
| `ALL_SLOW` | yes | 0.024 ± 0.013 | 0.354 | 0.296 | 2,998,272 |
| `ALL_EPISODIC` | yes | 0.020 ± 0.034 | 0.484 | 0.644 | 193,387 |
| `ALL_IGNORE` | yes | −0.082 ± 0.001 | 0.000 | 0.000 | 0 |

The heuristic here is the **grid argmax** selected by
`scripts/calibrate_heuristic.py` (`seen_threshold=1, revision_tolerance=0.8,
error_floor=0.25`, dev objective 0.4046 versus 0.3962 for the uncalibrated
defaults). Strengthening the primary comparator moved K1 *against* the method
under test, which is the direction the protocol requires.

`DESIGNED_MAPPING` scores 0.430, a full 0.133 below `ORACLE`. That gap is the
numeric statement of Amendment B: agreeing with the designer's intuition is not
the same as being right.

## 3. Contrasts (paired bootstrap, 95% CI, 3 dev seeds)

| Contrast | Difference | 95% CI | Excludes 0 |
|---|---|---|---|
| **K1** `LEARNED − HEURISTIC` | **−0.0189** | [−0.0235, −0.0104] | **yes — against the method** |
| **K2** `LEARNED − RANDOM_MATCHED` | +0.1108 | [+0.0927, +0.1264] | yes |
| **K4** `LEARNED − ALL_FAST` (best single depth) | +0.1556 | [+0.1498, +0.1589] | yes |
| **K7** `LEARNED − CAPACITY_MATCHED` | +0.1332 | [+0.1030, +0.1588] | yes |
| `PRIVILEGED_TASKID − HEURISTIC` | +0.0001 | [−0.0000, +0.0002] | no |
| `ORACLE − LEARNED` (headroom) | +0.1772 | [+0.1690, +0.1899] | yes |

Compute break-even `λ_compute`: **n/a** — `LEARNED` does not lead `HEURISTIC` at
the configured `λ_compute = 0.05`, so there is no advantage for extra compute to
explain. Router decision compute is 3.91% of total for `LEARNED` versus 0.19%
for `HEURISTIC`.

With three seeds these intervals are wide and none of this is confirmatory.

## 4. What the development run indicates

**Learned routing does not beat the fixed heuristic here — K1 fires.** On
development seeds, the seeds it was trained on and which therefore favour it,
the learned policy is 0.0189 *below* the calibrated heuristic with a 95% CI
excluding zero. This is consistent with Yoon (2026, arXiv:2606.30067) and is the
outcome the preregistration names as expected-plausible.

Against the *uncalibrated* heuristic the same policy was −0.0105 with a CI
spanning zero. Calibrating the comparator to its grid argmax turned an
ambiguous contrast into a clear one, against the proposed method. Both numbers
are reported; only the calibrated one counts, because a comparator that has not
been given the same search budget is not a fair test.

**Allocation content nonetheless matters (K2, K4, K7).** `RANDOM_MATCHED` spends
almost exactly the same budget on almost exactly the same action mix
(`LEARNED` 0.574 episodic / 0.426 fast; `RANDOM_MATCHED` 0.580 / 0.420) and
scores 0.111 lower. Since the two differ only in *which item receives which
action*, that gap is attributable to routing rather than to budget. `LEARNED`
also beats the best single-depth control by 0.156 and the capacity-matched exact
store by 0.133 while using 6.4× more writes than the latter — so the advantage
over depth-agnostic policies is not a capacity artefact.

**Privileged context identity is worth nothing here.** `PRIVILEGED_TASKID` is
indistinguishable from `HEURISTIC` (+0.0001, CI spanning zero). Regime
boundaries carry no signal beyond what value revision already provides, so K9 is
not in play — any future learned-routing advantage could not be explained away
as privileged task ID.

**The learned policy found a two-depth solution and stopped.** It uses only
`EPISODIC` (0.574) and `FAST` (0.426), never `IGNORE` or `SLOW`. Consequently it
beats the heuristic decisively on `LOCAL` (0.924 vs 0.804) but is much
worse on `STABLE` (0.642 vs 0.755) and slightly worse on `ONE_OFF` (0.876 vs
0.900), because nothing is ever consolidated durably. `ORACLE` reaches
`STABLE` 0.847 by spending its whole write budget on
durable consolidation. The learned policy's headroom is therefore concentrated
almost entirely in one class and one action.

That is a mechanism-level hypothesis about *why* K1 fired, and it is a
development observation, not a result. It suggests, but does not establish, that
the durable-write action is hard to discover because its payoff arrives hundreds
of steps later and its cost is immediate.

## 5. Preserved failures found by this run

*   The first `CAPACITY_MATCHED_EPISODIC` construction beat `ORACLE` outright
    (0.813 vs 0.559), which exposed that the episodic cost model omitted key
    storage and charged retrieval `O(|E|)` instead of `O(|E| · key_dim)`. Under
    that model exact storage had neither a space nor a search penalty and the
    capacity confound dominated everything. Fixed; see
    `docs/BENCHMARK.md` §6.
*   `HeuristicRouter.revision_tolerance` was a **dead parameter**: declared and
    swept, but never read, because the router used the pre-thresholded
    `value_revised` feature instead. The calibration grid was effectively 9
    points rather than 27, under-tuning the primary comparator. Caught because
    every value of the parameter scored identically. Fixed, and
    `tests/test_routers.py` now asserts every declared threshold is live.
*   The REINFORCE trainer did not train: development objective flat across 60
    epochs (0.113 → 0.079), policy collapsed to durable writes until the budget
    was exhausted (1,098 forced-ignore downgrades). Measured cause: per-decision
    resource cost is ~4 orders of magnitude below attributed utility
    (σ ratio 2.6 × 10⁻⁴), and interference and budget exhaustion are not
    attributable to individual decisions at all. Replaced with evolution
    strategies on the preregistered objective (0.317 → 0.386 over 60
    generations). The failed implementation is retained in `train.py`.

## 6. What EXP-000 does *not* establish

*   Nothing about H1. Same-seed training and evaluation; three seeds.
*   Nothing about language models.
*   Nothing about whether a better-trained policy would beat the heuristic. The
    ES budget was small and the search never found the durable-write action.

## 7. Next

EXP-001 confirmatory execution is gated on issue #1: freeze the heuristic grid
and ES hyperparameters on development seeds, then run all arms on the five
held-out confirmatory seeds.
