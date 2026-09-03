# EXP-000 — development calibration result

**Classification: `DEV_CALIBRATION`. This is not evidence.**
**Every manifest from this run is additionally stamped invalid: leakage test L5 fails.**

The learned policy was trained *and* selected on development seeds
`{11, 12, 13}`. Confirmatory seeds have not been executed. EXP-000 exists to
show the harness is mechanically valid and to expose design defects before the
protocol is frozen — not to test H1.

Date: 2026-09-02. Config: `src/plasticity_routing/config.py`.
Manifests: `results/run_*.json` (36). Validator: **REJECTED** (every arm carries
`leakage_audit_failed:L5`).

---

## 1. Headline: L5 fails, so nothing here is claimable

`scripts/audit_l5.py` retrains the whole procedure in a world where each query's
target key is redrawn uniformly from the keys already written at that moment.
The write stream, query count and query timing are byte-identical, so every
budget is unchanged; only the link between an item's observable prefix and
whether it will be needed later is destroyed.

| | advantage over `RANDOM_MATCHED` | 95% CI |
|---|---|---|
| real world | +0.2604 | [+0.2273, +0.3055] |
| time-shuffled world | +0.1357 | [+0.1137, +0.1616] |
| **ratio** | **0.521** | threshold **0.25** |
| **utility-attributable advantage `A_util`** | **+0.1247** | |

**L5 FAILS against its preregistered threshold.** Slightly more than half of the
learned router's advantage over budget-matched random routing is reproduced in a
world where future utility is pure noise.

The threshold was fixed before the script was first run and **has not been
relaxed**. It could be argued that `A_util = +0.1247` is the quantity that
matters and that it is clearly positive — and Amendment H did preregister
`A_util` before this run — but the ratio criterion is what L5 was declared on,
and changing a criterion after seeing the number it produced is exactly the
post-result rescue this protocol exists to prevent. L5 is recorded as failed.

What the failure means, precisely: **`RANDOM_MATCHED` is a weak control.** It
matches the learned router's *marginal* action distribution but not its
conditional structure, so any policy with conditional resource sense beats it
whether or not it has learned anything about which items deserve which depth.
The K2 contrast is therefore not a valid test of utility-driven routing on its
own, and EXP-000's earlier reading of it was wrong.

## 2. Validity gates, in protocol order

| Gate | Outcome |
|---|---|
| L1 feature whitelist purity | PASS |
| L2 hidden-label permutation invariance | PASS |
| L3 first-encounter class independence | PASS |
| L3-canary — detector catches a deliberate oracle leak | PASS |
| L4 causal ordering / future blindness | PASS |
| L5 time-shuffled control | **FAIL** (ratio 0.521 > 0.25) |
| L6 privilege declaration | PASS |
| L7 audit probes have no side effects | PASS |
| write ceiling respected by every arm | PASS |
| manifest validation | **REJECTED** (L5 invalidation stamp) |
| K5 ceiling / K6 floor / K8 bijection | not fired |
| C6 local stability of the optimum | 15/16 neighbourhood points |

The harness refused to print comparative metrics until `--proceed-on-audit-failure`
was passed explicitly, and then stamped every manifest invalid. The numbers below
are labelled diagnostics.

## 3. Arms (diagnostics only)

Mean over 3 development seeds. `LEARNED` is the policy selected by development
objective over `POLICY_SEEDS = (0, 1, 2)`; `HEURISTIC_EXT` is the best fixed rule
found under a **search budget matched to the ES budget** (15,300 development
rollouts each).

| arm | legal | objective | utility | forgetting | writes | IGNORE | EPIS | FAST | SLOW |
|---|---|---|---|---|---|---|---|---|---|
| `ORACLE` | **no** | 0.563 ± 0.016 | 0.905 | 0.246 | 2,999,984 | 0.270 | 0.276 | 0.235 | 0.219 |
| `LEARNED` | yes | 0.521 ± 0.012 | 0.850 | 0.260 | 2,236,763 | 0.179 | 0.227 | 0.516 | 0.078 |
| `HEURISTIC_EXT` | yes | 0.439 ± 0.025 | 0.810 | 0.309 | 2,804,000 | 0.232 | 0.017 | 0.648 | 0.102 |
| `DESIGNED_MAPPING` | **no** | 0.430 ± 0.036 | 0.822 | 0.295 | 2,999,952 | 0.292 | 0.231 | 0.264 | 0.213 |
| `PRIVILEGED_TASKID` | **no** | 0.405 ± 0.024 | 0.798 | 0.385 | 1,735,205 | 0.224 | 0.556 | 0.089 | 0.131 |
| `HEURISTIC` (3-param) | yes | 0.405 ± 0.024 | 0.798 | 0.385 | 1,736,741 | 0.224 | 0.556 | 0.088 | 0.132 |
| `RANDOM_MATCHED` | yes | 0.261 ± 0.044 | 0.701 | 0.466 | 2,154,016 | 0.183 | 0.231 | 0.517 | 0.069 |
| `CAPACITY_MATCHED_EPISODIC` | yes | 0.252 ± 0.035 | 0.712 | 0.637 | 193,387 | 0.000 | 1.000 | 0.000 | 0.000 |
| `ALL_FAST` | yes | 0.230 ± 0.024 | 0.679 | 0.556 | 2,652,160 | 0.000 | 0.000 | 1.000 | 0.000 |
| `ALL_SLOW` | yes | 0.024 ± 0.013 | 0.354 | 0.296 | 2,998,272 | 0.717 | 0.000 | 0.000 | 0.283 |
| `ALL_EPISODIC` | yes | 0.020 ± 0.034 | 0.484 | 0.644 | 193,387 | 0.000 | 1.000 | 0.000 | 0.000 |
| `ALL_IGNORE` | yes | −0.082 ± 0.001 | 0.000 | 0.000 | 0 | 1.000 | 0.000 | 0.000 | 0.000 |

## 4. Contrasts (paired bootstrap, 95% CI, 3 dev seeds)

| Contrast | Difference | 95% CI | Excludes 0 |
|---|---|---|---|
| **K1** `LEARNED − HEURISTIC_EXT` (matched budget) | +0.0819 | [+0.0674, +0.1089] | yes |
| `LEARNED − HEURISTIC` (3-param grid) | +0.1163 | [+0.1022, +0.1409] | yes |
| **K2** `LEARNED − RANDOM_MATCHED` | +0.2604 | [+0.2273, +0.3055] | yes |
| **K2 corrected** `A_util` | **+0.1247** | — | — |
| **K4** `LEARNED − ALL_FAST` | +0.2909 | [+0.2749, +0.3096] | yes |
| **K7** `LEARNED − CAPACITY_MATCHED` | +0.2685 | [+0.2322, +0.3102] | yes |
| `PRIVILEGED_TASKID − HEURISTIC` | +0.0001 | [−0.0000, +0.0002] | no |
| `ORACLE − LEARNED` (headroom) | +0.0420 | [+0.0386, +0.0476] | yes |

**K1 does not fire on development seeds under the fair comparator.** This
reverses the earlier EXP-000 report, and the reason is not subtle: that report
used a single policy initialisation, and it was the worst of three.

**This is not evidence for H1.** These are the seeds the policy was trained and
selected on. The confirmatory seeds exist precisely because a development-seed
lead is what an overfit policy also looks like.

## 5. Why the earlier K1 result reversed

| policy seed | dev objective | SLOW share |
|---|---|---|
| 0 (used by the first EXP-000) | 0.3883 | 0.000 |
| 1 | 0.4702 | 0.038 |
| 2 (**selected**) | 0.5209 | 0.078 |

Spread across three initialisations: **0.133** — roughly ten times the entire
gain from raising the ES budget from 60 to 200 generations (+0.015). Reporting a
single seed reported an accident of initialisation. `POLICY_SEEDS = (0, 1, 2)`
with best-on-development selection is now frozen, and the comparator receives a
matched search budget so the selection is not a free advantage.

## 6. Durable-write discoverability

`scripts/diagnose_slow.py`, at the frozen budget:

| condition | objective | SLOW share |
|---|---|---|
| best random init (seed 2) | 0.5209 | 0.078 |
| ES from a SLOW-primed initialisation | 0.4524 | 0.175 |
| ES from a behaviour clone of the heuristic (clone accuracy 0.957) | 0.4642 | 0.162 |
| heuristic reference | 0.4046 | 0.132 |

**Verdict: `slow_reachable_from_random_init`.** SLOW is retained when the search
starts inside it, and neither primed nor cloned initialisation beats the best
random one. So the frozen budget is *not* the limitation, and the earlier
"the learned policy never uses SLOW" was a single-seed artefact rather than a
fact about the problem.

## 7. Mechanism

`LEARNED` is now within 0.042 of the ORACLE ceiling and uses all four actions. Its
per-class profile differs sharply from the comparator's:

| arm | ONE_OFF | LOCAL | STABLE |
|---|---|---|---|
| `LEARNED` | 0.588 | **0.950** | 0.819 |
| `HEURISTIC_EXT` | 0.803 | 0.785 | 0.843 |
| `ORACLE` | 0.809 | 0.980 | 0.847 |

It buys a near-oracle `LOCAL` score by spending episodic capacity on the
recurrent class, and pays for it on `ONE_OFF`. Whether that trade generalises off
the training seeds is exactly what a confirmatory run would test.

## 8. Preserved failures found by this round

* **L5 fails** (§1), which invalidated the K2 interpretation this document
  previously carried.
* **An ES initialisation override was a silent no-op.** `init_bias` and
  `init_params` were applied *after* the parameter vector ES optimises had been
  snapshotted, so the first discoverability diagnostic compared three identical
  runs and returned "SLOW genuinely unhelpful". Void; fixed; regression test
  added.
* **The comparator had been searched ~190× less hard** than the learned router
  (81 rollouts versus 15,300). Correcting that lifted the fixed rule from 0.405
  to 0.439 and cut the K1 gap from +0.116 to +0.082.

## 9. What EXP-000 does not establish

Nothing about H1: same-seed training and selection, three seeds, and a failing
leakage control. Nothing about language models. Whether the `LOCAL`-heavy
strategy survives on unseen seeds is unknown.

## 10. Superseded by the confirmatory run

EXP-001 confirmatory has since executed under protocol v1.2 and supports H1
(K1 = +0.0907 [+0.0798, +0.1016]). This document is retained as the development
record; its numbers are development-seed diagnostics and remain non-evidential.
See [`EXP-001-RESULT.md`](EXP-001-RESULT.md).

## 11. Next (at the time of writing)

Protocol v1.0 **cannot be frozen** while L5 fails; `scripts/freeze_protocol.py`
lists it as a blocker and `scripts/validate_runs.py` refuses to certify a
confirmatory run without a frozen lock. The required next step is a stronger
control than `RANDOM_MATCHED` — one matched on conditional structure rather than
only on the marginal action distribution — or an explicit, justified revision of
L5's criterion made and logged **before** any confirmatory seed is touched.

The five confirmatory seeds have not been run.
