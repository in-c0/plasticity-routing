# Baseline and control definitions

Arms are grouped by what they are *for*. Only arms marked **claim-eligible** may
appear in a comparison that supports a conclusion.

## Primary arms

| ID | Name | Claim-eligible | What it is |
|---|---|---|---|
| A0 | `ALL_IGNORE` | yes | Writes nothing. The floor: shows what the objective's resource terms alone are worth. |
| A1 | `ALL_EPISODIC` | yes | Depth-agnostic control: everything to the exact store. Fails by eviction thrash and staleness. |
| A2 | `ALL_FAST` | yes | Depth-agnostic control: everything to the decaying substrate. Fails by interference and decay of durable facts. |
| A3 | `ALL_SLOW` | yes | Depth-agnostic control: everything to the durable substrate. Fails by permanent corruption from noise and revised values, and by write-budget exhaustion. |
| A4 | `HEURISTIC` | yes | **The primary comparator.** A fixed hand-designed ladder over recurrence, prediction error, and value revision. Promoted from courtesy baseline to primary comparator because Yoon (2026, arXiv:2606.30067) reports a simple rule matching or beating a learned RL allocation controller. |
| A5 | `RANDOM_MATCHED` | yes | Random routing whose action distribution is matched post-hoc to A7's realised histogram, so write and storage consumption are comparable. Isolates *which item goes where* from *how the budget was spent*. |
| A7 | `LEARNED` | yes | The method under test. Sees only `features.extract` output. |

The three depth-agnostic controls are required together, not as padding: each
fails for a **mechanistically different** reason, and if any one of them matches
A7 then the benchmark is not depth-stratified and the run is invalid (K4).

## Ceiling and privileged probes — never claim-eligible

| ID | Name | What it is |
|---|---|---|
| A6 | `ORACLE` | Best class-conditional mapping, found by exhaustive search over all `4^4` mappings on development seeds. Reads the hidden class. An **upper bound over the class-conditional policy family**, reported for calibration only. |
| C3 | `PRIVILEGED_TASKID` | The heuristic plus the regime id. Quantifies how much of any gap is simply access to privileged context identity. |

`ORACLE` is not "the right answer a good router should reach". A legal router
cannot reach it: on a first encounter SDW-1 makes the classes observationally
identical, so some decisions are irreducibly uncertain. A learned router
approaching ORACLE *on first encounters* is evidence of leakage, not of skill
(leakage test L3).

## Disentanglement arms

The brief requires that four things not be conflated. Each gets its own arm:

| Confound | Arm | Construction |
|---|---|---|
| routing benefit | A7 vs A4/A5 | matched budgets, matched substrate |
| **extra capacity** benefit | C1 `CAPACITY_MATCHED` | the strongest depth-agnostic control, re-run with total capacity raised to whatever A7 actually occupied |
| **extra compute** benefit | C2 `COMPUTE_MATCHED` | the strongest control, re-run with a compute allowance equal to A7's total *including* router decision compute |
| **privileged task/context ID** | C3 `PRIVILEGED_TASKID` | heuristic + regime id |

If A7's advantage over A4 disappears under C1 or C2, the advantage was capacity
or compute, not routing. That is kill criterion K7.

## Heuristic calibration

A4's thresholds are chosen from a small predeclared grid on **development seeds
only**, and frozen before confirmatory seeds are touched. The grid is declared
in [`../experiments/EXP-001-PREREG.md`](../experiments/EXP-001-PREREG.md).

A4 is calibrated to be *as strong as the grid allows*. Deliberately weakening
the primary comparator would be the most direct way to manufacture a positive
result, so the calibration procedure for A4 is identical in effort and budget to
the one used for A7's hyperparameters.

## Random-matching procedure (A5)

1. Run A7 on the seed; record its realised action histogram.
2. Construct `RandomMatchedRouter` with those probabilities.
3. Run A5 on the same seed and same world.

A5 therefore consumes approximately A7's write and storage budget while carrying
no information about *which* item deserves which depth. If A7 ≈ A5, any apparent
gain is a budget-allocation artefact rather than routing (K2).
