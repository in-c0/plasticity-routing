# EXP-001 preregistration — learned write-depth routing on SDW-1

**Status:** DRAFT LOCK candidate. This document must be committed, and the
protocol frozen, before any confirmatory result is inspected.

**Classification of everything produced so far:** `DEV_CALIBRATION`. No
confirmatory result exists at the time of writing.

---

## 1. Question

Under matched write, storage, and compute budgets, can a policy learn to
allocate each experience to a substrate of differing persistence and
interference character — from delayed future task utility alone — and does it
beat a fixed heuristic that uses the same decision-time information?

## 2. Hypotheses

**H1 (primary).** On held-out confirmatory seeds, `LEARNED` achieves a higher
mean objective than `HEURISTIC`, with a paired-bootstrap 95% CI on the
difference excluding zero.

**H2 (attribution).** Any H1 advantage is attributable to routing rather than to
budget consumption, capacity, compute, or privileged context: `LEARNED` also
beats `RANDOM_MATCHED`, survives `CAPACITY_MATCHED`, retains its advantage under
compute matching, and exceeds the advantage `PRIVILEGED_TASKID` obtains over
`HEURISTIC`.

**H3 (mechanism).** `LEARNED`'s action distribution conditioned on the hidden
class moves toward the ORACLE mapping over training — measured *post hoc* as a
diagnostic only. Routing agreement is never a training signal and never a
success criterion, because the ideal action is unobservable at decision time.

### Prior

The prior on H1 is **low**. Yoon (2026, arXiv:2606.30067) reports that a simple
similarity rule matches or beats a learned RL allocation controller under fixed
capacity, in the closest published comparison. A negative result here is the
expected-plausible outcome, is a valid experiment, and will be published as the
headline finding if it occurs.

## 3. Objective

```
J = task_utility
      − λ_forget  × forgetting
      − λ_storage × storage_norm
      − λ_write   × write_norm
      − λ_compute × compute_norm
```

Frozen weights: `λ_forget = 0.50`, `λ_storage = 0.10`, `λ_write = 0.10`,
`λ_compute = 0.05`. Results are additionally reported at `λ_forget = 0` as a
sensitivity check, because forgetting is partially double-counted in
`task_utility` (stated in
[`../docs/RESOURCE-NORMALIZATION.md`](../docs/RESOURCE-NORMALIZATION.md)).

## 4. Action space

`IGNORE`, `EPISODIC`, `FAST`, `SLOW`. Justification for this being the smallest
scientifically meaningful subset is in
[`../docs/LITERATURE.md`](../docs/LITERATURE.md) §5, and it is *tested*: the
benchmark is admissible only if the empirically optimal class→action mapping is
a bijection (criterion C5).

Excluded from EXP-001: `UPDATE_LATENT` (sibling track's variable),
`UPDATE_EXISTING_MODULE_i` and `SPAWN_NEW_MODULE` (add capacity — the primary
confound — and sit inside the modular-CL literature).

## 5. Benchmark

SDW-1, frozen configuration in `src/plasticity_routing/config.py`, specified in
[`../docs/BENCHMARK.md`](../docs/BENCHMARK.md). The optimal class→action mapping
was derived by exhaustive search over all 256 mappings on development seeds, not
asserted by the designer; it is
`NOISE→IGNORE, ONE_OFF→FAST, LOCAL→EPISODIC, STABLE→SLOW`, which **contradicts**
the design hypothesis and is retained as a preserved negative result.

## 6. Arms

Claim-eligible: `ALL_IGNORE`, `ALL_EPISODIC`, `ALL_FAST`, `ALL_SLOW`,
`HEURISTIC`, `RANDOM_MATCHED`, `LEARNED`.
Ceiling / privileged, never claim-eligible: `ORACLE`, `DESIGNED_MAPPING`,
`PRIVILEGED_TASKID`.
Disentanglement: `CAPACITY_MATCHED_EPISODIC`, plus the compute break-even
analysis.

Full definitions: [`../docs/BASELINES.md`](../docs/BASELINES.md).

## 7. Seeds

*   **Development seeds `{11, 12, 13}`** — benchmark calibration, heuristic
    threshold selection, ES hyperparameters, policy training.
*   **Confirmatory seeds `{20260902, 20260903, 20260904, 20260905, 20260906}`** —
    disjoint from development seeds, frozen in `config.py`, never inspected
    before protocol lock.

The learned policy is trained on development seeds and evaluated on confirmatory
seeds. A run that trains and evaluates on the same seeds is `DEV_CALIBRATION`
and cannot support a claim. All arms are paired on each seed.

## 8. Calibration, and what may not be tuned

Already performed, on development seeds, **before** the learned router was ever
run against the benchmark:

*   world/substrate parameters, selected by discriminability criteria C1–C6 in
    `scripts/calibrate_world.py`, which executes only the depth-agnostic
    controls, the heuristic, and ORACLE mappings;
*   the ORACLE mapping, by exhaustive search.

To be frozen before confirmatory seeds:

*   heuristic thresholds — **done**: the predeclared grid
    `seen_threshold ∈ {1,2,3} × revision_tolerance ∈ {0.7,0.8,0.9} ×
    error_floor ∈ {0.15,0.25,0.35}` was searched by
    `scripts/calibrate_heuristic.py` and the argmax
    `(1, 0.8, 0.25)` frozen as the default, selected to **maximise** heuristic
    performance on development seeds (0.4046 vs 0.3962 uncalibrated);
*   ES hyperparameters — **done**: `generations=100` (plateau at generation 86 by
    the rule fixed before running `scripts/es_budget_study.py`; 60 → 200 is worth
    only +0.015), `population=24`, `sigma=0.12`, `lr=0.06`, `hidden=16`,
    `seeds_per_generation=2`, `weight_decay=0.002`;
*   the policy-seed rule — **done**: `POLICY_SEEDS = (0, 1, 2)`, carrying forward
    the policy with the best **development** objective
    (`scripts/select_policy.py`). Policy-seed spread (0.133) is roughly ten times
    the entire ES-budget effect, so a single seed would report an accident of
    initialisation. Selection by *confirmatory* performance is prohibited;
*   the matched comparator — **done**: the learned router consumes
    `3 × 100 × 51 = 15,300` development rollouts, so
    `scripts/search_heuristic_matched.py` spends the same 15,300 on
    `ExtendedHeuristicRouter`. The primary comparator for H1 is the best legal
    non-learned router found under that matched budget (`HEURISTIC_EXT`, dev
    0.4390, versus 0.4046 for the three-parameter grid).

**Prohibited:** tuning the world, the cost table, the objective weights, the
substrate parameters, or the heuristic *after* seeing any confirmatory result;
weakening the heuristic; adding dummy writes; selecting among policy seeds by
confirmatory performance.

The heuristic's calibration budget is identical in effort to the learned
router's. Deliberately weakening the primary comparator is the most direct route
to a manufactured positive result and is the thing this clause exists to
prevent.

## 8b. Blocking status

**Protocol v1.0 is NOT frozen.** Leakage test L5 fails at the frozen
configuration: the learned router's advantage over `RANDOM_MATCHED` is +0.2604 in
the real world and +0.1357 in the time-shuffled world, a ratio of 0.521 against
the preregistered threshold of 0.25. The utility-attributable advantage
`A_util = +0.1247` is clearly positive, but the criterion L5 was declared on is
the ratio, and it has not been relaxed.

Confirmatory execution is blocked until either

1. a control stronger than `RANDOM_MATCHED` is introduced — matched on
   conditional structure, not only on the marginal action distribution — and L5
   is re-run against it; or
2. L5's criterion is explicitly revised, justified, and logged as a pre-result
   amendment **before** any confirmatory seed is executed.

Whichever is chosen must be recorded before the confirmatory run, not after.

## 9. Validity gates, in order

Inspected in this order; comparative metrics last.

1. leakage audit L1–L7 passes (`make l5 && make leakage`) — **currently failing at L5**;
2. `scripts/validate_runs.py` certifies the manifest set;
3. write ceiling respected by every arm; forced-ignore counts reported;
4. benchmark admissibility C1–C6 still holds on confirmatory seeds;
5. no kill criterion K3–K6, K8 fires;
6. only then, comparative metrics and the preregistered contrasts.

## 10. Reporting

Regardless of outcome: mean and per-seed objective, task utility, forgetting,
per-class utility, action distributions, the hidden-class × action confusion
diagnostic, all resource totals, router decision compute as a share of total
compute, the compute break-even `λ_compute`, paired-bootstrap 95% CIs for every
preregistered contrast, and the full manifest set.

Negative results are published. All development-stage negative findings already
recorded in [`../docs/BENCHMARK.md`](../docs/BENCHMARK.md) §6 remain in the
repository permanently.

## 11. Amendment log

Pre-result amendments are logged in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md#amendment-log). At the time
of writing: Amendments A–E, all made before any confirmatory run.

*   **Amendment D (2026-09-02) — arrival time leaked class.** Leakage test L3
    failed on an *untrained* policy: with small recurrent pools sampled
    uniformly, every recurrent key first appeared early, so a novel key seen
    late was probably single-use. Fixed by drawing introduction times uniformly
    over the whole lifetime, force-presenting each recurrent key at its
    introduction time, and emitting classes by independent per-step Bernoulli
    draws. Found by the repository's own leakage audit, not by inspection.

*   **Amendment E (2026-09-02) — the trainer could not see most of the
    objective.** The initial REINFORCE trainer used per-decision credit
    assignment. Measured on development seeds, a decision's resource cost is
    ~4 orders of magnitude below its attributed utility and vanishes under
    advantage standardisation; interference caused in *other* keys is not
    attributable to the causing write; and budget exhaustion is a whole-
    trajectory property. The trainer's development objective was flat across 60
    epochs and the policy collapsed to writing durably until the budget ran out.
    Replaced with evolution strategies optimising the preregistered objective
    directly as a black box, which requires no attribution assumption and no
    reward shaping. The REINFORCE implementation is retained in `train.py` and
    its failure is reported rather than deleted.

## 12. Gate to EXP-002

EXP-002 (any language model) **must not run** until the first mechanically valid
`state-promotion` engineering pilot has been reviewed and any defect it exposes
has been incorporated here as a pre-result amendment. EXP-001 is synthetic and
is not blocked by that gate. See
[`../docs/RELATION-TO-CCS.md`](../docs/RELATION-TO-CCS.md).
