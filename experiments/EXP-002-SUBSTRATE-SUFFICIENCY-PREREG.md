# EXP-002 substrate-sufficiency preregistration

**Status: PREREGISTERED, NOT RUN. No EXP-002 code exists. LM execution is not
authorized.** Written 2026-09-04, before implementation.

This is the gating artefact. Its purpose is to establish that each declared
action is a *viable destination* before any compute is spent comparing routers.
The sibling track's Phase-B v1 is the cautionary case: substantial compute went
into a routing comparison whose durable destination could not absorb a segment,
so the router accepted zero promotions and the comparison was uninformative
about routing.

Governing principle: **a router that never selects an action is only
interesting if that action was actually available.**

---

## 1. Frozen execution sequence

No stage may begin until every prior stage has passed. This ordering is the
substance of the amendment, not bureaucracy.

| # | stage | may not start until |
|---|---|---|
| 1 | representation sufficiency (§4) | — |
| 2 | `FAST` sufficiency (§5) | 1 passes |
| 3 | `SLOW` sufficiency (§6) | 2 passes |
| 4 | resource / compute accounting sanity (§8) | 3 passes |
| 5 | routing benchmark admissibility (§9) | 4 passes |
| 6 | router and comparator design (§10) | 5 passes |
| 7 | protocol freeze | 6 passes |
| 8 | LM comparative runs | 7 passes **and** explicit authorization |

Stages 1–5 are **arm-agnostic**: they involve no router, no comparator and no
routing policy of any kind. Nothing in them can favour the proposed method,
because the proposed method is not present.

`EPISODIC` sufficiency (§3) and the `IGNORE` justification (§7) are properties
of the store and of the item distribution respectively; they are checked as part
of stages 1 and 5 rather than occupying stages of their own.

## 2. Seeds

`EXP002_DEV_SEEDS = (64453528, 64634207, 64290453, 63740520, 62771248)`

Derived in `config.py` by SHA-256 from the literal label
`EXP-002-substrate-sufficiency-development-v1`, in `[60000000, 69999999]`.
`tests/test_config_seeds.py` re-derives them and asserts pairwise disjointness
from every other seed set in the project.

**These seeds are permanently excluded from any later EXP-002 comparative or
confirmatory run.** Once a sufficiency grid has been inspected on a seed, that
seed can no longer support an unbiased comparison. Confirmatory seeds are not
chosen yet and must not be chosen until stage 7.

## 3. `EPISODIC` sufficiency

**Question.** Can the store hold and exactly return `N` bindings within its
declared capacity?

This is a keying and retrieval question, not a capacity question. The failure
mode to exclude is embedding collision — two distinct items resolving to the
same entry, or retrieval returning a near neighbour instead of the exact match.

**Pass criterion.** With `N` = the benchmark's required concurrent binding count
and capacity ≥ `N`: exact-match recall = **1.00** on all `N`, on every
development seed. Anything below 1.00 is a retrieval defect, since exactness is
the whole reason this substrate exists.

**If it fails:** the keying scheme is wrong. Fix the keying, not the capacity.

## 4. Stage 1 — representation sufficiency

**Question.** Which substrate parameterizations can hold `N` distinct key→value
bindings such that writing one does not collapse the others, within the arm's
own write budget?

This replaces Amendment M's "use LoRA" with a functional requirement
(Amendment N-1). Candidates to be compared, none pre-selected:

| candidate | note |
|---|---|
| additive low-rank deltas on attention projections | leading candidate; the sibling track's correction |
| key-conditioned adapters | conditioning is explicit rather than emergent |
| hypernetwork-generated deltas | higher capacity, higher compute |
| external parametric associative memory keyed on the item | closest to SDW-1's delta-rule matrix |
| global prompt / prefix state | **control**, expected to fail — retained, not deleted |

**Procedure.** Per candidate, per development seed: present the `N` bindings
once in stream order under a fixed online budget; no replay, no consolidation,
no extra epochs, no held-out labels in optimization; score the `N` distinct
bindings after the final exposure. Record exact parameter count, write units,
tokens, loss trajectory, predictions, model and tokenizer revision, code SHA,
device and the §7 numerics check.

**Pass criterion.** Mean distinct-binding accuracy across development seeds
≥ **0.833**, and every individual seed ≥ **0.667**. These are the sibling
track's thresholds (5/6 and 4/6), adopted deliberately so the two tracks'
representation gates are comparable.

**Selection rule.** Smallest sufficient capacity that passes; ties broken by
worst-seed accuracy, then by the smaller learning rate. **Selection may not use
forgetting, routing rate, or any multi-arm ordering** — none of which exist at
this stage.

**If every candidate fails:** EXP-002 does not proceed. The honest report is
that no available parameterization supports the benchmark's binding cardinality
under the budget, which is a finding about the substrate space, not about
routing.

## 5. Stage 2 — `FAST` sufficiency (two-sided)

`FAST` must satisfy **both** halves. A substrate that acquires but never forgets
is not a distinct depth, and the action space collapses to three.

**5a. Acquisition.** Acquires a segment's bindings within the online budget:
mean accuracy ≥ **0.833**, worst seed ≥ **0.667**, matching §4.

**5b. Transience.** Retention decays on the intended timescale. Let `H` be the
benchmark's regime length. Required: retention at `H` steps after the last write
≤ **0.40** of peak, **and** retention at `H/4` ≥ **0.70** of peak.

The two-sided band is the point. The upper bound makes the substrate genuinely
transient; the lower bound stops it decaying so fast that it is indistinguishable
from `IGNORE` within a regime.

**If 5a fails:** the online budget or the parameterization is inadequate — go
back to stage 1.
**If 5b fails:** `FAST` and `SLOW` are not distinct depths. Either introduce an
explicit decay mechanism, or reduce the action space and say so. Do **not**
proceed to routing with two actions that are secretly one.

## 6. Stage 3 — `SLOW` sufficiency (two-sided)

This is the gate the sibling track's Phase-B v1 failed, and the reason this
document exists.

**6a. Absorption.** Starting from the paired initialized slow substrate,
consolidate using **only already-observed training evidence** at a **fixed write
budget** — the same number of parameter-element writes in every cell, so that
batch size and learning rate vary while write cost does not. Score current-segment
accuracy after consolidation on held-out bindings that never touched
optimization.

Pass: mean ≥ **0.833**, worst seed ≥ **0.667**.

**6b. Durability.** Retention at the benchmark's full lifetime horizon ≥
**0.80** of the post-consolidation score, with no further writes to that
substrate.

**Grid.** Consolidation batch size and slow learning rate, over a predeclared
grid, with the write budget held fixed across all cells. Reporting the whole
grid is mandatory; selecting a favourable cell after any routing score is
visible is prohibited.

**If 6a fails:** this is the sibling track's failure mode reproduced. `SLOW` is
not a viable destination, the four-action space is unjustified, and the correct
response is to fix consolidation or drop the action — **not** to run a routing
comparison and report that the router declined to use it.
**If 6b fails:** `SLOW` is not durable and is not distinct from `FAST`; see §5b.

## 7. `IGNORE` justification, and accelerator equivalence

### 7a. `IGNORE` — negative expected write value

`IGNORE` is **not** an absorption test (Amendment N-4). It is justified iff the
item distribution contains items whose expected write value is negative.

**Criterion.** There exists an identifiable item class for which writing to
*every* substrate strictly lowers the objective relative to not writing, once
resource cost and interference are counted, on every development seed. Report
the margin per substrate.

**If it fails:** `IGNORE` is decorative — the resource terms are too weak to
make abstention ever correct — and either the cost model or the item
distribution must change before routing is tested. This mirrors EXP-001, where
`ALL_IGNORE` scored −0.0827 and the noise class was never queried, making
abstention genuinely optimal for one class.

### 7b. Accelerator numerical equivalence

Reference: **CPU, float32**. Every accelerator/dtype configuration must pass
before any scored run. Refuse, do not warn.

Principle: **numerical drift is tolerable, decision drift is not.**

| check | tolerance |
|---|---|
| max abs logit deviation vs reference | ≤ `1e-3` (float32) / ≤ `5e-2` (bf16, fp16) |
| mean abs logit deviation | ≤ `1e-4` (float32) / ≤ `5e-3` (bf16, fp16) |
| **top-1 argmax agreement** | **100%** over ≥ 512 probe positions, all dtypes |
| **scored-metric agreement** on the sufficiency probe set | **exact**, all dtypes |
| adapter-gradient relative L2 deviation | ≤ `1e-3` (float32) / ≤ `1e-2` (reduced precision) |

The logit envelopes are loose because reduced precision legitimately drifts; the
two decision rows are exact because a backend that changes an argmax changes the
result. The sibling track's Mac executor found MPS auto-selected and returning
silently wrong numbers — a fully-formed, fully invalid run. Device, dtype,
library versions and the measured deviations are recorded in every manifest.

## 8. Stage 4 — resource and compute accounting sanity

Arm-agnostic. Establishes that the ledger measures what it claims before it is
used to adjudicate anything.

1. **Entitlement, not consumption** (Amendment N-2). Every arm receives an
   identical token entitlement, write ceiling and storage ceiling. Actual
   consumption is measured and reported. Under-consumption is permitted and is
   never padded with dummy work; **over-consumption is invalidating.**
2. **Directional confound rule.** A consumption spread is a confound only in the
   flattering direction. If the proposed arm consumes more than a baseline it
   beats, a compute-matched baseline or a frontier is required. If it consumes
   less and still wins, the spread is reported as evidence in its favour.
3. **Token envelope enforced in the validator**, not merely documented — the
   sibling track's 2× spread (10,080 / ~20,200 / 13,447) existed precisely
   because their Amendment A named the envelope but their validator did not
   check it.
4. **Router compute split** into amortized (already computed for the primary
   task) and marginal (extra forwards, extra parameters), reported separately
   and in a total-algorithmic-compute view.
5. **Ledger self-consistency:** independently recomputed totals must match the
   ledger exactly on a fixed synthetic trace.

## 9. Stage 5 — routing benchmark admissibility

Still arm-agnostic. The EXP-001 criteria, ported:

*   an oracle over class-conditional mappings clears a floor;
*   no arm saturates a ceiling;
*   the oracle beats both the best single-depth control and the fixed heuristic
    by a margin;
*   **the empirically optimal mapping is a bijection** — every one of the four
    actions uniquely best for exactly one class. Given §§3–7a this is now a
    genuine test rather than a formality: an action that failed its sufficiency
    gate cannot be uniquely optimal for anything;
*   that mapping is stable in a local neighbourhood.

## 10. Stage 6 — router and comparator design

*   **Prefer the cheapest sufficient router** (Amendment N-3). Try cheap stream
    statistics first, then hidden states the backbone already computed, and only
    then a dedicated forward pass. Report which tier was used and its measured
    share of total algorithmic compute.
*   **Escalation is graded:** below 10% of total algorithmic compute, a
    break-even analysis suffices; at or above 10%, a compute-matched comparator
    or a preregistered performance–compute frontier is mandatory.
*   **The comparator receives a matched search budget**, rollout for rollout, as
    in EXP-001 — where correcting a ~190× asymmetry moved the comparator from
    0.4046 to 0.4390 and cut the headline gap from +0.116 to +0.082.
*   The attribution control is the EXP-001 design: an **identically specified**
    policy trained where the prefix→future-utility relationship was destroyed.
    A marginal-frequency randomiser is retained as a secondary diagnostic only —
    EXP-001's L5a established it cannot carry a causal claim.

## 11. What this preregistration does not decide

No model, no benchmark content, no hypothesis, no threshold on any comparative
contrast, no confirmatory seed set, no statistic for the routing comparison.
Those belong to the EXP-002 protocol, which is written at stage 7 and requires
its own review.

## 12. Reporting

Every stage reports its full grid regardless of outcome, including failures.
A stage that fails stops the sequence and is published as a finding about the
substrate space — which, per the sibling track's Phase-B v1, is a real and
informative result rather than a setback.
