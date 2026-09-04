# EXP-002 — pre-result design under the State Promotion findings

**Status: DESIGN ONLY. Language-model execution is NOT authorized.**
No EXP-002 code has been written, no LM has been run, and no seed set exists.

## 1. Gate status

The EXP-002 dependency gate was: *do not run a confirmatory LM experiment until
the first mechanically valid State Promotion engineering pilot has been
reviewed.*

**That review has now occurred**, and the findings are public in the sibling
repository's issue tracker — not in its files, which is why this document cites
issues rather than paths:

*   [in-c0/state-promotion#1](https://github.com/in-c0/state-promotion/issues/1)
    — the pilot itself; `valid_for_confirmatory_interpretation: true`,
    zero invalidations, six arms.
*   [#4](https://github.com/in-c0/state-promotion/issues/4) — EXP-001R,
    representation sufficiency.
*   [#5](https://github.com/in-c0/state-promotion/issues/5) — Phase B,
    strong baselines.
*   [#6](https://github.com/in-c0/state-promotion/issues/6) — v2,
    consolidation sufficiency.

**Review satisfied → design may begin. Execution remains blocked** pending an
explicit authorization, because the constraints below change what EXP-002 should
be, and several of them are not yet implemented anywhere in this repository.

## 2. What the pilot found, and what each finding forces here

These are corrections to *our* design, adopted before any LM result exists. That
is the point of the gate: EXP-001 here must not bake in a defect that pilot
discovered.

### C-1. Match adaptation-token exposure, not merely parameter-write ceilings

The pilot measured a **2× spread** in adaptation-token exposure across arms:
10,080 (sequential, random) vs ~20,200 (replay, fixed) vs 13,447 (promotion).
Their Amendment A names the token envelope as a primary matched resource, and
their validator did not enforce it.

**Forced here.** EXP-001's ledger matches write elements, storage occupancy and
compute, but an LM version introduces token exposure as a distinct resource that
none of our current accounting captures. EXP-002's ledger must add a
token-exposure envelope, enforce it in the validator, and treat a spread beyond
a preregistered tolerance as invalidating — or report an explicit
efficiency curve instead of a point comparison.

### C-2. Decision-time routing compute must be load-bearing

The pilot quantified routing overhead and the answer was not incidental: **B5
spent 72.4% of its total algorithmic compute on routing decisions** — 1,008
routing forwards and 35,340 decision tokens against 13,447 training tokens
(3.49e13 vs 2.66e13 estimated frozen-backbone FLOPs). It spends roughly 2.6×
more compute deciding than adapting.

**Forced here.** In EXP-001 the router was a 340-parameter MLP and decision
compute was 4.9% of total — small enough that a break-even analysis sufficed.
That will not transfer. An LM router that reads the item is a forward pass per
decision, and at that ratio a compute-matched comparator is not optional.
EXP-002 must preregister **either** a compute-matched comparator **or** an
explicit performance–compute frontier, and must report the frontier rather than
selecting a favourable point after the learned arm is seen.

### C-3. The plastic substrate must be input-conditional

State Promotion's fast/slow state was a global learned soft prefix. Under it,
segment-0 acquisition sat at floor and the diagnostic showed **all six distinct
keys mapping to one candidate**: a global prefix can shift the output
distribution but cannot express arbitrary key→value bindings at the required
capacity. Their correction is additive low-rank adapter deltas on `q_proj` and
`v_proj` in every block, rank-matched across arms by rank linearity.

Their own framing is careful and worth preserving: *do not overclaim that a
fixed soft prompt has mathematically zero input-conditional expressive power* —
the frozen transformer still interacts with the query. The empirical failure was
sufficient to block their experiment; the theoretical claim was not made.

**Forced here.** This is the sharpest warning for us. EXP-002's `FAST` and
`SLOW` substrates **must not be global prompt vectors**. SDW-1's substrates were
input-conditional by construction (delta-rule associative matrices keyed on the
item), and that property is load-bearing, not incidental — a depth axis whose
substrates cannot bind keys to values collapses the whole routing problem.
Additive rank-matched LoRA deltas are the natural port.

### C-4. Consolidation must be trained under the configuration it is evaluated in

The pilot found slow consolidation being optimized with latent state disabled
and then evaluated with latent state present.

**Forced here.** EXP-002 must assert, mechanically, that every substrate is
trained under the same context/latent configuration in which it is scored. This
is the LM analogue of EXP-001's `SubstrateBank.probe` rule, where evaluator
probes were forbidden from mutating episodic LRU order; the same class of bug,
one level up.

### C-5. Preserve the pilot's provenance controls

Carry over, unchanged in spirit: the dynamic model-input audit (archive an
actual tokenized first online batch and first evaluation query per arm),
immutable Hugging Face snapshot pinning (they pin
`7ae557604adf67be50417f59c2c2f167def9a775` and resolve tokenizer assets by
hash rather than trusting `tokenizer.init_kwargs._commit_hash`), the
held-out-label firewall, and optimizer-state persistence across segment
boundaries.

**Forced here.** EXP-001's `SELECTED_POLICY_SHA256` / `MATCHED_HEURISTIC_SHA256`
pinning is the same discipline applied to checkpoints; EXP-002 extends it to the
backbone snapshot and tokenizer assets.

### C-6. Accelerator numerical-equivalence guard

The pilot's Mac executor found **MPS returning wrong numbers silently** and
auto-selected by default. This is an execution-validity failure, not a
performance issue: a silently wrong backend produces a fully-formed, fully
invalid result. Their fix was a device numerics guard plus an explicit
`--device`.

**Forced here.** EXP-002 must run a numerical-equivalence check against a
reference device before any scored run, and refuse rather than warn. EXP-001 was
pure NumPy on CPU and never faced this; that immunity does not survive the port.

### C-7. The durable substrate must be shown able to absorb content *before* routing to it is meaningful

Not in the original six, and the most consequential for us. State Promotion's
Phase-B v1 is **complete and negative** on development seeds: B5 accepted **zero
promotions**, was therefore identical to B4, and failed H1 against B2. The cause
was localized — fast acquisition usually passed the first gate, but 48-step
batch-1 slow consolidation typically drove current-segment accuracy to chance,
so rollback correctly rejected every candidate. Their v2 response is an
**arm-agnostic consolidation-sufficiency gate** on fresh development seeds,
before any new comparative score is exposed.

**Forced here.** If the `SLOW` substrate cannot absorb a segment under plausible
compute, a router that never selects it is not failing — the action is simply
unavailable, and the four-action space is not justified. EXP-001 has the
analogous machinery already: benchmark admissibility criterion **C5** requires
the empirically optimal mapping to be a *bijection*, i.e. every action uniquely
best for some class. EXP-002 must run a **per-substrate sufficiency gate before
any routing comparison**, and treat a substrate that fails it as evidence that
the action set is wrong, not that the router is.

This also reframes a result we already have. In EXP-001 the learned policy
initially never used `SLOW`; we established by diagnostic that `SLOW` was
*reachable* (a primed initialisation retained it at 17.5%, a behaviour clone at
16.2%, and the best random seed used it at 7.8%). Had we not run that check we
would have mistaken an optimizer artefact for a fact about the problem — which
is exactly the error State Promotion's v1 avoided by localizing its own failure.

## 3. Consequences for the EXP-002 design

| EXP-001 as built | EXP-002 must instead |
|---|---|
| NumPy associative matrices | rank-matched additive LoRA on `q_proj`/`v_proj`, **never** a global soft prefix (C-3) |
| write / storage / compute ledger | **plus** an enforced adaptation-token envelope (C-1) |
| router = 340-param MLP, 4.9% of compute | LM-scale router; compute-matched comparator **or** a preregistered frontier (C-2) |
| CPU NumPy, no device risk | numerical-equivalence guard, explicit device, refuse on mismatch (C-6) |
| policy/comparator pinned by SHA-256 | **plus** backbone snapshot and tokenizer assets pinned by hash (C-5) |
| C5 bijection check on the benchmark | **plus** a per-substrate sufficiency gate run before any routing comparison (C-7) |
| probe non-interference (L7) | train/evaluate configuration equivalence asserted mechanically (C-4) |

## 4. What is explicitly *not* decided here

No architecture, no model, no benchmark, no seed set, no hypothesis, no
threshold, no statistic. This document records constraints that any EXP-002
design must satisfy; the design itself, and its preregistration, come later and
require their own review.

## 5. Execution gate

EXP-002 LM execution remains **blocked**. Unblocking requires, at minimum:

1. a preregistration satisfying C-1 through C-7;
2. per-substrate sufficiency demonstrated on development seeds (C-7) before any
   comparative arm is run;
3. a frozen protocol lock in the style of v1.0–v1.2, with equivalence assertions;
4. explicit authorization to spend LM compute and a held-out seed set.

Until then this repository's empirical claim remains what EXP-001 established:
one controlled proof-of-principle on one synthetic benchmark with five seeds.
