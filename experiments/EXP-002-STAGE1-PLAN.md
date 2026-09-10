# EXP-002 Stage 1 — implementation plan (representation sufficiency)

**Status: PLAN. No Stage-1 experimental code. No LM execution. No model loaded.**
Written 2026-09-10 under Amendment O.

Stage 1 answers one arm-agnostic question: **which substrate parameterizations
can hold `N` distinct key→value bindings such that writing one does not collapse
the others, within the arm's own write budget?** No router, no comparator and no
routing policy exists while it runs, so nothing in it can favour the proposed
method.

---

## 1. What this plan must produce, in order

| # | artifact | gate |
|---|---|---|
| 1.1 | binding generator + its tests | deterministic, seed-reproducible, no LM |
| 1.2 | substrate adapter interface + per-family implementations | no LM in the interface; families are pluggable |
| 1.3 | write/token/compute ledger for Stage 1 | entitlement-based (Amendment N-2) |
| 1.4 | accelerator equivalence harness | §7b tolerances; refuses |
| 1.5 | **freeze artifact + validator** | refuses while any field unset |
| 1.6 | Stage-1 runner | reads the freeze; cannot run unfrozen |
| 1.7 | mechanical tests for all refusal paths | — |

**1.1–1.5 and 1.7 contain no model and no training loop.** Only 1.6 touches an
LM, and it is the last thing built and separately authorized.

## 2. Binding generator (1.1)

The task is arbitrary nonce key→value recall. Design constraints follow directly
from Amendment O-2: because there is nothing to generalise from, the *binding*
is always seen in training and only the **presentation** is held out.

```
Binding      := (key_id, key_surface, value_token)
Presentation := (binding, template_id, position, distractors)
```

Per binding the generator emits `P` presentations, partitioned by the freeze
into `P_train` and `P_eval` with **disjoint template ids**. The learner
optimizes on `P_train`; scoring uses `P_eval` for the same bindings.

Requirements:

*   nonce keys and values — no lexical overlap with plausible pretraining
    content, so a score reflects the write and not the prior;
*   a **prior-only control**: score `P_eval` with an untouched backbone before
    any write. Required to be at chance. If a family "passes" while the control
    is already above chance, the task is leaking and the result is void;
*   deterministic from seed; regenerating from the same seed is byte-identical;
*   no evaluation presentation ever reachable from the training split.

Tests: determinism, train/eval template disjointness, nonce-ness (no key or
value appears in a declared vocabulary allowlist), prior-control-at-chance.

## 3. Substrate interface (1.2)

One narrow interface so families are swappable and the gate compares like with
like:

```python
class Substrate(Protocol):
    def write(self, presentations, *, budget: WriteBudget) -> WriteReceipt: ...
    def read(self, presentation) -> Prediction: ...
    def parameter_elements(self) -> int: ...
    def step(self) -> None: ...          # decay hook; no-op unless declared
    def snapshot(self) -> bytes: ...     # for retention measurement
```

Families to implement (Amendment N-1 — capacity is the requirement, LoRA is a
candidate, not the frozen architecture):

| family | capacity knob | note |
|---|---|---|
| additive low-rank delta on attention projections | rank | leading candidate |
| key-conditioned adapter | bottleneck width | conditioning explicit |
| external parametric associative memory | key dim | closest to SDW-1's delta rule |
| global prompt/prefix | token count | **control**, expected to fail |

The prompt/prefix control is implemented and reported, not skipped. Its expected
failure is what makes the others' passes meaningful, and the sibling track's
result is evidence about one parameterization under one budget rather than a
proof of zero expressive power.

`step()` exists so a family can declare explicit decay (§5b case (i)). Families
that do not decay return a no-op, and their transience must then be established
by interference (case (ii)). The declaration lives in the freeze, not in code
comments.

## 4. Ledger (1.3)

Ported from EXP-001 with the Amendment N-2 correction:

*   every family receives the **same entitlement** — token, write-element and
    optimizer-step ceilings;
*   actual consumption is measured and reported per cell;
*   **under-consumption is never padded**; over-consumption is invalidating;
*   parameter-element writes counted from the **actual** parameter count, never
    from a nominal capacity knob.

Test: a family that under-consumes is not penalised and emits no dummy work; a
family that exceeds entitlement is refused.

## 5. Accelerator equivalence (1.4)

Implements §7b. Reference CPU/float32. Refuses rather than warns. Retained
knowingly conservative per Amendment O: it can cause a false rejection of a good
device, which costs a device, but it cannot manufacture a flattering result.

## 6. The freeze (1.5) — the terminal artifact of this plan

`scripts/freeze_exp002_stage1.py` emits `results/exp002/stage1_freeze.json` only
when **every** field in prereg §13 is set. Until then it exits non-zero and
lists each unset field.

The Stage-1 runner (1.6) takes its parameters **only** from that artifact —
there is no CLI for `N`, no CLI for the grids, no CLI for the budgets — exactly
as `run_exp001_confirmatory.py` takes its seeds only from the protocol lock.

### Values still requiring owner sign-off

Proposed below, **deliberately not frozen** — these are scientific choices, not
implementation details, and freezing them unilaterally is what Amendment O-1
exists to prevent.

| field | proposal | rationale |
|---|---|---|
| backbone | `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775` | the sibling track's pinned snapshot; makes the two tracks' representation gates directly comparable |
| `N` | 6 | matches the sibling track's PALS cardinality, so the 5/6 and 4/6 thresholds mean the same thing |
| presentations per binding `P` | 8, split 6 train / 2 eval, disjoint templates | their 48 exposures = 6 × 8, with the split added per O-2 |
| capacity grid | rank ∈ {1, 2, 4} and the per-family equivalent | their EXP-001R grid |
| learning-rate grid | {3e-4, 1e-3, 3e-3} | their EXP-001R grid |
| update budget | 48 optimizer steps | their 48 exposures, once, in stream order |
| token entitlement | measured from the 48-exposure stream, applied to every family | entitlement not consumption |
| `FAST` transience mechanism | **undecided** — the one field with no defensible default | see below |
| interference sets | `A, B, C` disjoint, `|A|=|B|=|C|=N`, order fixed | minimum that makes §6b non-trivial |
| device/dtype | CPU / float32 | the sibling track found MPS silently wrong |

**`FAST` transience has no default and should not get one from me.** Explicit
decay (case i) is closer to SDW-1 and is cleanly measurable in isolation;
interference-driven transience (case ii) is closer to how an adapter actually
behaves in an LM and needs the sequential design. The choice changes what §5b
measures, so it is an owner decision.

## 7. Ordering, and what stops the plan

Build order is 1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.7, then 1.6 only after the freeze
validates and LM execution is authorized separately.

The plan stops — and reports rather than adapts — if:

*   the prior-only control is above chance (task leaks; §2);
*   every family fails the §4 bar (finding about the substrate space, not about
    routing);
*   the accelerator equivalence harness refuses on every available device.

## 8. Out of scope

No routing policy. No comparative arms. No `SHUFFLE_TRAINED` control. No EXP-002
score of any kind. No confirmatory or replication seeds. Stages 2–8 of the
preregistered sequence are untouched by this plan.
