# EXP-003 — Allocation × Internal-Commitment Factorial

**Status: DESIGN ONLY. Not adopted, not preregistered, not executed.**
No EXP-003 code exists, no seed has been drawn, and no confirmatory run is
authorized. This document exists so the owning track can decide whether to adopt
the node. The preregistration draft is [`EXP-003-PREREG-DRAFT.md`](EXP-003-PREREG-DRAFT.md).

EXP-002 is untouched and remains the LM port / substrate-export track.

## 1. The gap this closes

EXP-001 produced admissible confirmatory evidence that **allocation content
matters** under matched write, storage and compute budgets. The CCS umbrella
recorded it as *partial-scope* support for CCS-C4, because CCS-C4 claims
something EXP-001 cannot reach:

> allocation is **separable** from COMMIT_INTERNAL — one can be varied while a
> distinct commit policy is held fixed, producing a measurable difference.

The reason is structural and visible in `substrates.py`:

```python
elif action == SLOW:
    self.slow.write(key, value, repeats=self.cfg.slow_consolidation_steps)
```

**In EXP-001, choosing `SLOW` *is* committing.** Durable state is written
directly by the allocator; there is no FAST→SLOW promotion anywhere in the
codebase. Allocation and internal commitment are the same act, so no EXP-001 arm
can hold one fixed while varying the other. This is not a weakness of the result
— it is the wrong shape for the question.

## 2. The one structural change

**Remove `SLOW` from the allocation action space, and make durability reachable
only through a separate commitment operator.**

```text
EXP-001    allocator ──▶ {IGNORE, EPISODIC, FAST, SLOW}
                                            └── durable, written directly

EXP-003    allocator ──▶ {IGNORE, EPISODIC, FAST}
                                        │
                                        │  FAST content is a *candidate*
                                        ▼
                          COMMIT_INTERNAL operator ──▶ SLOW
                          (schedule only, no evidence gate)
```

After this change the two operators are structurally distinct:

- **ALLOCATE** decides *what becomes a candidate for durability*, and where
  non-candidates go instead.
- **COMMIT_INTERNAL** decides *whether and when candidates become durable*.

This is the minimal edit that makes the factorial possible. It is also closer to
the CCS decomposition than EXP-001 was: ACCUMULATE → ALLOCATE → COMMIT_INTERNAL
with a real boundary at the last arrow.

### Commitment mechanism, specified without importing State Promotion

The owner constraint is that evidence-gated commitment may be added only if it
can be specified without importing an unvalidated State Promotion mechanism.
This design therefore uses **schedule-only** commitment:

At each commit event, the pending window of FAST-allocated `(key, value)` pairs
since the last event is promoted into `slow` via the existing
`AssociativeMatrix.write`. **No selection, no gating, no ranking.** Which items
promote is determined entirely by *when* the event fires and what is pending —
never by evidence about their utility.

That keeps COMMIT_INTERNAL as pure timing/regime. It also means EXP-003 settles
separability from commitment *timing*, not from evidence-gated commitment. See
§7.

## 3. Factorial structure

**3 × 2 fully crossed, paired seeds.**

### ALLOCATE factor `A`

| level | router | role |
|---|---|---|
| **A1** `LEARNED` | ES-trained `LearnedRouter` over the 15-feature decision-time whitelist, restricted to 3 actions | the method under test |
| **A2** `HEURISTIC_EXT` | `ExtendedHeuristicRouter` calibrated under a matched search budget | strong fixed comparator, per EXP-001 precedent |
| **A3** `ACTION_MATCHED_RANDOM` | random allocation matched to A1's **realized per-action counts** | the causal control |

A3 is the load-bearing arm. It matches A1's action *mix* — therefore its FAST
count, therefore (under an identical commitment schedule) its **realized commit
count** — while randomising *which* item receives which action. `A1 − A3` is
allocation **content** with commitment volume held fixed.

### COMMIT_INTERNAL factor `C`

| level | regime | cadence |
|---|---|---|
| **C1** `IMMEDIATE` | promote each pending FAST write at the step it occurs | 1 |
| **C2** `CADENCE_k` | promote the pending window every `k` steps | `k`, frozen from dev calibration |

Both are predeclared before any confirmatory seed. `k` is chosen on development
seeds by a criterion that mentions no allocation arm (§5).

### Reported anchors, not part of the primary 3 × 2

| anchor | purpose |
|---|---|
| **C0** `NEVER` | no promotion; `slow` stays empty. Establishes the commitment factor has a real range, and is the satisfiability check for the C main effect. |
| `ORACLE_3` | exhaustive search over the reduced `3^4 = 81` class→action mappings, per regime. Ceiling only, never claim-eligible. |
| `SUPPLY_MATCHED_RANDOM` | random allocation with FAST count forced equal **by construction** rather than matched post hoc. Secondary; see §7.3. |

## 4. Primary criterion, and the failure criterion CCS-C4 needs

Let `Δ_A(c) = A1 − A3` within commitment regime `c`, paired bootstrap over
confirmatory seeds, 95% CI.

**P1 — allocation effect at fixed commitment.**
`Δ_A(C1) > 0` and `Δ_A(C2) > 0`, each CI excluding zero, with realized commit
counts matched within tolerance (§5).

**P2 — replication across commitment regimes.**
P1 holds in **both** regimes with the **same sign**.

**CCS-C4 settlement is claimed only if P1 ∧ P2.**

### The predeclared failure criterion

> **F-C4 fires — the apparent allocation advantage is fully explained by
> commitment timing or count — if, after matching realized commit counts and
> write/storage/compute budgets, `Δ_A(c)`'s CI includes zero in *either*
> commitment regime.**

Two further failure paths, both reported rather than worked around:

- **F-SIGN.** `Δ_A(C1)` and `Δ_A(C2)` have opposite signs with both CIs
  excluding zero. Allocation's optimum then *depends on* the commitment regime,
  which contradicts separability directly and is a finding about **CCS-C1**, not
  only about CCS-C4.
- **F-DOMINATED.** The commitment main effect at fixed allocation,
  `Δ_C = |A3(C1) − A3(C2)|`, exceeds `max_c Δ_A(c)` by more than the pooled CI
  **and** `Δ_A` fails P1. Reported as: on this world, when is more important
  than where.

All three are satisfiable by some possible policy, and each has a concrete
numeric form. That check is deliberate — EXP-003's sibling track froze a
criterion that no arm could meet (`modular-consolidation` D7), and the umbrella
adopted "a preregistered success criterion must be satisfiable by some possible
policy" as programme rule 13.

### Interaction

The `A × C` interaction is **reported but not a success criterion**. A small
interaction with P1 ∧ P2 satisfied is the strongest available evidence for
separability. A large interaction is evidence against it, and F-SIGN is its
sharp form.

## 5. Resource matching

Inherited from EXP-001's `RESOURCE-NORMALIZATION.md`, plus what the commitment
axis newly requires.

| quantity | rule |
|---|---|
| write ceiling | one hard ceiling shared by every cell; no arm may exceed it |
| storage | episodic capacity identical; `storage_total` reported including footprint growth |
| router compute | charged per decision via `Ledger.charge_router`, reported separately |
| **commit compute** | **new.** Every promotion write is charged. Reported separately from allocation writes, so a regime that commits more cannot look free. |
| **realized commit count** | **new, and load-bearing.** Reported per cell. For the `A1 − A3` contrast it must match within a predeclared tolerance; A3 is constructed to make this hold. |
| commit budget | a hard cap `C_max` on promotions per lifetime, shared across cells within a regime, so that a regime cannot win by committing without limit |
| paired seeds | every contrast on identical seed sets; the runner raises otherwise |

**Why the commit-count match is the whole design.** Without it, an allocator that
chooses FAST more often gets more durable writes under the same cadence, and
`A1 − A3` would confound *what was placed where* with *how much was committed* —
reproducing EXP-001's ambiguity one level up. A3 exists to remove exactly that.

## 6. Novelty check

**Not claimed as novel.** Factorial designs; separating *what to store* from
*when to consolidate*; consolidation scheduling; sample-selection versus
replay-scheduling ablations in continual learning; fixed-cadence consolidation
as a control. All of this is standard, and `docs/LITERATURE.md` already names the
work that owns the allocation half.

What is claimed, conditionally, is narrow: **the crossed cell**. The CCS
programme currently has both margins and neither interaction —

| | allocation varied | commitment varied |
|---|---|---|
| `state-promotion` B3/B4/B5 | fixed | **varied** (fixed / random / gated) |
| `plasticity-routing` EXP-001 | **varied** | conflated with allocation |
| **EXP-003** | **varied** | **varied** |

Neither existing track can produce an interaction term, because each holds
constant the thing the other varies — and EXP-001 cannot even hold commitment
constant, since it has no separate commitment. EXP-003 is the smallest design
that crosses them.

This is a contribution about **experimental attribution**, not about
architecture, matching the posture of the rest of the programme.

## 7. Can this actually identify ALLOCATE independently of COMMIT_INTERNAL?

The honest answer is **yes for a specific and bounded sense, and no for the
strongest reading of CCS-C4**. Four points, in decreasing order of comfort.

### 7.1 What it does identify — cleanly

`Δ_A(c)` at matched realized commit counts is a genuine allocation-content
effect that **cannot** be attributed to commit timing or commit volume, because
both are held fixed by construction within the contrast. That is exactly the
inference EXP-001 could not license, and it is the gap the umbrella flagged.

### 7.2 It can detect failure of separability, which is rarer and more valuable

A factorial is one of the few designs that can *refute* the decomposition rather
than merely fail to support it. F-SIGN is a direct contradiction of CCS-C1's
claim that ALLOCATE and COMMIT_INTERNAL are separable operators. Most designs in
this programme can only decline to support a claim; this one can kill one.

### 7.3 The factors are not fully orthogonal, and this is a real limitation

Allocation determines the **supply** of commit candidates. An allocator that
rarely chooses FAST produces few promotions under any cadence, so the commitment
factor's effective range is not constant across allocation levels.

- For the **primary contrast** this is handled: A3 matches A1's realized action
  counts, so supply is equal where it matters.
- For the **interaction term** it is only partly handled. A2 (`HEURISTIC_EXT`)
  has no reason to produce A1's FAST count, so the `A2` row's cells differ in
  commit supply as well as in allocation content. The interaction should
  therefore be read as descriptive, and `SUPPLY_MATCHED_RANDOM` is included as
  a secondary arm that forces equal FAST count by construction rather than
  matching post hoc.

**This limitation must be stated in any result, not discovered afterwards.**

### 7.4 It settles separability from commitment *timing*, not from commitment *policy*

COMMIT_INTERNAL is operationalised here as a schedule, because the owner
constraint forbids importing State Promotion's unvalidated evidence gate. A
result therefore supports:

> allocation is separable from commitment **timing**

and not the broader:

> allocation is separable from commitment **as such**, including evidence-gated
> commitment

An evidence-gated third regime would close that, but only once such a gate is
independently validated. Until then this is a **partial-scope settlement path**,
and the umbrella should record it as one rather than as a full falsifier for
CCS-C4.

### 7.5 Benchmark calibration does not carry over

Removing `SLOW` from the action space changes the world. EXP-001's frozen
optimum was `NOISE→IGNORE, ONE_OFF→FAST, LOCAL→EPISODIC, STABLE→SLOW`; with
`SLOW` unreachable directly, `STABLE` must route through FAST and survive
promotion. So the reduced-action world must be re-calibrated from scratch on
development seeds: non-ceiling, non-floor, and with a bijective optimum over the
`3^4 = 81` mappings **per commitment regime**. If the optimum is not bijective in
a regime, the three-action space is not justified there and that regime is
invalid — the direct analogue of EXP-001's K8.

**Assuming EXP-001's calibration transfers would be the most likely way to get
this experiment wrong.**

## 8. Discipline carried over from EXP-001

Unchanged, and non-negotiable:

- preregistration committed **before** the runner, and before any confirmatory seed;
- **satisfiable** success and failure criteria (§4), checked for satisfiability explicitly;
- development / confirmatory seed separation, disjoint and frozen in `config.py`,
  derived deterministically via `derive_seeds` from a declared label so the sets
  cannot be reshaped to suit a result;
- the L1–L7 leakage suite, unchanged, plus the L5b cross-world attribution gate;
  the commitment operator is **not** a router and receives no features, which
  must be asserted mechanically;
- machine-readable manifests pinning source-tree hash, config hash, seed,
  environment, every resource total, realized commit counts and the leakage
  verdict;
- **reveal order** — every gate evaluated and passed before any comparative
  metric is computed, printed or aggregated;
- negative results preserved, including aborted attempts and failed controls
  (L5a's retention as permanently-failed and non-gating is the precedent);
- no GitHub Actions; local gates only.

## 9. What adoption would require

Before this is more than a design:

1. The owning track adopts the node and freezes the preregistration.
2. The reduced-action world is re-calibrated on development seeds, per §7.5.
3. `k` for `CADENCE_k` is frozen from a criterion mentioning no allocation arm.
4. `C_max` and the commit-count tolerance are frozen.
5. A commitment operator is implemented in `substrates.py` with tests asserting
   it receives no decision-time features.
6. Confirmatory seeds are derived and frozen, disjoint from `DEV_SEEDS`,
   `POLICY_SEEDS`, `AUDIT_SEEDS`, `CONFIRMATORY_SEEDS` and `REPLICATION_SEEDS`.

**None of this has been done. No seed has been drawn.**
