# EXP-003 preregistration — DRAFT, NOT FROZEN

**Status: DRAFT. NOT ADOPTED, NOT FROZEN, NOT EXECUTED.**

This document is a candidate protocol. It is **not** a preregistration until the
owning track adopts it, fills every `TO BE FROZEN` field from development seeds,
and commits it before the runner exists. No seed has been drawn. No EXP-003 code
has been written.

Design rationale and the identifiability analysis are in
[`EXP-003-DESIGN.md`](EXP-003-DESIGN.md).

## 1. Question

Under matched write, storage, compute and **realized commit counts**, does the
*content* of a write-depth allocation policy change outcomes when the internal
commitment schedule is held identical — and does that effect replicate across
predeclared commitment regimes?

## 2. Architecture under test

Allocation action space reduced to `{IGNORE, EPISODIC, FAST}`. `SLOW` is
**unreachable by the allocator**. Durable content enters `slow` only through the
commitment operator, which promotes the pending window of FAST-allocated pairs
on a fixed schedule, with **no selection and no evidence gate**.

Mechanical requirements, to be asserted by tests:

1. No router may emit `SLOW`; a router that does invalidates the run.
2. The commitment operator receives **no decision-time features** and no
   utility signal. It is a clock, not a policy.
3. Promotion order within a window is the arrival order of pending items; no
   ranking, no filtering, no deduplication by utility.
4. Every promotion write is charged to the ledger as commit compute, separately
   from allocation writes.

## 3. Factors and cells

**A** ∈ {`LEARNED`, `HEURISTIC_EXT`, `ACTION_MATCHED_RANDOM`}
**C** ∈ {`IMMEDIATE`, `CADENCE_k`}

Six cells, fully crossed, paired seeds. Reported anchors outside the primary
factorial: `NEVER` (C0), `ORACLE_3`, `SUPPLY_MATCHED_RANDOM`.

`ACTION_MATCHED_RANDOM` is constructed **post hoc within each seed** to match
`LEARNED`'s realized per-action counts in that seed and regime, following the
`RandomMatchedRouter` precedent.

## 4. Hypotheses

**H1 (primary).** `Δ_A(c) = LEARNED − ACTION_MATCHED_RANDOM > 0` with a paired
bootstrap 95% CI excluding zero, in **both** commitment regimes, at matched
realized commit counts.

**H2 (secondary).** `LEARNED − HEURISTIC_EXT > 0` in both regimes. Reported;
**not** a settlement criterion for CCS-C4, because a learned-versus-fixed
contrast speaks to whether the policy must be learned, not to whether allocation
is separable from commitment.

**H3 (descriptive).** The `A × C` interaction is small relative to the `A` main
effect. Reported, not a success criterion, and read subject to §7.3 of the
design.

## 5. Success and failure criteria

CCS-C4 settlement is claimed **only** if H1 holds in both regimes with the same
sign.

| id | fires when | conclusion |
|---|---|---|
| **F-C4** | `Δ_A(c)` CI includes zero in **either** regime, after commit-count and budget matching | the apparent allocation advantage is fully explained by commitment timing or count. **CCS-C4 is not settled by this experiment, and EXP-001's partial support does not extend.** |
| **F-SIGN** | `Δ_A(C1)` and `Δ_A(C2)` have opposite signs, both CIs excluding zero | allocation's optimum depends on the commitment regime. Separability is contradicted; this is a finding about **CCS-C1**. |
| **F-DOMINATED** | `Δ_C` at fixed allocation exceeds `max_c Δ_A(c)` by more than the pooled CI **and** H1 fails | on this world, *when* dominates *where*. |

### Satisfiability check — required before freezing

Each criterion must be shown reachable by some possible policy, on development
seeds only:

- **H1 satisfiable**: `ORACLE_3 − ACTION_MATCHED_RANDOM` must exceed zero with a
  CI excluding zero in both regimes. If the ceiling policy cannot beat the
  count-matched control, no legal arm can, and the benchmark cannot test H1.
- **F-C4 satisfiable**: trivially, by a null.
- **F-SIGN satisfiable**: requires the two regimes to differ enough that a
  sign flip is possible in principle; verified by the C main effect at
  `ORACLE_3` being nonzero.

**A criterion that fails its satisfiability check invalidates the protocol, not
the method.** This clause exists because a sibling track froze a success rule
that no arm could meet, and the programme adopted satisfiability as a rule.

## 6. Benchmark validity gates

Per commitment regime, on development seeds. A regime failing any of these is
**invalid** and carries no comparison.

| id | gate |
|---|---|
| **V1** | non-ceiling: no legal arm exceeds 0.95 task utility |
| **V2** | non-floor: `ORACLE_3` objective ≥ TO BE FROZEN |
| **V3** | depth-stratified: `ORACLE_3` exceeds the best single-depth control by ≥ TO BE FROZEN |
| **V4** | **bijective optimum over the `3^4 = 81` mappings, per regime.** If not bijective, the three-action space is unjustified in that regime |
| **V5** | commitment has range: `ORACLE_3(C1) ≠ ORACLE_3(C0=NEVER)` with a CI excluding zero |
| **V6** | commit-count tolerance achievable: `|commits(LEARNED) − commits(ACTION_MATCHED_RANDOM)| ≤ TO BE FROZEN` |

**V4 and V5 are new relative to EXP-001** and are the gates most likely to fail,
because the reduced action space is a genuinely different world (design §7.5).

## 7. Leakage controls

The full L1–L7 suite, unchanged, plus the L5b cross-world attribution gate, plus
one addition specific to this design:

| id | control |
|---|---|
| L1–L4, L6, L7 | unchanged from EXP-001 |
| **L5b** | cross-world utility-shuffle negative control; gating |
| L5a | retained as permanently failed and non-gating |
| **L8 (new)** | **commitment blindness.** Assert mechanically that the commitment operator's promotion sequence is a function of the schedule and arrival order alone — invariant to a permutation of hidden class labels and to the utility signal. A commitment operator that can see utility is an evidence gate, which this design deliberately excludes. |

## 8. Seeds

Derived deterministically via `config.derive_seeds` from a declared label, so the
sets cannot be reshaped to suit a result — the `REPLICATION_SEED_LABEL`
precedent.

| role | source | status |
|---|---|---|
| development | `EXP-003-dev-seeds-v1` | TO BE DERIVED |
| confirmatory | `EXP-003-confirmatory-seeds-v1` | TO BE DERIVED |

Both must be verified disjoint from **every other seed set declared in
`config.py`** — enumerated programmatically rather than listed by name, since
that module gains seed sets as tracks are added (EXP-002's substrate-sufficiency
sets arrived on `main` at `9c2ddc3` while this document was being drafted). A
test must re-derive and compare against the full enumeration, so a later seed set
cannot silently overlap. Confirmatory seed count: TO BE FROZEN, not fewer than the 5 used by
EXP-001, and EXP-001R's reasoning — that five seeds and 20,000 bootstrap
resamples do not make an effect broadly established — argues for more.

## 9. Frozen configuration — all TO BE FROZEN from development seeds

- `k` for `CADENCE_k`, from a predeclared cadence set, by a criterion mentioning
  no allocation arm;
- `C_max` commit budget per lifetime;
- commit-count matching tolerance (V6);
- V2 and V3 thresholds;
- the re-calibrated reduced-action world configuration;
- ES budget and policy-seed selection rule, following EXP-001's
  best-on-development rule with `POLICY_SEEDS` reported rather than chosen post hoc;
- `HEURISTIC_EXT` search budget, matched to the learned arm's, and **never**
  weakened.

## 10. Reveal order

1. All L-gates and V-gates evaluated.
2. Satisfiability checks confirmed.
3. Resource and commit-count accounting validated against the frozen lock.
4. **Only then** are comparative metrics computed.

The runner must stop before step 4 if any gate fails, and must not compute,
print or aggregate any contrast before that point. Aborted attempts are retained
with a note, per the EXP-001 precedent.

## 11. Prohibited

- Tuning the world, cost table, cadence, commit budget, matching tolerance or
  thresholds after seeing any confirmatory result.
- Weakening `HEURISTIC_EXT`, or giving it a smaller search budget than the
  learned arm.
- Adding dummy writes or dummy commits to equalise a budget.
- Selecting among policy seeds by confirmatory performance.
- Allowing the commitment operator to observe utility, hidden class, or any
  decision-time feature.
- Reusing EXP-001's frozen calibration without re-establishing V1–V4 in the
  reduced action space.
- Reporting `H2` as evidence for CCS-C4.

## 12. Scope of any result

A result here would support, at most:

> On SDW-1-style synthetic substrates, allocation content is separable from
> commitment **timing**.

It would **not** support separability from evidence-gated commitment, and would
make **no** language-model claim. EXP-002 remains the LM track and is unaffected
by this document.

## 13. Negative-result path

If F-C4 fires, the result is published as a negative one: EXP-001's allocation
advantage did not survive the introduction of an independent commitment axis, and
CCS-C4 remains unsettled with its partial-scope evidence intact. That outcome is
more informative than the positive one for the programme, because it would mean
the umbrella's decision not to promote CCS-C4 on EXP-001 was not merely cautious
but correct.
