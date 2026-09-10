# EXP-004 — Allocation × **Endogenous** Internal Commitment

**Status: DESIGN ONLY. Not adopted, not preregistered, not executed.**
No code, no seed drawn. Successor to
[`EXP-003-DESIGN.md`](EXP-003-DESIGN.md), which closes the *exogenous* half of
CCS-C4 and explicitly declines the rest.

## 1. The half EXP-003 leaves open

EXP-003's binding scope says a successful run supports separability of allocation
content from **exogenous** commitment timing — a schedule fixed in advance that
does not respond to the stream — and does **not** establish separability from a
learned or adaptive internal-commit policy.

That residue is not a technicality. A clock **cannot interact** with allocation:
it has no inputs, so nothing the allocator does can change what it does. An
endogenous commit policy reads the same stream the allocator is acting on, and in
the general case reads state the allocator has just modified. Separability from a
clock is close to trivial; separability from a policy is the real claim.

## 2. Why this is hard, stated precisely

With an endogenous commit policy `C`, holding `C`'s **function** fixed does not
hold `C`'s **behaviour** fixed. Varying the allocation policy `A` changes the
substrate, which changes `C`'s inputs, which changes when it commits.

```text
A ──▶ substrate state ──▶ X-features ──▶ C's decisions ──▶ outcome
 │                                                            ▲
 └────────────────────────────────────────────────────────────┘
                    direct path, not through C
```

Realized commit counts therefore differ across `A` **even at fixed `C`**. The
control that carries EXP-003 — `ACTION_MATCHED_RANDOM`, which matches realized
commit volume by construction — cannot be constructed here, because `C` decides
the volume, not the design.

Worse, the naive repair is wrong. Matching or conditioning on realized commit
count is **conditioning on a mediator**: commits sit on the causal path from
allocation to outcome, so removing them removes part of the effect being
measured. Any design that "controls for commits" without saying so is estimating
a controlled direct effect while reporting it as a total one.

## 3. The information set is the independent variable

The existing 15-feature decision-time whitelist in `features.py` already
partitions along exactly the line this experiment needs.

**S-features — stream-derived, allocation-invariant.** Recomputable from the
stream prefix alone; nothing the allocator does changes them:

`bias`, `is_novel`, `log_times_seen`, `presentation_recency`,
`log_times_queried`, `query_recency`, `value_revised`, `value_agreement`,
`past_query_hit_rate`, `t_frac`

**X-features — substrate-derived, allocation-coupled.** Every one is a function
of what the allocator has written:

`episodic_occupancy`, `fast_norm`, `slow_norm`, `write_budget_remaining`,
and `pred_error` — which is `1 - utility(answer, value)` where `answer` is read
back out of the substrate, so it is allocation-coupled despite looking like a
stream quantity.

**That partition is the design.** Separability is not binary; it is a function of
how much the commit policy can see. Four rungs:

| rung | commit policy sees | coupling to allocation |
|---|---|---|
| **C0** `EXOGENOUS` | nothing — fixed cadence | none (EXP-003's regime, carried as bridge) |
| **C1** `STREAM-ADAPTIVE` | S-features only | **none by construction** |
| **C2** `STATE-COUPLED` | S ∪ X | genuine, by construction |
| **C3** `LEARNED` | S ∪ X, ES-trained | genuine, and optimised |

C1 is the load-bearing rung: a commit policy that genuinely adapts to the stream
while being **provably blind to allocation**. At C1 the two operators are
orthogonal by construction and separability is cleanly identifiable. At C2/C3 it
is not, and §6 says exactly what remains recoverable.

Committing is schedule-free at every rung above C0: the policy decides *when* the
pending window promotes, never *which* items promote. Selection stays out, so
this is not `state-promotion`'s evidence gate — see §8.

## 4. Three tests, in increasing dependence on learned policies

Separability means: *the best setting of one operator does not depend on the
setting of the other.* Three ways to ask that, from cheapest and cleanest to most
realistic.

### T1 — Oracle-mapping invariance *(ground truth; no policy involved)*

Recompute the empirically optimal class→action mapping by **exhaustive search**
independently under each commitment rung.

- **If the optimal mapping is identical across C0–C3**, allocation's ideal
  setting does not depend on the commitment regime, in the world itself,
  regardless of any policy's ability to find it. That is separability at ground
  truth.
- **If it changes**, separability is **false at the level of the world**, and no
  learned-policy result can rescue it.

This is the strongest test in the design and it is nearly free — the search space
is `3^4 = 81` mappings per regime. It should be the first thing run and the first
thing reported.

### T2 — Transplant penalty *(the practitioner's question)*

Train an allocation policy under commitment regime `C_i`; evaluate it under
`C_j`, all cells crossed. If allocation is separable from commitment, a policy
trained under one regime should not lose materially under another. If it must be
re-learned per regime, they are entangled.

**This reuses the L5b crossover machinery verbatim** — that control already
trains two policies (`real`, `shuffled`), pins both by SHA-256 in
`SELECTED_POLICY_SHA256`, loads without retraining, and evaluates the full cross
on one-shot audit seeds. EXP-004 changes what the two policies were trained on,
not how the crossover works.

Primary quantity: **transfer penalty** `Δ_ij = J(A_j under C_j) − J(A_i under C_j)`
— what you lose by having trained under the wrong commitment regime.

### T3 — Interaction equivalence *(effect-level)*

Is the *allocation effect* invariant across commitment regimes?

**This must be an equivalence test, not a null-hypothesis test.** Separability is
a claim that an interaction is *absent*, and a non-significant interaction is not
evidence of absence — it is frequently evidence of insufficient power. So: **TOST
against a predeclared equivalence margin δ**, with three possible outcomes
declared in advance:

| outcome | reading |
|---|---|
| interaction CI **inside** ±δ | separability supported at this rung |
| interaction CI **excludes** ±δ | separability **refuted** at this rung |
| CI **wider** than ±δ | **inconclusive — underpowered.** Not "no interaction found." |

The third outcome is the one that gets misreported in practice, and it is
declared here so it cannot be quietly turned into support.

## 5. Satisfiability — the gate that makes all three tests meaningful

**If the commitment factor does not matter, every separability test passes
vacuously.** An allocation effect is trivially invariant across regimes that are
themselves indistinguishable.

Required before any separability claim, on development seeds:

- **G1 — commitment has range.** The C main effect at fixed allocation is nonzero
  with a CI excluding zero. If commitment regime does not change outcomes on this
  benchmark, EXP-004 cannot test separability from it and must say so instead of
  reporting an invariance.
- **G2 — the rungs are distinguishable.** C1/C2/C3 must differ from C0 and from
  each other in *realized commit behaviour* (count, timing distribution). A
  "state-coupled" policy whose commit sequence is statistically indistinguishable
  from a clock is a clock.
- **G3 — C1 is provably allocation-blind.** Mechanically assert that C1's input
  vector is bit-identical across allocation arms on the same seed. This is a
  test, not a claim; it is what licenses C1's clean identification.

G1 is the satisfiability check programme rule 13 requires, and it is the one most
likely to fail, because SDW-1 was designed to make *allocation* matter and has
never been asked whether *commitment timing* does.

## 6. What is and is not identifiable — the honest answer

The user-facing question is whether this design can identify ALLOCATE
independently of an endogenous COMMIT_INTERNAL. Rung by rung:

**C1 — yes, cleanly.** Inputs are provably allocation-invariant (G3), so varying
allocation cannot change commit behaviour through any path. `A` and `C` are
orthogonal by construction and all three tests are unconfounded.

**C2/C3 — partially, and only because this is a simulator.** Commit behaviour is
a *mediator* of allocation's effect. Two quantities are separable in principle:

- the **total effect** of allocation, including its effect via commit behaviour —
  always estimable, and what T2's transplant penalty measures;
- the **controlled direct effect**, holding commit behaviour fixed — requires no
  unmeasured confounding of the mediator–outcome relation. In an observational
  setting that is untestable; **here it is satisfied by construction**, because
  allocation is randomly assigned, every variable is observed, and the commit
  policy can be held to a recorded sequence and replayed. The simulator is what
  makes the CDE identifiable at all, and that must be stated whenever it is
  reported.

**T1 is unaffected by any of this** — it involves no policy and no mediation, only
exhaustive search over ground truth.

**What no rung closes.** Separability could hold for these commit families and
fail for another. C3 is the broadest rung available, but it is one policy class
trained by one optimiser on one benchmark. EXP-004 does not deliver "allocation
is separable from commitment"; it delivers "separable from these four commitment
families on SDW-1", which is a large advance on "separable from a clock" and
still not the unqualified claim.

## 7. Predeclared ledger consequence

Binding, in the manner of EXP-003 §12, and fixed before any seed is drawn.

**If T1 ∧ T2 ∧ T3 support separability at C1–C3, with G1–G3 passed:**
CCS-C4 may move to `confirmatory-supported`, **scoped to SDW-1 synthetic
substrates**. The residue would then be *benchmark generality*, not
commit-family coverage — a different and more ordinary kind of gap than the one
EXP-003 leaves.

**If T1 fails** — the optimal mapping changes across commitment regimes —
**CCS-C4 is falsified**, not merely unsupported, and CCS-C1's separability
premise takes the same hit. This is the outcome most worth running the
experiment for.

**If T1 holds but T2/T3 fail**, the ground truth is separable while policies
entangle. That is a finding about *learnability*, not about the architecture, and
must be reported as such rather than as a failure of CCS-C4.

**If G1 fails**, no separability claim is produced at all, in any direction.

## 8. Not `state-promotion`'s mechanism, and an ownership question

Every commit policy here decides **when** the pending window promotes, never
**which** items promote. Selection is excluded at every rung. So this is not
`state-promotion`'s evidence-gated consolidation, and nothing from that track is
imported, validated or assumed.

**But this experiment does reach into COMMIT_INTERNAL, which is
`state-promotion`'s domain.** EXP-003 set the precedent that `plasticity-routing`
hosts the crossing, and continuity argues for keeping it here: same substrate,
same world, same lock machinery, same leakage suite, and T2 reuses L5b directly.

That is a programme-level ownership call, not one this document should make
silently. It is flagged for the owner, with the recommendation that
`plasticity-routing` hosts and `state-promotion` is notified rather than
consulted for permission — since nothing here constrains that track's protocol.

## 9. Discipline

Unchanged from EXP-001/EXP-003 and non-negotiable: preregistration before the
runner; satisfiability checked explicitly; disjoint dev/confirmatory seeds
derived from a declared label and verified against **every** seed set declared in
`config.py`; the L1–L7 leakage suite plus L5b, plus L8 commitment-blindness from
EXP-003 and **L9** (C1 allocation-blindness, §5 G3); machine-readable manifests;
reveal order enforced; negative results and aborted attempts retained; no GitHub
Actions.

**One addition specific to this design.** Programme rule 17 — short-horizon
mechanism rankings do not predict long-horizon rankings, now
`confirmatory-supported` as CCS-C14 — applies directly here, because T1 and T2
are *ranking* comparisons. Horizon must therefore be a reported factor at
reduced resolution, and any separability finding is scoped to the horizons
tested. A separability result at one stream length does not license a claim about
separability in general, and the design must not be run at a single horizon and
then described as though it had been.
