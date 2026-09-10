# EXP-004 preregistration — DRAFT, NOT FROZEN

**Status: DRAFT. NOT ADOPTED, NOT FROZEN, NOT EXECUTED.** No seed drawn, no code
written. Not a preregistration until the owning track adopts it, fills every
`TO BE FROZEN` field from development seeds, and commits it before a runner
exists. Rationale and the identifiability analysis are in
[`EXP-004-DESIGN.md`](EXP-004-DESIGN.md).

Depends on EXP-003 being adopted and read out first: EXP-004's C0 rung **is**
EXP-003's regime, and serves as the bridge that makes the two comparable.

## 1. Question

Is allocation content separable from an **endogenous** internal-commitment
policy — one that decides when to commit from what it has observed — and does
separability depend on how much of the substrate that policy can see?

## 2. Factors

**A — allocation**, action space `{IGNORE, EPISODIC, FAST}` per EXP-003; `SLOW`
unreachable by the allocator.

| level | |
|---|---|
| `A1 LEARNED` | ES-trained over the 15-feature whitelist |
| `A2 HEURISTIC_EXT` | calibrated fixed rule, matched search budget, never weakened |
| `A3 ACTION_MATCHED_RANDOM` | matched to A1's realized per-action counts within seed and rung |

**C — internal commitment**, decides only *when* the pending FAST window
promotes; never *which* items.

| level | input set | coupling |
|---|---|---|
| `C0 EXOGENOUS` | none (fixed cadence `k`) | none — EXP-003 bridge |
| `C1 STREAM-ADAPTIVE` | S-features only | none, asserted by L9 |
| `C2 STATE-COUPLED` | S ∪ X | genuine |
| `C3 LEARNED` | S ∪ X, ES-trained | genuine, optimised |

S/X partition exactly as in the design §3. **The partition is frozen with the
protocol**; moving a feature between S and X after seeing results invalidates the
run.

**H — horizon**, reported factor: `TO BE FROZEN`, minimum two stream lengths.
Required by programme rule 17 / CCS-C14, since T1 and T2 are ranking comparisons.

## 3. Validity gates — evaluated before any comparative metric

| id | gate | fires ⇒ |
|---|---|---|
| **G1** | C main effect at fixed allocation nonzero, CI excludes zero | if it fails, commitment does not matter on this benchmark and **no separability claim is produced in any direction** |
| **G2** | C0–C3 differ in realized commit count and timing distribution | a rung indistinguishable from a clock is a clock; drop it, do not report it as a rung |
| **G3 / L9** | C1's input vector is **bit-identical across allocation arms** on the same seed | if it fails, C1 is not allocation-blind and the clean-identification rung does not exist |
| **V1–V4** | non-ceiling, non-floor, depth-stratified, **bijective optimal mapping per rung** | per EXP-003; recalibrated, not inherited |
| **L1–L8** | full leakage suite incl. L5b and EXP-003's commitment blindness | unchanged |

**G1 is the satisfiability check.** SDW-1 was built to make *allocation* matter
and has never been asked whether *commitment timing* does. This gate failing is a
live possibility and is a legitimate outcome, not a bug to tune away.

## 4. Tests and criteria

### T1 — Oracle-mapping invariance (primary, ground truth)

Exhaustive search over all `3^4 = 81` class→action mappings, independently per
`(C, H)` cell.

- **Supports separability:** the argmax mapping is identical across C0–C3 at each
  horizon.
- **Refutes separability:** the mapping differs across rungs. **This falsifies
  CCS-C4** — allocation's ideal setting depends on the commitment regime in the
  world itself — and no policy-level result can rescue it.

Report the full mapping table, not just whether it changed.

### T2 — Transplant penalty (primary, policy level)

Train `A1` under each rung; evaluate every trained policy under every rung.
Policies pinned by SHA-256 and loaded without retraining, per the L5b precedent.

`Δ_ij = J(A_j under C_j) − J(A_i under C_j)` — the loss from having trained under
the wrong regime.

- **Supports:** `max_ij Δ_ij` within equivalence margin `δ_T` (`TO BE FROZEN`).
- **Refutes:** any `Δ_ij` exceeding `δ_T` with CI excluding it.

### T3 — Interaction equivalence (secondary, effect level)

**TOST against predeclared margin `δ_I` (`TO BE FROZEN`)**, on the allocation
effect `A1 − A3` across rungs. Three declared outcomes:

| CI vs ±`δ_I` | recorded as |
|---|---|
| inside | separability supported at this rung |
| excludes | separability refuted at this rung |
| wider | **INCONCLUSIVE — underpowered.** May **not** be reported as "no interaction found". |

A power analysis on development seeds must show `δ_I` is reachable at the planned
seed count, or the seed count rises. Equivalence testing needs more seeds than
superiority testing; budgeting as though it does not is the standard way this
design fails.

### Disagreement between tests is a result

T1 holding while T2 fails means the ground truth is separable but policies
entangle — a finding about **learnability**, not architecture, and it must be
reported that way rather than as a CCS-C4 failure.

## 5. Mediation accounting — mandatory at C2/C3

At C2/C3, realized commit behaviour is a **mediator** of allocation's effect.

1. **Report the total effect** — allocation varied, commit behaviour free to
   respond. This is the headline.
2. **Report realized commit counts and timing per cell.** Always. A total effect
   accompanied by a large commit-count difference is not the same finding as one
   without.
3. **The controlled direct effect may be reported additionally**, obtained by
   replaying a recorded commit sequence while varying allocation. When reported,
   it must carry the sentence that it is identifiable **only because this is a
   simulator with randomized assignment and full observability**, and is not a
   quantity an observational study could recover.
4. **Never report the CDE as the effect of allocation** without that
   qualification. Conditioning on a mediator and calling the remainder "the
   effect" is the specific error this section exists to prevent.

## 6. Seeds

Derived via `config.derive_seeds` from declared labels, verified disjoint from
**every** seed set enumerated in `config.py` — enumerated programmatically, not
listed by name.

| role | label | status |
|---|---|---|
| development | `EXP-004-dev-seeds-v1` | TO BE DERIVED |
| confirmatory | `EXP-004-confirmatory-seeds-v1` | TO BE DERIVED |

Count `TO BE FROZEN`, informed by the T3 power analysis and by EXP-001R's
reasoning that five seeds and 20 000 bootstrap resamples do not make an effect
broadly established.

## 7. Frozen configuration — all TO BE FROZEN from development seeds

Cadence `k` for C0; thresholds for C1/C2; ES budget and policy-seed rule for
C3 and A1; `δ_T`, `δ_I`; horizons; V2/V3 thresholds; the recalibrated
reduced-action world; the S/X feature partition.

## 8. Reveal order

G-gates and V-gates → L-gates → satisfiability confirmed → resource and
commit-count accounting validated against the lock → **then** T1, then T2, then
T3. The runner halts before the first comparative metric if any gate fails.
Aborted attempts retained with a note.

## 9. Prohibited

- Moving a feature between S and X after seeing any result.
- Reporting a wide T3 interval as "no interaction found".
- Reporting a controlled direct effect as the allocation effect (§5.4).
- Dropping a rung that fails G2 and describing the remainder as the full ladder.
- Claiming separability if **G1** failed.
- Inheriting EXP-003's calibration instead of re-establishing V1–V4.
- Running at one horizon and describing the result as separability in general.
- Importing any `state-promotion` selection mechanism; commitment here chooses
  *when*, never *which*.

## 10. Predeclared ledger consequence

Per design §7, binding:

- **T1 ∧ T2 ∧ T3 support, G1–G3 passed** → CCS-C4 may move to
  `confirmatory-supported`, **scoped to SDW-1 synthetic substrates**, residue
  being benchmark generality rather than commit-family coverage.
- **T1 refutes** → **CCS-C4 falsified**, and CCS-C1's separability premise takes
  the same hit.
- **T1 holds, T2/T3 fail** → a learnability finding; CCS-C4 stays unresolved.
- **G1 fails** → no separability claim in any direction; the experiment reports
  that commitment timing does not matter on SDW-1, which is itself worth
  recording.

## 11. Scope of any result

Supports at most: *on SDW-1-style synthetic substrates, at the tested horizons,
allocation content is separable from these four commitment families.*

Not supported: separability from commitment families outside the tested ladder;
separability from evidence-**gated** commitment that selects which items promote;
any language-model claim. EXP-002 remains the LM track and is unaffected.
