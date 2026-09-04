# EXP-001R preregistration — replication of the confirmatory result on 32 untouched seeds

**Status: preregistered, NOT RUN.** Written 2026-09-04, after EXP-001
confirmatory completed and before any replication seed was executed.

## 1. Why this exists

EXP-001 supports H1 under its protocol, but on **five** seeds. The preregistered
paired-bootstrap criterion passed; that is what it was for. It does not follow
that the effect is broadly established: 20,000 bootstrap resamples do not turn
five independent lifetimes into 20,000 observations. The bootstrap describes
uncertainty about the mean of a small sample, not the breadth of the effect.

EXP-001R raises the inferential sample to 32 while changing **nothing else**.

## 2. What is frozen and must not change

Everything scientific is inherited unchanged from protocol v1.2:

*   the world, substrate and cost configuration (config hash `7a2d06f526ace1c0`);
*   the objective and its weights;
*   the four-action space;
*   **both policy checkpoints**, by SHA-256 (`SELECTED_POLICY_SHA256`) — the
    policy is *not* retrained;
*   **the comparator artefact**, by SHA-256 (`MATCHED_HEURISTIC_SHA256`) — the
    matched-budget search is *not* re-run;
*   the ORACLE and designed mappings;
*   every contrast definition and success criterion.

This is a replication, not a second experiment. If any of the above changes it
is no longer EXP-001R.

## 3. Seeds

`REPLICATION_SEEDS` — 32 seeds derived deterministically in `config.py` from
the literal label `EXP-001R-replication-seeds-v1` by SHA-256, in the range
`[40000000, 49999999]`.

Derivation rather than hand-picking is deliberate: it makes the provenance
machine-checkable, so nobody can later claim the list was chosen to suit a
result. `tests/test_config_seeds.py` re-derives the list and asserts pairwise
disjointness from the development, audit and confirmatory sets.

These seeds are **untouched**: not development, not audit, not confirmatory.
They are spent once.

## 4. Hypotheses

**H1R (primary).** `LEARNED − HEURISTIC_EXT > 0` with a paired-bootstrap 95% CI
excluding zero, on the 32 replication seeds.

**H2R (attribution).** `LEARNED − SHUFFLE_TRAINED > 0` with a CI excluding zero,
and the advantage survives capacity matching (K7) and exceeds what privileged
context identity buys (K9).

**H3R (consistency, secondary).** The EXP-001 point estimate
`K1 = +0.0907` lies inside the EXP-001R 95% CI. This is reported as a
consistency check, **not** as a pass/fail gate: a replication whose interval
excludes the original estimate while still excluding zero has replicated the
*effect* and refined the *magnitude*, and saying otherwise would be a
sleight of hand.

## 5. Pass rule

EXP-001R **replicates** iff H1R and H2R both hold and no benchmark-invalidating
gate (K3–K6, K8) fires.

If H1R fails, the honest statement is that the EXP-001 result did not replicate
at n=32, and the EXP-001 claim must be narrowed accordingly in the paper. That
outcome is publishable and is not to be rescued by retraining, reselecting the
policy seed, re-searching the comparator, or extending the seed set.

**Reporting is unconditional.** The result is reported whichever way it falls,
alongside the EXP-001 numbers, with both sample sizes stated.

## 6. Procedure

1. Freeze protocol **v1.3**, asserting field-by-field equivalence to v1.2 on
   every scientific field. The only admitted delta is the replication runner and
   the seed list.
2. Run every claim-eligible arm plus the ceiling/privileged and disentanglement
   arms once on all 32 replication seeds, through a runner with the same
   guarantees as `run_exp001_confirmatory.py`: seeds only from the lock, no seed
   CLI, hash-verified artefacts, `CONFIRMATORY`-class manifests to a separate
   directory, one-shot, and the reveal order enforced in code.
3. Inspect in the preregistered order: leakage → manifest validation →
   resource ceilings → C1–C6 → K3–K6/K8 → comparative metrics last.
4. Recompute C5 by exhaustive 256-mapping search on the replication seeds.
   C6 may again be inherited, and must be labelled as inherited.

## 7. What EXP-001R cannot fix

It does not broaden the benchmark. SDW-1 remains one synthetic world with
linear associative substrates, and a 32-seed replication on the same generator
is evidence about that generator, not about continual learning in general. It
makes no language-model claim; EXP-002 remains separately gated.

Nor does it revisit policy selection. The policy was chosen on development data
over three seeds; EXP-001R inherits that choice rather than re-litigating it.
A study of selection sensitivity would be a different experiment with its own
preregistration.
