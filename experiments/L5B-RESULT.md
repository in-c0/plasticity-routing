# L5b — cross-world utility-shuffle negative control

**Classification: one-shot validity audit. Preregistered in
[`EXP-001-PREREG.md`](EXP-001-PREREG.md) §8a (Amendment L), committed
2026-09-03 before any cross-world number was inspected.**

**Verdict: PASSED.**

## Design

`R` = the policy trained on the real development worlds.
`S` = the *identically specified* policy trained on the time-shuffled
development worlds — same `LearnedRouter`, same 15-feature legal whitelist, same
ES budget (100 generations), same `POLICY_SEEDS = (0,1,2)`, same
best-on-development selection rule. The only difference is whether training
preserved the prefix→future-utility relationship.

Both are frozen artefacts pinned by SHA-256 in `config.SELECTED_POLICY_SHA256`;
the audit loads them and never retrains. Hash match verified at run time.

Evaluated once on `AUDIT_SEEDS = 91001..91032` — neither a training nor a
confirmatory set. The cross was deliberately **not** evaluated on the
development seeds, because `R` was selected for real-dev performance and `S` for
shuffled-dev performance, which would bias that comparison in `R`'s favour.

All four cells ran under one cost table and one substrate configuration
(config hash `125819c1d3634236`), and every cell respected the write ceiling.

## Result

Mean objective over 32 audit seeds:

| | evaluated on **real** | evaluated on **shuffled** |
|---|---|---|
| policy `R` (trained on real) | **+0.5130** | −0.0898 |
| policy `S` (trained on shuffled) | +0.0201 | +0.0183 |

| quantity | estimate | 95% CI | excludes 0 positively |
|---|---|---|---|
| `Delta_real = J_RR − J_SR` | **+0.4929** | [+0.4822, +0.5032] | yes |
| `I = (J_RR − J_SR) − (J_RS − J_SS)` | **+0.6011** | [+0.5902, +0.6122] | yes |

Paired bootstrap at the audit-seed level, 20,000 resamples, fixed seed. Both
quantities are positive on **32/32 individual audit seeds**:
`Delta_real` ranges +0.4216 … +0.5407 and `I` ranges +0.5275 … +0.6658.

## Reading

The pattern is a **true crossover**, not a level shift: `R` beats `S` on real
worlds (+0.513 vs +0.020) *and* `S` beats `R` on shuffled worlds (+0.018 vs
−0.090). `R` is not a universally better network — it is specifically better
when the evaluation world's utility structure matches the one its training saw,
and specifically worse when it does not.

That is the attribution the experiment needed and that `RANDOM_MATCHED` could
not supply. Training on a real prefix→future-utility relationship buys +0.493 on
the identical real environment over an otherwise identical policy trained when
that relationship had been destroyed.

## What this does and does not establish

It establishes that the learned router's advantage **depends on future utility
having been predictable during training** — the hypothesised informative
relationship, isolated against a null that retains every nuisance mechanism
(same network, same features, same budget, same seeds, same selection rule,
same resource accounting).

It does **not** establish H1. H1 is `LEARNED` versus the matched-budget fixed
rule on held-out confirmatory seeds, and those seeds remain unspent. L5b is a
validity gate, not the experiment.

## Relationship to L5a

L5a is retained permanently as **FAILED** (ratio 0.521 against a threshold of
0.25) and is not the attribution gate. Its threshold was never relaxed. What it
falsified is the adequacy of the marginal `RANDOM_MATCHED` control: that control
preserves only `P(A)`, while the learned router also has `P(A | X)`, so generic
conditional resource management clears it with no utility knowledge at all.

L5b is a **stricter** null than L5a, not a weaker one. Where L5a compared a
policy against a randomiser matched on its own action frequencies, L5b compares
two policies of identical class, capacity, budget and selection procedure that
differ only in what there was to learn.

## Artefacts

`results/l5b_cross_world.json` — per-seed values for all four cells, both
bootstrap results, both policy SHA-256s, the shared config hash, the full audit
seed list, the git SHA, the source-tree hash and the environment.
