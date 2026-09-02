# Resource-normalization policy

Resource accounting is mandatory. A run that cannot report its consumption, or
that exceeds a ceiling, is **invalid** and may not be compared.

## Why this is not optional here

The claim under test is that *routing* helps. The three cheapest ways to fake
that claim are to give the proposed method more storage, more writes, or more
compute than its controls. The ledger exists to make each of those visible in
the manifest of every run.

## Common currency

All arms share one `CostConfig`. There is no per-arm cost table.

### Writes — parameter/value **elements exposed to a write**

| Action | Elements per write |
|---|---|
| `IGNORE` | 0 |
| `EPISODIC` | `value_dim` |
| `FAST` | `key_dim × value_dim` |
| `SLOW` | `key_dim × value_dim × slow_write_multiplier` |

A rank-1 delta update touches the whole matrix, so a parametric write really is
~`key_dim` times more expensive per item than an episodic entry. Consolidation
into the durable substrate runs `slow_consolidation_steps` passes, hence the
multiplier. This is the same element-write currency used by the sibling
`state-promotion` track, so the two tracks' budgets are directly comparable.

**Hard ceiling.** Every arm has the identical `write_element_ceiling`. When an
action would exceed it, the action is downgraded to `IGNORE` and counted in
`forced_ignores_budget`. No arm may exceed the ceiling; no arm is required to
*reach* it. An arm that spends less because its policy declined to write is
exhibiting the mechanism under test, not gaining an unfair advantage — but its
underspend is reported, and the budget-matched random control (A5) exists
precisely so that underspending cannot be mistaken for good routing.

### Storage — occupancy integral

Footprint summed over timesteps:

```
storage_element_steps = Σ_t [ |episodic_t| × value_dim  +  2 × key_dim × value_dim ]
```

The parametric term is constant for every arm, so it cancels in comparisons; the
episodic term grows with residency. That is exactly the non-parametric /
parametric trade-off and it must be charged, or "store everything episodically"
becomes free.

### Compute

*   **read compute** — episodic search cost plus parametric read cost per query;
*   **router decision compute** — `router.n_params` charged on *every* decision.

Router compute is reported **separately and additionally**, never folded
silently into a "compute matched" claim. This follows Amendment E of the sibling
track: a routing gate's forward pass is decision-time algorithmic compute and
must be visible. Every non-trivial router pays it, including the fixed heuristic
(charged as a small linear probe), so "the heuristic is free" is not an
unpriced advantage.

## Normalisation

Raw element counts are divided by shared reference denominators describing the
most expensive admissible behaviour, so each term lands in roughly `[0,1]` and
the objective weights are interpretable:

```
storage_ref = lifetime × (episodic_capacity × value_dim + 2 × key_dim × value_dim)
write_ref   = write_element_ceiling
compute_ref = lifetime × (episodic_capacity + 2 × key_dim × value_dim)
```

## Objective

```
J = task_utility
      − λ_forget  × forgetting
      − λ_storage × storage_norm
      − λ_write   × write_norm
      − λ_compute × compute_norm
```

with `task_utility` the mean per-query recall utility in `[0,1]` and
`forgetting` the mean retention regression measured by evaluator-only audit
probes.

**Known double-counting caveat, stated rather than hidden.** Forgetting is
partly reflected in `task_utility` already, because a forgotten fact scores
badly when queried. The separate `forgetting` term is retained because queries
are sparse and irregular, so utility alone under-measures retention between
query events. The consequence is that retention regressions are weighted more
heavily than a pure utility objective would weight them. `λ_forget` is frozen
before confirmatory runs and reported; results are also reported at
`λ_forget = 0` as a sensitivity check.

## Reporting requirements

Every run manifest records, without exception: write elements (total and by
action), storage element-steps, read compute, router compute, total compute,
normalised terms, all penalty components, budget utilisation, forced-ignore
count, and episodic evictions.
