# Leakage policy and tests

> The routing decision may use only information available at decision time.

This is the failure mode most likely to produce an exciting and worthless
result, so the tests are written **before** meaningful results and gate every
run.

## The rule

A feature is admissible if and only if it is a function of

*   the stream prefix up to and including the current write event, and
*   the agent's own internal state.

Inadmissible: the hidden class, the regime id, the ideal action, the future
query schedule, any future utility, and every evaluator-only audit probe.
`features.FORBIDDEN_SOURCES` names these; `features.FEATURE_NAMES` is the
whitelist. Adding a feature requires editing the whitelist, which makes the
review surface exactly one small module.

## Training signal vs decision input

These are different and must not be confused:

*   **Decision input** — what `Router.act` receives. Prefix-only. Always.
*   **Training signal** — what the offline trainer optimises against. This
    *does* include outcomes occurring after a decision, because "learn where
    writing pays off" has no other meaning. This is ordinary offline
    policy-gradient credit assignment.

The trained policy is then frozen and evaluated on seeds it never saw. A run
where the policy is trained and evaluated on the same seed is a development
diagnostic and is classified `DEV_CALIBRATION`, never `CONFIRMATORY`.

## The tests

| ID | Test | Fails when |
|---|---|---|
| **L1** | *Feature-set purity.* The extractor signature takes explicit scalars/arrays, never an `Event`; no forbidden name appears in the whitelist; the vector length matches the whitelist. | a hidden field can reach a policy at all |
| **L2** | *Class-permutation invariance.* Permute hidden class labels while holding the observable stream fixed. Every legal router's decision sequence must be bit-identical. | a legal router's behaviour tracks permuted labels |
| **L3** | *First-encounter independence.* SDW-1 makes classes observationally identical on first sight. Estimate the mutual information `I(hidden class ; action)` over first encounters, against a 500-fold permutation null. | MI exceeds the permutation null (p < 0.05) — the router "knows" a class it cannot yet distinguish |
| **L4** | *Causal ordering / future blindness.* Re-run the router on a stream truncated after step `t`. Decisions up to `t` must be identical to the full-stream run. | any feature depends on the future |
| **L5** | *Time-shuffled control.* Retrain in a world where each query's target key is redrawn uniformly from the keys already written at that moment. The write stream, query count, and query timing are byte-identical, so all budgets are unchanged; only the link between an item's observable prefix and whether it will be needed later is destroyed. | the router's advantage over budget-matched random routing survives the shuffle |
| **L6** | *Privilege declaration.* Assert that the rollout engine passes `privileged=None` to every router with empty `privileged_fields`, and that `ORACLE`/`PRIVILEGED_TASKID` raise if run without their declared fields. | a legal arm silently receives privileged data |
| **L7** | *Probe non-interference.* Assert that evaluator audit probes leave episodic LRU order and every ledger counter unchanged. | the evaluator perturbs the system it measures |

L3 deserves emphasis because it is the test that would catch a *subtle* leak.
The others catch structural mistakes; L3 catches an information-theoretic one.

### L5's criterion is comparative, and why

The shuffle cannot destroy every regularity. Keys written earlier stay live
longer and so are queried more often, which leaves a residual
past-to-future query correlation (reported per seed by `scripts/audit_l5.py`).
A router may legitimately exploit that residue, so L5 does **not** demand a zero
advantage in the shuffled world. The preregistered criterion, fixed before the
script was first run, is:

> L5 passes iff the shuffled-world advantage over budget-matched random routing
> is at most **0.25×** the real-world advantage, or its paired-bootstrap 95% CI
> includes zero.

Both advantages are measured the same way — `objective(LEARNED) −
objective(RANDOM_MATCHED)` — each in its own world, each against a control
matched to that world's own learned action histogram. A large surviving
advantage would mean the router is winning for some reason other than genuine
future utility, and would make the real-world result uninterpretable.

L5 trains a second policy, so it is far more expensive than L1–L4/L6/L7. It runs
separately (`make l5`) and caches its verdict together with the source-tree hash
it was computed under; `scripts/audit_leakage.py` then verifies that the cached
verdict exists, passes, and is current for the tree being audited.

## Gate

`make leakage` runs the audit and writes `results/leakage_audit.json`. Every run
manifest embeds the audit outcome. `scripts/validate_runs.py` refuses to certify
any run whose leakage audit is missing or failing, and refuses to certify a
comparison that includes a non-`legal` arm among its claim-supporting arms.

A failing leakage test invalidates the run. It is not a warning.
