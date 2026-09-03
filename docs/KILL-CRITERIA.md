# Falsification and kill criteria

Preregistered. A criterion that fires is reported, not worked around.

## Benchmark-invalidating criteria

If any of these fire, **no comparison on that configuration means anything**.
Recalibration is permitted only on development seeds and must be logged as a
pre-result amendment.

| ID | Fires when | Meaning |
|---|---|---|
| **K3** | any leakage test L1–L7 fails | the router had information it could not have at decision time; the run is void |
| **K4** | a single-depth control (`ALL_EPISODIC` / `ALL_FAST` / `ALL_SLOW`) matches or beats the learned router | the world is not depth-stratified; there is nothing for routing to decide |
| **K5** | every arm exceeds 0.95 task utility | ceiling effect |
| **K6** | ORACLE objective < 0.45 | floor effect; the world is unsolvable or dominated by noise |
| **K8** | the optimal class→action mapping is not a bijection | the declared four-action space is not justified by the benchmark |

## Hypothesis-falsifying criteria

These do **not** invalidate the experiment. They falsify the hypothesis, and a
negative result here is a publishable outcome, not a failure to be retried.

| ID | Fires when | Conclusion |
|---|---|---|
| **K1** | `LEARNED` does not exceed `HEURISTIC` by a paired-bootstrap 95% CI excluding zero | **learned routing adds nothing over a fixed rule.** This replicates Yoon (2026, arXiv:2606.30067) in a new setting and is reported as the headline result |
| **K2** | `LEARNED − SHUFFLE_TRAINED` is not positive with a paired 95% CI excluding zero | an identically specified policy trained without any prefix→future-utility relationship does just as well, so the advantage is not utility-driven routing |
| **K7** | `LEARNED`'s advantage over `HEURISTIC` disappears under `CAPACITY_MATCHED` or under compute matching | the gain was capacity or compute, not routing |
| **K9** | `PRIVILEGED_TASKID` − `HEURISTIC` ≥ `LEARNED` − `HEURISTIC` | privileged context identity explains as much as learned inference; the interesting quantity was never routing |
| **K10** | the time-shuffled control (L5) still beats `RANDOM_MATCHED` | the router is exploiting structure other than genuine future utility |

### Why K2 is defined on `SHUFFLE_TRAINED`

`RANDOM_MATCHED` preserves only the marginal `P(A)`. The learned router also has
`P(A | X)`, and L5a measured that conditional resource sense alone beats the
marginal control by +0.136 in a world where future utility is pure noise. A
control that matches action frequencies therefore cannot carry a causal claim.

`SHUFFLE_TRAINED` holds the network, the feature whitelist, the ES budget, the
policy seeds and the selection rule fixed, and varies only whether training saw
a real prefix→future-utility relationship. `LEARNED − SHUFFLE_TRAINED` is
consequently the contrast that isolates the hypothesis.

`RANDOM_MATCHED` and `A_util` are retained as **secondary diagnostics** and are
still reported, but they no longer bear the attribution claim.

### The superseded `A_util` framing

`RANDOM_MATCHED` matches the learned router's *marginal* action distribution,
not its conditional structure. Measured, that turns out to be far too weak a
control: in the L5 time-shuffled world, where future utility is unpredictable by
construction, the learned policy still beat its own matched control by +0.126 —
slightly *more* than the +0.111 it managed in the real world. Conditional
resource sense alone clears that bar.

The corrected statistic subtracts the shuffled-world advantage from the real
one:

```
A_util = [obj(LEARNED) − obj(RANDOM_MATCHED)]_real
       − [obj(LEARNED) − obj(RANDOM_MATCHED)]_shuffled
```

Both worlds are trained with an identical procedure and budget and differ only
in whether future utility is predictable, so `A_util` isolates the part of the
advantage that actually depends on learning where writing pays off. At the
configuration that produced EXP-000 it was **−0.015**.

## Prior

The prior on K1 *not* firing is deliberately set low. Yoon (2026) reports a
simple similarity rule matching or beating a learned RL allocation controller
under fixed capacity, and that is the closest published comparison to this one.
The primary comparator is therefore the fixed heuristic, calibrated on the same
grid budget as the learned router's hyperparameters. **Deliberately weakening
the heuristic would be the most direct way to manufacture a positive result and
is prohibited.**

## Success criterion for H1

`LEARNED` beats `HEURISTIC` on the objective with a paired-bootstrap 95% CI
excluding zero, on held-out confirmatory seeds, **and** survives K2, K7, K9, and
K10, **and** every leakage test passes.

Anything less is a negative or inconclusive result and is reported as such.

## Protocol commitments

*   Preregister before inspecting confirmatory results.
*   Development and confirmatory seeds are disjoint and frozen in
    `src/plasticity_routing/config.py`.
*   Calibration uses controls and the oracle only; the learned router is never
    run during benchmark calibration.
*   Do not tune thresholds, costs, learning rates, or the world to make the
    proposed method win.
*   Preserve all negative results, including the ones already recorded in
    [`BENCHMARK.md`](BENCHMARK.md) §6.
*   Inspect in order: leakage → mechanical validity → resource budgets →
    comparative metrics. Never the reverse.
