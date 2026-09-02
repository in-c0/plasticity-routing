# Architecture

## The question this architecture is built to answer

Given a new experience, can an agent learn **which part of its substrate should
change** — including the option of not changing at all — from delayed future
utility, under interference and storage/compute constraints?

The architecture is deliberately minimal. Every component exists either because
the question requires it, or because a specific confound requires controlling.

```text
                      stream event
                           |
              +------------+------------+
              |                         |
           WRITE                      QUERY
              |                         |
              v                         v
   features.extract  (prefix-only)   SubstrateBank.recall
              |                         |
              v                         v
        Router.act  -> action        utility in [0,1]
              |                         |
              v                         |
        Ledger.can_afford               |
       (hard write ceiling)             |
              |                         |
              v                         |
   +----------+----------+----------+   |
   |          |          |          |   |
 IGNORE   EPISODIC     FAST       SLOW  |
   |          |          |          |   |
   |     exact store  decaying   durable|
   |     bounded cap  parametric parametric
   |          |          |          |   |
   +----------+----+-----+----------+   |
                   |                    |
              SubstrateBank <-----------+
                   |
                   v
        evaluator-only audit probes (no side effects, no budget)
```

## Components

### `world.py` — SDW-1, the benchmark

Generates one lifetime of a stream whose items belong to four hidden classes
with different ideal storage depths. The hidden class is never observable at
decision time. See [`BENCHMARK.md`](BENCHMARK.md).

### `substrates.py` — the three depths

| Action | Mechanism | Advantage | Price |
|---|---|---|---|
| `IGNORE` | nothing | free | no future recall |
| `EPISODIC` | exact key→value store, capacity `C`, LRU | exact recall | footprint grows per item; eviction; stale entries answer *confidently wrong* |
| `FAST` | delta-rule associative matrix, decay `γ<1` | fixed footprint, many associations, self-clearing | decays; interferes with other parametric content |
| `SLOW` | delta-rule associative matrix, no decay, `k×` consolidation steps | durable | expensive writes; interference is **permanent** — a wrong slow write is lasting damage |

The parametric substrates have a *constant* footprint and an effective capacity
near `key_dim`; the episodic store has a *growing* footprint and a hard item
capacity. That asymmetry is the whole trade-off, and it is what makes each
action uniquely useful for some class of item.

### `features.py` — the leakage firewall

The single path by which anything reaches a router. Fifteen whitelisted
features, each a function of the stream prefix and the agent's own state. See
[`LEAKAGE.md`](LEAKAGE.md).

### `routers.py` — policies

Controls, the fixed heuristic, the budget-matched random control, the oracle
ceiling, the privileged task-ID probe, and the learned policy. Each declares
`legal` and `privileged_fields`; the rollout engine hands privileged information
*only* to routers that declare it. See [`BASELINES.md`](BASELINES.md).

### `ledger.py` — resource accounting

One cost table, one hard ceiling, shared by every arm. See
[`RESOURCE-NORMALIZATION.md`](RESOURCE-NORMALIZATION.md).

### `agent.py` — rollout engine

Enforces three invariants:

1. privileged fields reach only routers that declare them;
2. evaluator audit probes use `SubstrateBank.probe`, which has **no side
   effects** — using the ordinary read path would refresh episodic LRU order and
   let the evaluator silently change the behaviour it is measuring;
3. the write ceiling is hard: an unaffordable action is downgraded to `IGNORE`
   and the downgrade is counted, rather than the budget being exceeded.

### `train.py` — offline credit assignment

REINFORCE with a blended advantage: per-decision attributed future utility (low
variance, blind to interference elsewhere) mixed with the lifetime objective
(captures interference, high variance).

**The distinction that matters.** The *policy* sees only prefix features at
decision time. The *trainer* sees outcomes that occur after a decision, because
"learn where writing pays off" has no other meaning. That is ordinary offline
policy-gradient training, not leakage — but the two must never be confused, so
`tests/test_leakage.py` asserts the separation mechanically.

## Deliberate exclusions

*   **`UPDATE_LATENT`** — persistent latent state is the sibling
    `state-promotion` track's independent variable. Including it here would
    confound the two tracks.
*   **`UPDATE_EXISTING_MODULE_i` / `SPAWN_NEW_MODULE`** — these add capacity,
    which is the most dangerous confound for this claim, and they place the work
    inside the well-developed modular-CL literature.
*   **Any language model** — EXP-001 is a synthetic toy. See the gate in the
    README.

## Amendment log

Pre-result design changes, recorded before any confirmatory result:

*   **2026-09-02, Amendment A — the "oracle" was not an upper bound.** The first
    calibration sweep used a hand-asserted class→action mapping as ORACLE. The
    fixed heuristic beat it in *every* configuration, which showed the assumed
    mapping was simply wrong, not that the heuristic was superhuman. ORACLE is
    now defined as the argmax over all `4^4` class-conditional mappings, found
    by search on development seeds. An assumption presented as a ceiling is a
    bug, and it would have inflated any later claim of "near-oracle" routing.

*   **2026-09-02, Amendment B — the designer's intended mapping was falsified.**
    The benchmark was designed expecting
    `ONE_OFF→EPISODIC, LOCAL→FAST`. Exhaustive search on development seeds
    found the optimum to be `ONE_OFF→FAST, LOCAL→EPISODIC` instead: episodic
    capacity cannot hold the local key population, while the fast substrate
    retains a transient trace comfortably over a one-off's short query delay.
    The intended mapping is retained in code as `INTENDED_MAPPING` and is
    reported alongside the true optimum, because the divergence is evidence
    that the ground truth was *discovered* rather than planted. See
    [`BENCHMARK.md`](BENCHMARK.md) §5.

*   **2026-09-02, Amendment C — bijectivity became an admissibility criterion.**
    Early configurations had non-bijective optima (two classes sharing one
    optimal action), which would mean the declared four-action space was not
    justified by the benchmark. Admissibility now requires that every action be
    uniquely optimal for exactly one class.
