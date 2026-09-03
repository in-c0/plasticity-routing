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

*   **2026-09-02, Amendment H — L5 failed, and the budget-matched random control
    does not mean what EXP-000 said it meant.** Implementing the time-shuffled
    control showed that the learned router's advantage over `RANDOM_MATCHED`
    (+0.111) is *fully reproduced* in a world where future utility has been
    randomised (+0.126, ratio 1.14 against a 0.25 threshold). In the shuffled
    world the policy adopts a completely different profile — 72% `IGNORE` +
    28% `SLOW`, versus 57% `EPISODIC` + 43% `FAST` in the real world — and still
    beats its own matched control.

    The conclusion is that `RANDOM_MATCHED` matches only the *marginal* action
    distribution, so any policy with conditional resource sense beats it whether
    or not it has learned anything about which items deserve which depth.
    EXP-000's reading of K2 ("allocation content does matter") was therefore
    wrong, and is corrected.

    K2 is redefined around the **utility-attributable advantage**

        A_util = [obj(LEARNED) − obj(RANDOM_MATCHED)]_real
               − [obj(LEARNED) − obj(RANDOM_MATCHED)]_shuffled

    which was **−0.015** at the configuration that produced EXP-000: essentially
    zero. Both worlds train under an identical procedure and budget, so the
    difference isolates the part of the advantage that depends on future utility
    being predictable at all.

*   **2026-09-02, Amendment I — the ES budget question, settled once.**
    `scripts/es_budget_study.py` ran 200 generations at two policy seeds. Under
    the rule fixed before running (no +0.004 improvement over a 40-generation
    window), the mean running-best development objective plateaus at
    **generation 86**; 60 → 100 generations is worth only +0.001, and 60 → 200
    only +0.015. The budget is frozen at **100 generations** — a modest increase
    with margin over the plateau, justified by convergence evidence rather than
    by outcome.

    The study also found the thing that actually mattered: **policy-seed
    variance dwarfs the budget.** At 60 generations three initialisations
    spanned 0.386 / 0.469 / 0.502 on the same development seeds, against a
    heuristic at 0.405. EXP-000 reported policy seed 0 — the worst of the three
    — so its K1 result was substantially an artefact of a single initialisation.
    `POLICY_SEEDS = (0, 1, 2)` is now frozen, with the carried-forward policy
    selected by **development** objective (`scripts/select_policy.py`).
    Selecting by confirmatory performance would be seed-shopping and is
    prohibited.

*   **2026-09-02, Amendment J — the comparator gets a matched search budget.**
    Selecting a policy seed on development data is only fair if the comparator
    is searched as hard. The learned router consumes
    `3 seeds × 100 generations × 51 rollouts = 15,300` development rollouts; the
    three-parameter heuristic grid consumed 81. `ExtendedHeuristicRouter` adds
    an explicit durable-write rule (recurrence and query-evidence conditions), a
    budget guard, and configurable defaults — eight knobs, all decision-time
    legal — and `scripts/search_heuristic_matched.py` searches it with the
    **same 15,300 rollouts**. The primary comparator for H1 is the best legal
    non-learned router found under that matched budget.

*   **2026-09-02, Amendment K — an initialisation override was a silent no-op.**
    The first durable-write discoverability diagnostic returned
    "SLOW genuinely unhelpful", but `slow_primed` and `cloned` had produced
    traces *identical* to the default initialisation. `train_router_es`
    snapshotted the parameter vector ES optimises **before** applying
    `init_bias` / `init_params`, so both overrides were discarded on the first
    perturbation. The diagnostic's verdict was void. Fixed, and
    `tests/test_routers.py` now asserts that each override changes the
    generation-zero objective.

    The random-initialisation arms of that run were unaffected and are
    informative on their own: policy seed 2 reached 0.5015 while selecting
    `SLOW` 13.9% of the time, close to the heuristic's 13.2%. So the durable
    write is **reachable** under the frozen budget; seed 0 simply failed to find
    it.

*   **2026-09-03, Amendment L — L5a stays failed; L5b is a stronger null.**
    (Letter `L`: `I` was already used for the ES-budget decision above, and
    reusing it would corrupt this log.)

    **L5a is retained permanently, unchanged, as FAILED** — ratio 0.521 against
    a preregistered threshold of 0.25. The threshold is **not** relaxed and the
    test is not deleted, rewritten, or reinterpreted.

    What L5a falsified is worth stating precisely, because it is easy to
    misread. It did **not** show that the learned router is exploiting a leak.
    It showed that **`RANDOM_MATCHED` is an inadequate attribution control**:
    it preserves only the marginal `P(A)`, whereas the learned router can
    exploit `P(A | X)` for generic resource management that has nothing to do
    with future utility. A control that only matches action frequencies cannot
    separate "knows which items deserve which depth" from "knows when it is
    worth writing at all". So L5a is a finding about the control, not about the
    hypothesis — and it is exactly why the confirmatory run was blocked.

    **L5b — cross-world utility-shuffle negative control.** Let `R` be the
    policy trained on the real development worlds and `S` the *identically
    specified* policy trained on the time-shuffled development worlds: same
    `LearnedRouter`, same legal feature whitelist, same ES budget, same three
    policy seeds, same selection rule. The single difference is whether training
    preserved the prefix→future-utility relationship. This follows the standard
    negative-control principle: the null retains every nuisance mechanism we
    want to control for and removes only the hypothesised informative
    relationship.

    Evaluated as a 2x2 cross on a **fresh one-shot audit seed set**:

        J_RR = J(R, real)      J_SR = J(S, real)
        J_RS = J(R, shuffled)  J_SS = J(S, shuffled)

    Primary quantity, and the crossover interaction:

        Delta_real = J_RR - J_SR
        I          = (J_RR - J_SR) - (J_RS - J_SS)

    `Delta_real` asks the right question: on the identical real environment,
    does training with genuine future-utility structure buy anything over an
    otherwise identical router trained when that structure was destroyed? A
    positive `I` says the difference is specifically tied to *matching* the
    policy's training utility structure to the evaluation world's, rather than
    `R` simply being a universally better network.

    **Gate:** paired bootstrap at the audit-seed level; the 95% CI for **both**
    `Delta_real` and `I` must exclude zero on the positive side. No new ratio,
    and no minimum effect size chosen after seeing data — zero is the natural
    null.

    **Why not evaluate the cross on the development seeds.** `R` was selected
    for real-dev performance and `S` for shuffled-dev performance, so a
    development-seed cross is selection-biased in `R`'s favour. `AUDIT_SEEDS =
    91001..91032` is frozen here, before any cross-world number is inspected,
    and is neither a training nor a confirmatory set.

    **Explicitly rejected alternatives.** (i) A router fitted to reproduce
    `P(A | X)` from the real learned policy — that would clone precisely the
    item-specific conditional structure under test. (ii) Partitioning the
    fifteen features into "resource" and "utility" subsets — the whitelist mixes
    key history, prediction error, memory occupancy, budget and time, and
    choosing that partition *after* seeing L5a would be a fresh researcher
    degree of freedom.

    **Consequences, fixed in advance.** If L5b fails, EXP-001 stops before
    confirmatory execution and the negative result is published: adaptive
    routing may beat marginal random allocation, but the experiment cannot
    establish that the advantage arises from learning delayed future utility.
    **No L5c will be designed.** If L5b passes, L5a is retained as a failed
    historical diagnostic, L5b becomes the attribution validity gate, the full
    leakage suite is re-run, and only then may protocol v1.0 be frozen and the
    confirmatory seeds touched.

    In the confirmatory experiment `SHUFFLE_TRAINED` (= `S`) becomes a real arm
    and **K2 becomes the paired contrast `LEARNED - SHUFFLE_TRAINED > 0`** with
    a CI excluding zero. `RANDOM_MATCHED` and the `A_util` decomposition are
    retained but **demoted to secondary diagnostics**; they no longer carry the
    causal-attribution claim.

