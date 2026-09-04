# Adaptive Plasticity Routing

**Working research project:** learned, resource-aware allocation of new
experience across substrates of differing persistence — the **ALLOCATE** stage
of CCS.

> Status: **EXP-001 confirmatory complete.** H1 is supported on the five held-out
> seeds under protocol v1.2. See
> [`experiments/EXP-001-RESULT.md`](experiments/EXP-001-RESULT.md).
> This is a claim about a synthetic benchmark; no language-model claim is made,
> and EXP-002 remains blocked.

## The narrow claim

Not "where should learning go?", and not "can an agent route to experts". After
a literature audit ([`docs/LITERATURE.md`](docs/LITERATURE.md)), the defensible
question is:

> Can a policy learn, from **delayed future task utility under a unified
> resource budget**, to allocate each experience to a substrate of differing
> **persistence and interference character** — and is any resulting gain
> attributable to *routing*, rather than to added capacity, added compute, or
> access to privileged task identity?

**We do not claim** that routing to experts is new, that using multiple memory
systems is new, or that deciding whether to write to memory is new. All three
are well established; §1 of the literature audit names the work that owns them.

**The prior on the hypothesis is low.** Yoon (2026, arXiv:2606.30067) reports
that a simple similarity rule matches or beats a learned RL allocation
controller under fixed capacity. The fixed heuristic is therefore the *primary
comparator*, not a courtesy baseline, and a negative replication is a
preregistered publishable outcome.

## Action space

EXP-001 uses the smallest subset that keeps write *depth* a real axis:

```
IGNORE      no write, no storage, no interference
EPISODIC    exact entry; bounded capacity; evictable; search cost; footprint grows
FAST        parametric write that decays; cheap; interferes transiently
SLOW        parametric write that persists; expensive; interferes permanently
```

`UPDATE_LATENT` is excluded (the sibling track's variable);
`UPDATE_EXISTING_MODULE_i` and `SPAWN_NEW_MODULE` are excluded (they add
capacity — the primary confound — and sit inside the modular-CL literature).

Whether four actions are *justified* is not assumed but tested: the benchmark is
admissible only if the empirically optimal class→action mapping is a bijection.

## Benchmark

**SDW-1** ([`docs/BENCHMARK.md`](docs/BENCHMARK.md)) — a synthetic stream with
four hidden classes that genuinely deserve different depths:

| Class | Queried | Why one depth suits it |
|---|---|---|
| `NOISE` | never | any storage or interference is pure waste |
| `ONE_OFF` | once, soon | needs brief availability, not durability |
| `LOCAL` | often, within a regime; **value redefined each regime** | durable storage becomes actively wrong |
| `STABLE` | often, up to 2200 steps later | must survive decay and eviction |

The hidden class is never observable at decision time. On a first encounter the
classes are observationally identical by construction, so only recurrence, value
revision, and past query traffic — all prefix functions — can separate them.

**The optimal mapping was found by exhaustive search over all 256
class→action mappings, not asserted.** It is
`NOISE→IGNORE, ONE_OFF→FAST, LOCAL→EPISODIC, STABLE→SLOW` — which **contradicts
the design hypothesis** (`ONE_OFF→EPISODIC, LOCAL→FAST`, ranked 9th of 256).
That divergence is preserved rather than engineered away: it is the evidence
that the ground truth was discovered rather than planted.

## Objective

```
future task utility − forgetting penalty − storage cost − write cost − compute cost
```

Resource accounting is mandatory. One cost table and one hard write ceiling are
shared by every arm; router decision compute is charged on every decision and
reported separately. See
[`docs/RESOURCE-NORMALIZATION.md`](docs/RESOURCE-NORMALIZATION.md).

## Arms

Claim-eligible: `ALL_IGNORE`, `ALL_EPISODIC`, `ALL_FAST`, `ALL_SLOW`,
`HEURISTIC`, `RANDOM_MATCHED`, `LEARNED`.
Ceiling / privileged, never claim-eligible: `ORACLE`, `DESIGNED_MAPPING`,
`PRIVILEGED_TASKID`.
Disentanglement: `CAPACITY_MATCHED_EPISODIC` and a compute break-even analysis,
so that routing benefit, capacity benefit, compute benefit, and privileged
task-ID benefit are never conflated. See [`docs/BASELINES.md`](docs/BASELINES.md).

## Leakage

> The routing decision may use only information available at decision time.

Fifteen whitelisted features, all prefix functions, reaching a policy through
one small auditable module. Seven tests (L1–L7) including an
information-theoretic first-encounter independence test with a permutation null,
and a canary asserting the detector catches a deliberate oracle leak. A failing
leakage test invalidates the run. See [`docs/LEAKAGE.md`](docs/LEAKAGE.md).

## Preserved negative results

The repository keeps its own failures, because they constrain the design:

1. **The first "oracle" was an assumption, not a bound** — the fixed heuristic
   beat it in every configuration. ORACLE is now derived by exhaustive search.
2. **The designed class→action mapping was falsified** by that search.
3. **Arrival time leaked the hidden class** — caught by leakage test L3 on an
   *untrained* policy, before any result was inspected.
4. **The episodic cost model was wrong** — entries did not store their keys and
   retrieval was charged `O(|E|)`, so a capacity-matched exact store beat the
   oracle. Caught by the capacity-disentanglement control.
5. **A heuristic hyperparameter was dead** — declared and swept but never read,
   so the primary comparator's calibration grid was effectively a third of its
   stated size. Under-tuning the comparator flatters the proposed method, so
   this mattered: fixing it and re-calibrating moved the headline contrast
   from −0.010 (CI spanning zero) to −0.019 (CI excluding zero), *against* the
   proposed method.
6. **L5a fails, permanently and on purpose.** Over half the learned router's
   advantage over budget-matched random routing is reproduced in a world where
   future utility is pure noise (ratio 0.52 against a 0.25 threshold). The
   threshold was **not** relaxed and the test was not rewritten. What it
   falsified is the *control*: `RANDOM_MATCHED` preserves only the marginal
   `P(A)`, so conditional resource sense alone clears it. It is retained as a
   failed historical diagnostic and replaced — as the attribution gate — by the
   stricter cross-world control L5b.
7. **An ES initialisation override was a silent no-op**, voiding the first
   durable-write discoverability diagnostic.
8. **The comparator had been searched ~190× less hard** than the learned router
   (81 rollouts vs 15,300).
9. **The first trainer could not see most of the objective** — per-decision
   credit assignment misses resource cost by ~4 orders of magnitude, misses
   interference entirely, and cannot see budget exhaustion. Its development
   objective was flat across 60 epochs. Replaced by evolution strategies on the
   preregistered objective; the failed implementation is retained.

Items 3–9 were found by this repository's own tests, controls, and calibration
sweeps rather than by inspection of results, which is what they are for.

## Attribution: L5b

The attribution question — *is the advantage actually about learning future
utility?* — is settled by a cross-world negative control
([`experiments/L5B-RESULT.md`](experiments/L5B-RESULT.md)). Let `R` be the
policy trained on real worlds and `S` an **identically specified** policy
trained on time-shuffled worlds: same network, features, ES budget, policy seeds
and selection rule; the only difference is whether training preserved the
prefix→future-utility relationship. Evaluated once on 32 fresh audit seeds that
are neither training nor confirmatory:

| | evaluated on real | evaluated on shuffled |
|---|---|---|
| `R` (trained on real) | **+0.5130** | −0.0898 |
| `S` (trained on shuffled) | +0.0201 | +0.0183 |

`Delta_real = +0.4929` [+0.4822, +0.5032] and the crossover interaction
`I = +0.6011` [+0.5902, +0.6122]; both positive on 32/32 seeds. A true
crossover — `R` wins on real, `S` wins on shuffled — so `R` is not simply a
better network.

## Confirmatory result

Executed once on the five held-out seeds, protocol v1.2, every gate passed in
the preregistered order.

| contrast | estimate | 95% CI |
|---|---|---|
| **K1** `LEARNED − HEURISTIC_EXT` (matched search budget) | **+0.0907** | [+0.0798, +0.1016] |
| **K2** `LEARNED − SHUFFLE_TRAINED` (attribution) | +0.4893 | [+0.4553, +0.5232] |
| **K7** `LEARNED − CAPACITY_MATCHED` | +0.2577 | [+0.2430, +0.2753] |
| **K9** `PRIVILEGED_TASKID − HEURISTIC` | −0.0003 | [−0.0008, −0.0000] |
| `ORACLE − LEARNED` (headroom) | +0.0420 | [+0.0335, +0.0542] |

**H1 supported.** K1 is positive on 5/5 individual seeds, and the development
estimate (+0.0819) held on unseen seeds (+0.0907). `LEARNED` lands within 0.042
of the class-conditional oracle ceiling using all four actions.

The headline number is deliberately the *narrowest* one available: the
comparator is the best fixed rule found under a search budget matched to the
learned router's, rollout for rollout.

Full result, including limitations and the one aborted execution attempt:
[`experiments/EXP-001-RESULT.md`](experiments/EXP-001-RESULT.md).

## Layout

```
docs/     LITERATURE  ARCHITECTURE  BENCHMARK  BASELINES
          RESOURCE-NORMALIZATION  LEAKAGE  KILL-CRITERIA  RELATION-TO-CCS
experiments/  EXP-001-PREREG.md   EXP-000-RESULT.md
src/plasticity_routing/  world  substrates  features  routers  agent
                         ledger  train  metrics  manifest  config
scripts/  calibrate_world  sensitivity  audit_leakage  run_exp000  validate_runs
tests/    world  substrates  leakage  ledger  routers  metrics  manifest
```

## Running it

```bash
python3.12 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
```

```bash
make calib
```

`make calib` runs the test gate, the leakage audit, EXP-000 development
calibration, and manifest validation. Individual targets: `make test`,
`make leakage`, `make exp000`, `make validate`.

Pure NumPy, CPU only, no GitHub Actions.

## Experiment sequence

*   **EXP-001 — confirmatory synthetic. COMPLETE, H1 supported.** Five held-out
    seeds, protocol v1.2. Paper:
    [`paper/main.tex`](paper/main.tex). Result:
    [`experiments/EXP-001-RESULT.md`](experiments/EXP-001-RESULT.md).
*   **EXP-001R — replication on 32 untouched seeds. Preregistered, not run.**
    Frozen policy, comparator and configuration unchanged; raises the
    inferential sample from 5 to 32. See
    [`experiments/EXP-001R-PREREG.md`](experiments/EXP-001R-PREREG.md).
*   **EXP-002 — language model. Gate review SATISFIED; design in progress;
    execution NOT authorized.** Seven pre-result constraints from the State
    Promotion pilot ([`experiments/EXP-002-DESIGN.md`](experiments/EXP-002-DESIGN.md)),
    refined by Amendment N and turned into a gating preregistration
    ([`experiments/EXP-002-SUBSTRATE-SUFFICIENCY-PREREG.md`](experiments/EXP-002-SUBSTRATE-SUFFICIENCY-PREREG.md)).

    Frozen sequence: representation sufficiency → `FAST` sufficiency →
    `SLOW` sufficiency → resource/compute accounting sanity → routing benchmark
    admissibility → router/comparator design → protocol freeze → LM comparative
    runs. Stages 1–5 are arm-agnostic. **A router that never selects an action
    is only interesting if that action was actually available.**

### Historical

*   **EXP-000 — development calibration.** Harness smoke test on SDW-1. Trains
    and evaluates on development seeds, so it is **not evidence**. See
    [`experiments/EXP-000-RESULT.md`](experiments/EXP-000-RESULT.md).
    Preregistration: [`experiments/EXP-001-PREREG.md`](experiments/EXP-001-PREREG.md).

## Relation to CCS and sibling tracks

This repo is **ALLOCATE** only, and must not be broadened into a full CCS
demonstration. `state-promotion` fixes the pathway (fast → slow) and learns
*when* to move along it; this track asks *which pathway* an experience should
take, including no write at all. Neither `state-promotion` nor
`adaptive-commitment` is modified by this work. See
[`docs/RELATION-TO-CCS.md`](docs/RELATION-TO-CCS.md).

## Reproducibility policy

*   Preregister before inspecting confirmatory results.
*   Development and confirmatory seeds are disjoint and frozen in `config.py`.
*   Calibrate the benchmark using only controls and the oracle; never run the
    learned router during calibration.
*   Never weaken the primary comparator; calibrate it on the same budget as the
    proposed method.
*   Charge router decision compute, and report it separately from substrate
    compute.
*   Evaluator audit probes must have no side effects and no budget.
*   Publish negative results, including the repository's own design failures.
*   Every run carries a manifest pinning source-tree hash, config hash, seed,
    environment, all resource totals, and the leakage-audit verdict.

## Licence

Apache-2.0.
