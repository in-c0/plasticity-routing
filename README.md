# Adaptive Plasticity Routing

**Working research project:** learned, resource-aware allocation of new
experience across substrates of differing persistence — the **ALLOCATE** stage
of CCS.

> Status: **pre-result / experiment scaffold.** No empirical claim is made yet.
> Everything in `results/` is classified `DEV_CALIBRATION` and is not evidence.

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
5. **The first trainer could not see most of the objective** — per-decision
   credit assignment misses resource cost by ~4 orders of magnitude, misses
   interference entirely, and cannot see budget exhaustion. Its development
   objective was flat across 60 epochs. Replaced by evolution strategies on the
   preregistered objective; the failed implementation is retained.

Items 3, 4 and 5 were found by this repository's own tests and controls rather
than by inspection of results, which is what they are for.

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

*   **EXP-000 — development calibration.** Harness smoke test on SDW-1. Trains
    and evaluates on development seeds, so it is **not evidence**. See
    [`experiments/EXP-000-RESULT.md`](experiments/EXP-000-RESULT.md).
*   **EXP-001 — confirmatory synthetic.** Train on development seeds, evaluate
    on five held-out confirmatory seeds. Gated on issue #1. Preregistered in
    [`experiments/EXP-001-PREREG.md`](experiments/EXP-001-PREREG.md).
*   **EXP-002 — language model. BLOCKED.** Must not run until the first
    mechanically valid `in-c0/state-promotion` engineering pilot has been
    reviewed, so that EXP-001 here does not bake in a defect that pilot
    discovers.

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
