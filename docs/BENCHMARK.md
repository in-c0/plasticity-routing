# SDW-1 — Stratified Depth World, version 1

A synthetic continual stream in which different observations genuinely deserve
different storage depths, and in which the hidden class that determines the
ideal depth is **never observable at decision time**.

## 1. Why synthetic, and why this one

The claim is about *allocation*, and allocation effects are easy to fake with
capacity or compute. A synthetic world lets every resource be counted exactly,
lets an upper bound be computed by exhaustive search, and lets leakage be tested
information-theoretically. None of that is possible on a language-model stream,
which is why EXP-001 is a toy and EXP-002 is gated.

## 2. Hidden classes

| Class | Generative behaviour | Queried |
|---|---|---|
| `NOISE` | fresh key, random value | never |
| `ONE_OFF` | fresh key, random value | exactly once, 4–60 steps later |
| `LOCAL` | key from a reused pool; value **redefined every regime** | repeatedly, within its regime only |
| `STABLE` | key from a pool; value fixed for the lifetime | repeatedly, up to 2200 steps later |

`NOISE` is never queried, so any capacity or interference it consumes is pure
waste — this is what makes `IGNORE` a necessary action rather than a decorative
one. `LOCAL` values are **revised** across regimes, so a durable write of a
local rule is not merely wasteful but actively wrong later.

## 3. Design contract

Enforced by `tests/test_world.py`:

1. **First encounters are observationally ambiguous.** Key and value vectors are
   drawn i.i.d. from the same distribution for every class. Tested by a
   permutation test on per-class mean vectors, not by a closed-form tolerance.
2. **Arrival time carries no class information.** Recurrent keys are introduced
   at times drawn uniformly over the *whole* lifetime and force-presented at
   their introduction time, and each class is emitted by an independent
   Bernoulli draw per step. Both are corrections; see §6.
3. **Only recurrence, value revision, and past query traffic separate the
   classes** — all prefix functions, all legal.
4. `NOISE` and `ONE_OFF` keys never repeat; `LOCAL` and `STABLE` keys do.
5. `STABLE` values never change; `LOCAL` values do.
6. `STABLE` keys are probed hundreds of steps after their write, so durability
   is actually tested.

Consequence: on a first encounter, no legal policy can do better than the
class prior. A router that appears to know the class on first sight is leaking,
which is leakage test L3.

## 4. Substrate trade-off

| | Episodic | Fast | Slow |
|---|---|---|---|
| fidelity | exact | blurred by interference | blurred by interference |
| footprint | grows: `key_dim + value_dim` per entry | constant `key_dim × value_dim` | constant |
| effective capacity | hard item limit, LRU eviction | ~`key_dim` associations | ~`key_dim` associations |
| read cost | similarity search, `|E| × key_dim` | constant | constant |
| persistence | until evicted | decays at `γ = 0.997` | permanent |
| write cost | `key_dim + value_dim` | `key_dim × value_dim` | `× 4` for consolidation |
| failure mode | eviction thrash; stale entries answer confidently wrong | decay; interference | **permanent** corruption |

An episodic entry must store its key or it could not be retrieved, and reading
the store is a search over it. Charging otherwise makes a large exact store look
free and lets "just make the episodic store bigger" win trivially — which is
the capacity confound this track exists to avoid.

## 5. Admissibility, and how the ground truth was found

The ideal mapping is **not asserted by the designer**. It is found by exhaustive
search over all `4^4 = 256` class-conditional mappings on development seeds
11/12/13 (`routers.search_best_mapping`). A configuration is admissible only if:

| | Criterion |
|---|---|
| C1 | ORACLE objective ≥ 0.45 — the world is solvable |
| C2 | no arm exceeds 0.95 task utility — not trivially easy |
| C3 | ORACLE − best single-depth control ≥ 0.06 — depth genuinely stratifies |
| C4 | ORACLE − heuristic ≥ 0.06 — the benchmark can discriminate a learned policy from a fixed rule |
| C5 | the optimal mapping is a **bijection** — every one of the four actions is uniquely best for exactly one class, so the declared action space is justified |
| C6 | the optimal mapping is stable in a local neighbourhood of the nuisance parameters |

C6 is deliberately *local*. A large change to a resource — say 60% more episodic
capacity — is expected to shift the optimum, because resource-dependence of the
right allocation is the whole thesis. Only knife-edge sensitivity is
disqualifying.

**Measured (`scripts/sensitivity.py`, dev seeds 11/12, exhaustive search at each
point).** The optimum `[IGNORE, FAST, EPISODIC, SLOW]` holds at **15 of 16**
neighbourhood points:

| axis | values tested | optimum unchanged |
|---|---|---|
| `episodic_capacity` | 20, 22, 24, 26, 28 | yes (all) |
| `fast_decay` | 0.995, 0.996, 0.997, 0.998, 0.999 | yes except **0.999** |
| `slow_lr` | 0.5, 0.7, 0.9 | yes (all) |
| `n_local_slots` | 40, 44, 48 | yes (all) |

The single boundary is `fast_decay = 0.999`, where the optimum becomes
`[IGNORE, IGNORE, EPISODIC, FAST]` and is no longer a bijection. This is not
fragility of the benchmark but the point at which the *fast* substrate stops
being a distinct depth: at `γ = 0.999` a trace retains `0.999^300 ≈ 0.74` across
a whole regime, so "fast" and "slow" cease to differ in persistence and one of
the four actions becomes redundant. The frozen value `0.997` sits two grid steps
away from that boundary, which is the required margin.

### The frozen configuration

`key_dim=96`, `value_dim=16`, `lifetime=3000`, `regime_len=300`,
`n_stable_keys=70`, `n_local_slots=44`, class prior `(0.26, 0.26, 0.28, 0.20)`,
`one_off_delay ∈ [4,60]`, `stable_query_horizon=2200`,
`episodic_capacity=24`, `fast_lr=1.0`, `fast_decay=0.997`, `slow_lr=0.7`,
`slow_consolidation_steps=4`, `write_element_ceiling=3_000_000`.

**Optimal mapping:** `NOISE→IGNORE, ONE_OFF→FAST, LOCAL→EPISODIC, STABLE→SLOW`.

**This is not the mapping the benchmark was designed around.** The design
hypothesis was `ONE_OFF→EPISODIC, LOCAL→FAST`; search placed that mapping 9th
of 256. The optimum swaps them: 44 local keys cannot fit a 24-entry episodic
store... but they are the most-queried class, so they win the store anyway,
while a one-off needs only to survive a short delay, which the decaying
parametric substrate does at no capacity cost.

The divergence is retained and reported rather than engineered away. It is the
evidence that the ground truth was *discovered by search* rather than planted by
the researcher, which is the single strongest defence against the charge that
the world was built around its answer.

## 6. Corrections made during calibration

All pre-result, all on development seeds, all logged in
[`ARCHITECTURE.md`](ARCHITECTURE.md#amendment-log):

*   **The "oracle" was an assumption, not a bound.** The first sweep used a
    hand-asserted mapping; the fixed heuristic beat it in every configuration.
    ORACLE is now derived by search.
*   **The designed mapping was falsified** (above).
*   **Bijectivity became an admissibility criterion** after early configurations
    produced optima in which two classes shared one action, which would have
    meant the four-action space was not justified.
*   **Arrival time leaked class.** With small recurrent pools sampled uniformly,
    every recurrent key first appeared early, so "novel key seen late" implied
    single-use. Leakage test L3 caught this on an *untrained* policy. Fixed by
    uniform introduction times over the whole lifetime plus independent
    per-class Bernoulli emission.
*   **A heuristic hyperparameter was dead.** `HeuristicRouter.revision_tolerance`
    was declared and swept, but the router read the pre-thresholded
    `value_revised` feature instead, whose cut is fixed in `features.py`. The
    calibration grid was therefore effectively 9 points, not 27 — silently
    under-tuning the *primary comparator*, which is the direction that flatters
    the method under test. Caught because every setting of the parameter scored
    identically. `tests/test_routers.py` now asserts that every declared
    threshold changes behaviour.
*   **Episodic cost was understated.** Entries did not store their keys and
    retrieval was charged O(|E|). Under that model a capacity-matched episodic
    control beat the oracle, because exact storage had no space or search
    penalty. Corrected to `key_dim + value_dim` per entry and `|E| × key_dim`
    per read.

The last two were found by the repository's own tests and controls rather than
by inspection of results, which is the intended function of both.
