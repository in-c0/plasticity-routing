# EXP-001 — confirmatory result

**Classification: `CONFIRMATORY`. One shot. Protocol v1.2.**

Executed 2026-09-03 on the five held-out seeds
`{20260902, 20260903, 20260904, 20260905, 20260906}`, taken from the frozen
lock. The runner has no seed CLI. Both policy checkpoints and the comparator
artefact were verified against the hashes pinned in `config.py` before any
rollout ran. Manifests: `results/confirmatory/` (65), independently re-validated
against `protocol_v1.2_lock.json` — **VALIDATION PASSED**, no invalidation
reasons.

**H1 is supported.**

---

## 1. Gates, in preregistered order

Comparative metrics were computed but withheld until all of these passed. The
runner stops before the headline comparison if any invalidating gate fires.

| gate | outcome |
|---|---|
| L1 feature whitelist purity | PASS |
| L2 hidden-label permutation invariance | PASS |
| L3 first-encounter class independence (+ leak canary) | PASS |
| L4 causal ordering / future blindness | PASS |
| L6 privilege declaration | PASS |
| L7 audit probes have no side effects | PASS |
| **L5a** marginal-random time-shuffled control | **RETAINED FAILED, non-gating** |
| **L5b** cross-world utility-shuffle negative control | **PASS** |
| manifest validation against the frozen lock | PASS |
| write ceilings / resource accounting | PASS (no arm exceeded; max 100.0%) |
| C1 ORACLE ≥ 0.45 | 0.5480 PASS |
| C2 max task utility ≤ 0.95 | 0.8970 PASS |
| C3 ORACLE − best single depth ≥ 0.06 | 0.3216 PASS |
| C4 ORACLE − HEURISTIC_EXT ≥ 0.06 | 0.1327 PASS |
| C5 optimal mapping bijective and equal to the frozen one | `[IGNORE, FAST, EPISODIC, SLOW]` PASS |
| C6 local stability of the optimum | inherited from the frozen calibration (15/16 neighbourhood points); **not recomputed** on confirmatory seeds |
| K3, K4, K5, K6, K8 | none fired |

## 2. Arms

Mean over the five confirmatory seeds.

| arm | legal | objective | utility | forgetting | IGNORE | EPIS | FAST | SLOW |
|---|---|---|---|---|---|---|---|---|
| `ORACLE` | **no** | 0.5480 | 0.897 | 0.254 | 0.284 | 0.270 | 0.227 | 0.219 |
| **`LEARNED`** | yes | **0.5060** | 0.849 | 0.282 | 0.184 | 0.232 | 0.503 | 0.081 |
| `HEURISTIC_EXT` | yes | 0.4153 | 0.801 | 0.327 | 0.223 | 0.019 | 0.653 | 0.105 |
| `DESIGNED_MAPPING` | **no** | 0.4133 | 0.820 | 0.320 | 0.307 | 0.223 | 0.258 | 0.212 |
| `HEURISTIC` (3-param) | yes | 0.4070 | 0.802 | 0.393 | 0.234 | 0.553 | 0.091 | 0.122 |
| `PRIVILEGED_TASKID` | **no** | 0.4068 | 0.802 | 0.393 | 0.233 | 0.553 | 0.091 | 0.122 |
| `CAPACITY_MATCHED_EPISODIC` | yes | 0.2483 | 0.715 | 0.651 | 0.000 | 1.000 | 0.000 | 0.000 |
| `RANDOM_MATCHED` | yes | 0.2274 | 0.684 | 0.487 | 0.188 | 0.227 | 0.501 | 0.085 |
| `ALL_FAST` | yes | 0.2264 | 0.685 | 0.573 | 0.000 | 0.000 | 1.000 | 0.000 |
| `ALL_SLOW` | yes | 0.0269 | 0.365 | 0.312 | 0.720 | 0.000 | 0.000 | 0.280 |
| `SHUFFLE_TRAINED` | yes | 0.0167 | 0.249 | 0.185 | 0.844 | 0.000 | 0.000 | 0.156 |
| `ALL_EPISODIC` | yes | −0.0051 | 0.478 | 0.679 | 0.000 | 1.000 | 0.000 | 0.000 |
| `ALL_IGNORE` | yes | −0.0827 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |

## 3. Preregistered contrasts

Paired bootstrap over the five seeds, 20,000 resamples.

| contrast | estimate | 95% CI | excludes 0 |
|---|---|---|---|
| **K1** `LEARNED − HEURISTIC_EXT` | **+0.0907** | [+0.0798, +0.1016] | yes |
| **K2** `LEARNED − SHUFFLE_TRAINED` | **+0.4893** | [+0.4553, +0.5232] | yes |
| **K7** `LEARNED − CAPACITY_MATCHED` | +0.2577 | [+0.2430, +0.2753] | yes |
| **K9** `PRIVILEGED_TASKID − HEURISTIC` | −0.0003 | [−0.0008, −0.0000] | yes (negative) |
| `LEARNED − RANDOM_MATCHED` (secondary) | +0.2786 | [+0.2551, +0.3057] | yes |
| `ORACLE − LEARNED` (headroom) | +0.0420 | [+0.0335, +0.0542] | yes |

K1 is positive on **5/5 individual seeds** (+0.0755 … +0.1066).

### H1 success rule

| criterion | outcome |
|---|---|
| K1 `LEARNED > HEURISTIC_EXT`, CI excludes 0 | PASS |
| K2 `LEARNED > SHUFFLE_TRAINED`, CI excludes 0 | PASS |
| K7 survives capacity matching | PASS |
| K9 privileged task ID explains less than learning | PASS (it explains nothing: −0.0003) |
| K10 cross-world attribution (L5b) | PASS |

**H1: SUPPORTED.**

## 4. What was predicted and what held

The development-seed diagnostic put K1 at +0.0819. The confirmatory value is
**+0.0907** — slightly larger on seeds the policy never saw, which is not what
an overfit policy looks like.

The mechanism is the same one visible on development seeds. `LEARNED` buys a
near-oracle `LOCAL` score and pays for it on `ONE_OFF`:

| arm | ONE_OFF | LOCAL | STABLE |
|---|---|---|---|
| `ORACLE` | 0.776 | 0.971 | 0.848 |
| `LEARNED` | 0.559 | **0.952** | 0.824 |
| `HEURISTIC_EXT` | 0.786 | 0.769 | 0.844 |

It reaches within 0.042 of the class-conditional oracle ceiling while using all
four actions, and it does so under the same cost table and the same hard write
ceiling as every other arm (76.9% of budget, versus 98.7% for the comparator).

## 5. Honest limitations

*   **One synthetic benchmark, five seeds.** SDW-1 is a toy with linear
    associative substrates. "Interference" here is subspace overlap, not
    anything a transformer does. No language-model claim is made or implied;
    EXP-002 remains blocked.
*   **K2's margin is inflated by a weak absolute control.** `SHUFFLE_TRAINED`
    scores 0.0167, near the floor, so +0.489 overstates how informative that
    single number is. The sharper evidence is the **L5b crossover**, where the
    same policy `S` *beats* `R` in the shuffled world (+0.018 vs −0.090). That
    is what shows `S` is a competent policy for the world it was trained on
    rather than a broken one, and it is why the attribution holds.
*   **C6 was inherited, not recomputed.** Local stability of the optimal mapping
    was established during frozen calibration across 16 neighbourhood points; it
    was not re-swept on confirmatory seeds. C5 *was* recomputed here by
    exhaustive search over all 256 mappings and recovered the frozen optimum.
*   **The comparator is the best fixed rule found under a matched search
    budget**, not the best fixed rule that exists. A better hand-designed
    router may exist outside the searched family.
*   **The policy was selected on development data** over three seeds
    (spread 0.133). That selection is legitimate — it never touched confirmatory
    seeds — but the comparator's matched 15,300-rollout search is what makes it
    fair, and a reader should weigh both together.
*   `ORACLE` and `DESIGNED_MAPPING` read the hidden class and are ceilings only.
    `PRIVILEGED_TASKID` reads the regime id. None is claim-eligible.

## 6. Execution provenance

*   Protocol **v1.2**, equivalent to v1.1 and transitively to v1.0 on every
    scientific field (config hash, all three seed lists, world/substrate/cost
    configuration, ES configuration, heuristic parameters, both mappings). v1.0
    and v1.1 are preserved verbatim.
*   **One aborted attempt is recorded**, not hidden. Attempt 01 stopped at gate
    2 because the validator resolved a hard-coded v1.0 lock path and rejected
    v1.1 manifests. No comparative metric was computed, printed or aggregated —
    gates 3–5 were never reached and no contrast was produced. Its manifests are
    retained under `results/confirmatory-aborted-01/` with a note. The fix
    touched lock resolution only and moved no scientific quantity.

## 7. Claim

On SDW-1, under matched write, storage and compute budgets, a policy trained
only on decision-time-legal features learns a write-depth allocation that beats
the best fixed rule found under an equal search budget, by +0.0907
[+0.0798, +0.1016]. The advantage is attributable to having learned from a
genuine prefix→future-utility relationship (K2, and the L5b crossover), not to
capacity (K7), not to privileged context identity (K9), and not to budget
allocation alone (`RANDOM_MATCHED` secondary contrast).

This is a claim about a synthetic benchmark. It is not a claim about language
models, and it is not a claim that routing to experts or using multiple memory
systems is novel.
