# results/

**Everything here is classified `DEV_CALIBRATION`. None of it is evidence.**

The learned router was trained *and* evaluated on development seeds
`{11, 12, 13}`. Confirmatory seeds have not been touched. See
[`../experiments/EXP-000-RESULT.md`](../experiments/EXP-000-RESULT.md) for the
write-up and [`../experiments/EXP-001-PREREG.md`](../experiments/EXP-001-PREREG.md)
for what a confirmatory run would require.

| File | Contents |
|---|---|
| `run_<ARM>_seed<N>.json` | per-arm, per-seed manifest: source-tree hash, config hash, environment, all resource totals, metrics, diagnostics, embedded leakage verdict |
| `exp000_summary.json` | aggregated arm summaries, preregistered contrasts, ES training history |
| `validation.json` | `scripts/validate_runs.py` verdict over the manifest set |
| `leakage_audit.json` | runtime leakage audit L1–L7 |
| `heuristic_calibration.json` | predeclared heuristic grid search on development seeds |
| `sensitivity.json` | neighbourhood stability of the optimal class→action mapping (criterion C6) |

Manifests are committed deliberately: a result that cannot be reproduced from
its manifest does not count, so the manifests are part of the deliverable rather
than a build artefact.

`calibration.json` from the initial `scripts/calibrate_world.py` sweep is **not**
retained, because it was produced under a superseded episodic cost model
(entries did not store keys; retrieval was charged `O(|E|)`). Keeping it would
invite comparison against numbers that no longer mean the same thing. The
provenance of the frozen configuration is documented in
[`../docs/BENCHMARK.md`](../docs/BENCHMARK.md) §5–6, and `sensitivity.json`
re-derives the optimal mapping under the corrected model.
