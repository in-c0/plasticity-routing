# Aborted confirmatory attempt 01 — 2026-09-03

**Aborted at gate 2 (manifest validation). No comparative metric was computed,
printed, or aggregated.** Gates 3, 4 and 5 were never reached; no contrast, no
arm summary and no per-class table was produced. The run log stopped inside the
validator's failure list.

## Cause

`scripts/validate_runs.py` resolved the protocol lock from a hard-coded path,
`results/protocol_v1_lock.json` — the **v1.0** lock. The manifests were produced
under **v1.1**, whose source-tree hash necessarily differs (v1.1 adds the
confirmatory runner). Every manifest was therefore rejected with
`confirmatory_run_off_frozen_source_tree`.

This is an execution-layer defect in the validator, not a scientific finding.
The gate behaved correctly: it refused to certify manifests it could not match
to a frozen lock, and it did so *before* the reveal order permitted any
comparative metric to be inspected.

## Why re-running is legitimate

The one-shot rule exists to prevent inspecting confirmatory results and then
changing something. Nothing was inspected: the abort happened two gates before
any metric is computed. The fix touches only lock resolution inside the
validator and changes no world, objective, action space, architecture,
checkpoint, comparator, threshold, seed or statistic.

Because fixing the validator changes the source tree, the v1.1 lock became
stale and protocol **v1.2** was frozen, again asserting field-by-field
equivalence on every scientific field. v1.0 and v1.1 are preserved.

## Retained artefacts

The manifests from this attempt are kept here for provenance. They are
`CONFIRMATORY`-classified but **were never certified**, and no result was ever
read from them.
