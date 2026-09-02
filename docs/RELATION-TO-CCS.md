# Relation to CCS

The emerging CCS decomposition is:

*   **ACCUMULATE** — continue gathering / thinking
*   **ALLOCATE** — choose which substrate should change
*   **COMMIT** — perform the internal/external commitment

**This repository is about ALLOCATE, and only ALLOCATE.**

## Scope boundary

EXP-001 must not be broadened into a full CCS demonstration. Concretely:

*   No ACCUMULATE decision is modelled. The agent does not choose whether to
    keep thinking; the stream arrives and each write event demands an
    allocation decision.
*   No COMMIT machinery is modelled. There is no external action, no tool call,
    no irreversible outward-facing commitment. The `SLOW` action is a durable
    *internal* write, not a commitment in the CCS sense.
*   No latent-state accumulation is modelled — `UPDATE_LATENT` is excluded, and
    persistent latent state is the sibling track's variable.

## Sibling tracks

| Repo | CCS stage | Question |
|---|---|---|
| `in-c0/state-promotion` | ALLOCATE→COMMIT boundary | Should information be promoted from fast to slow state, gated on evidence rather than a schedule? |
| `in-c0/adaptive-commitment` | COMMIT | (not modified by this track) |
| `in-c0/plasticity-routing` | **ALLOCATE** | Where should learning go at all — including nowhere? |

State Promotion fixes the pathway (fast → slow) and learns *when* to move along
it. This track asks *which pathway* an experience should take, including the
option of no write. The two are complementary and deliberately not merged: if
allocation and promotion timing varied in one experiment, neither effect would
be attributable.

Neither `state-promotion` nor `adaptive-commitment` is modified by this work.

## Dependency gate

This track may proceed immediately with literature review, formulation,
benchmark design, preregistration, simulator infrastructure, and baselines that
do not depend on State Promotion outcomes — all of which are complete or in
progress here.

**A confirmatory language-model experiment in this repository is blocked** until
the first mechanically valid State Promotion engineering pilot has been
reviewed. The reason is specific rather than procedural: State Promotion's
pilot is the first place a mechanical defect in the shared two-timescale
formulation would surface — write-budget normalisation, optimizer-state
persistence across segment boundaries, latent-architecture matching between
routing controls, and decision-time routing compute accounting are all shared
concerns. EXP-001 here must not unknowingly bake in a defect that pilot
discovers.

EXP-001 as specified is a **synthetic toy** and is therefore not blocked by that
gate. EXP-002 (any language model) is blocked. If State Promotion produces
information relevant to this experiment, it is incorporated transparently as a
pre-result design amendment, logged in
[`ARCHITECTURE.md`](ARCHITECTURE.md#amendment-log), **before** this track
reaches confirmatory execution.
