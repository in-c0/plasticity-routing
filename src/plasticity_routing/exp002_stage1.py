"""EXP-002 Stage-1 freeze schema and validator (Amendment O-1).

Harness only. This module contains **no model, no task, and no training loop**;
it decides whether a Stage-1 configuration is complete enough to be executed
reproducibly, and refuses otherwise.

The problem it solves: the substrate-sufficiency preregistration froze its
*pass thresholds* but not the quantities those thresholds are applied to. Under
that state the selection rule "smallest sufficient capacity" is not a selection
rule, because the grid it selects over could still be chosen after seeing a
candidate result. Every field below must therefore be set before the first
Stage-1 result exists, and the validator refuses while any is missing.

Refusal, not discipline: `scripts/freeze_exp002_stage1.py` cannot emit an
artifact until `validate_stage1_freeze` returns no blockers.
"""

from __future__ import annotations

from typing import Any

#: Required fields, grouped as in EXP-002-SUBSTRATE-SUFFICIENCY-PREREG.md §13.
#: Data-driven so the tests can assert that *each* field independently causes a
#: refusal, rather than trusting that the validator checks them all.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "backbone": (
        "repo_id", "revision_sha", "tokenizer_revision",
        "tokenizer_asset_sha256", "dtype", "device",
    ),
    "task": (
        "generator_module", "generator_source_sha256", "n_bindings",
        "presentations_per_binding", "train_template_ids", "eval_template_ids",
        "template_set", "seeds",
    ),
    "candidates": (
        "families", "capacity_grid", "learning_rate_grid", "batch_size_grid",
    ),
    "budgets": (
        "update_budget_steps", "token_entitlement", "write_element_ceiling",
    ),
    "mechanisms": (
        "fast_transience_mechanism",
    ),
    "interference": (
        "sets", "lifetime_horizon",
    ),
    "selection": (
        "rule", "tie_breaks",
    ),
    "accelerator": (
        "equivalence_report_sha256", "equivalence_passed",
    ),
    "provenance": (
        "git_sha", "source_tree_sha256", "config_hash",
    ),
}

#: Legal values for the FAST transience mechanism (prereg §5b, Amendment O-3).
TRANSIENCE_MECHANISMS = ("explicit_decay", "interference")


def _unset(value: Any) -> bool:
    """A field counts as unset if it is missing, None, or an empty container.

    Zero and False are *set* -- a zero budget is a decision, and refusing it
    here would hide a real (and invalid) configuration behind a "missing field"
    message instead of the budget check that should catch it.
    """
    if value is None:
        return True
    if isinstance(value, (str, list, tuple, dict, set)) and len(value) == 0:
        return True
    return False


def validate_stage1_freeze(freeze: dict) -> list[str]:
    """Return a list of blockers. Empty means the configuration may be frozen."""
    blockers: list[str] = []

    # -- completeness ------------------------------------------------------
    for group, fields in REQUIRED_FIELDS.items():
        section = freeze.get(group)
        if not isinstance(section, dict):
            blockers.append(f"missing group: {group}")
            continue
        for field in fields:
            if field not in section:
                blockers.append(f"missing field: {group}.{field}")
            elif _unset(section[field]):
                blockers.append(f"unset field: {group}.{field}")

    if blockers:
        # Semantic checks below index into fields that may not exist yet.
        return blockers

    task = freeze["task"]
    cand = freeze["candidates"]
    budg = freeze["budgets"]
    mech = freeze["mechanisms"]
    intf = freeze["interference"]

    # -- presentation split (Amendment O-2) --------------------------------
    train_t = set(task["train_template_ids"])
    eval_t = set(task["eval_template_ids"])
    overlap = sorted(train_t & eval_t)
    if overlap:
        blockers.append(
            f"train/eval presentation templates overlap: {overlap}. What is held out "
            "is the presentation, not the binding; a shared template leaks the "
            "optimized instance into scoring."
        )
    declared = set(task["template_set"])
    unknown = sorted((train_t | eval_t) - declared)
    if unknown:
        blockers.append(f"template ids not in template_set: {unknown}")
    if len(train_t) + len(eval_t) != task["presentations_per_binding"]:
        blockers.append(
            f"presentations_per_binding={task['presentations_per_binding']} does not equal "
            f"|train|+|eval| = {len(train_t)}+{len(eval_t)}"
        )
    if not eval_t:
        blockers.append("no evaluation presentations: absorption would be unmeasurable")

    # -- seeds must be the preregistered development set --------------------
    from .config import EXP002_DEV_SEEDS

    if tuple(task["seeds"]) != tuple(EXP002_DEV_SEEDS):
        blockers.append(
            f"task.seeds {tuple(task['seeds'])} != preregistered EXP002_DEV_SEEDS "
            f"{tuple(EXP002_DEV_SEEDS)}"
        )

    # -- grids --------------------------------------------------------------
    for fam in cand["families"]:
        if fam not in cand["capacity_grid"]:
            blockers.append(f"no capacity grid for declared family: {fam}")
    for name in ("capacity_grid", "learning_rate_grid", "batch_size_grid"):
        grid = cand[name]
        values = grid.values() if isinstance(grid, dict) else [grid]
        for v in values:
            if _unset(v):
                blockers.append(f"empty grid: candidates.{name}")

    # -- budgets ------------------------------------------------------------
    for field in ("update_budget_steps", "token_entitlement", "write_element_ceiling"):
        if not isinstance(budg[field], int) or budg[field] <= 0:
            blockers.append(f"budgets.{field} must be a positive integer")

    # -- transience mechanism (Amendment O-3) -------------------------------
    m = mech["fast_transience_mechanism"]
    if m not in TRANSIENCE_MECHANISMS:
        blockers.append(
            f"mechanisms.fast_transience_mechanism must be one of {TRANSIENCE_MECHANISMS}, "
            f"got {m!r}. The valid FAST transience test differs by mechanism, so it "
            "cannot be left implicit."
        )
    elif m == "explicit_decay" and _unset(mech.get("decay_constant")):
        blockers.append(
            "mechanisms.decay_constant is required when fast_transience_mechanism is "
            "'explicit_decay'"
        )

    # -- interference schedule (Amendment O-2) ------------------------------
    sets = intf["sets"]
    if len(sets) < 2:
        blockers.append(
            "interference.sets needs at least two sets: durability with no intervening "
            "writes is satisfied by doing nothing"
        )
    seen: set[Any] = set()
    for s in sets:
        members = set(s.get("members", []))
        if not members:
            blockers.append(f"interference set {s.get('name')!r} is empty")
        clash = sorted(members & seen)
        if clash:
            blockers.append(f"interference sets must be disjoint; repeated members: {clash}")
        seen |= members
    if not isinstance(intf["lifetime_horizon"], int) or intf["lifetime_horizon"] <= 0:
        blockers.append("interference.lifetime_horizon must be a positive integer")

    # -- accelerator --------------------------------------------------------
    if freeze["accelerator"]["equivalence_passed"] is not True:
        blockers.append("accelerator.equivalence_passed must be True before freezing")

    return blockers


def is_frozen(freeze: dict) -> bool:
    return not validate_stage1_freeze(freeze)


def blank_stage1_freeze() -> dict:
    """A fully-unset skeleton, for authoring. Validates to all-blockers by design."""
    return {group: {f: None for f in fields} for group, fields in REQUIRED_FIELDS.items()}
