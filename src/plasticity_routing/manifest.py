"""Machine-readable run manifests.

A result that cannot be reproduced from its manifest does not count. Each
manifest pins the source tree, the exact configs, the seed, the arm, every
resource total, and the leakage-audit outcome, and it carries an explicit
`classification` so a development pilot can never be mistaken for evidence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "plasticity-routing/manifest/1"

CLASSIFICATIONS = ("SMOKE", "DEV_CALIBRATION", "ENGINEERING_PILOT", "CONFIRMATORY")


#: Generated artefacts excluded from the source fingerprint. `PROTOCOL-v*.md` is
#: *written by* the freeze, so including it would make every freeze invalidate
#: the very audit verdicts it just checked.
_GENERATED = ("PROTOCOL-v",)


def source_tree_sha256(root: Path) -> str:
    """Deterministic fingerprint of the source tree.

    Covers `.py`, `.md`, `.toml` and `.cff` files, excluding the virtualenv, git
    metadata, the `results/` output directory, and generated artefacts.
    """
    h = hashlib.sha256()
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".md", ".toml", ".cff"}
        and ".venv" not in p.parts
        and ".git" not in p.parts
        and "results" not in p.parts
        and not p.name.startswith(_GENERATED)
    )
    for p in paths:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def config_hash(*configs: Any) -> str:
    blob = json.dumps([asdict(c) if is_dataclass(c) else c for c in configs], sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def environment() -> dict:
    import numpy as np

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def build_manifest(
    *,
    root: Path,
    classification: str,
    arm: str,
    result,
    world_cfg,
    sub_cfg,
    cost_cfg,
    train_cfg=None,
    dev_seeds: list[int] | None = None,
    leakage: dict | None = None,
    notes: str = "",
) -> dict:
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"classification must be one of {CLASSIFICATIONS}")
    from .metrics import confusion_table, routing_agreement_diagnostic

    return {
        "schema": SCHEMA_VERSION,
        "classification": classification,
        "arm": arm,
        "router": result.router_name,
        "decision_time_legal": result.legal,
        "seed": result.seed,
        "dev_seeds": dev_seeds or [],
        "git_sha": git_sha(root),
        "source_tree_sha256": source_tree_sha256(root),
        "config_hash": config_hash(world_cfg, sub_cfg, cost_cfg, train_cfg),
        "environment": environment(),
        "world_config": asdict(world_cfg),
        "substrate_config": asdict(sub_cfg),
        "cost_config": asdict(cost_cfg),
        "train_config": asdict(train_cfg) if train_cfg is not None else None,
        "metrics": {
            "task_utility": result.task_utility,
            "forgetting": result.forgetting,
            "objective": result.objective,
            "n_queries": result.n_queries,
            "utility_by_class": result.utility_by_class,
            "episodic_evictions": result.episodic_evictions,
        },
        "resources": result.ledger,
        "action_histogram": result.action_histogram,
        "diagnostics": {
            "routing_agreement": routing_agreement_diagnostic(result.confusion),
            "confusion_hidden_class_by_action": confusion_table(result.confusion),
        },
        "leakage_audit": leakage,
        "invalidation_reasons": [],
        "notes": notes,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
