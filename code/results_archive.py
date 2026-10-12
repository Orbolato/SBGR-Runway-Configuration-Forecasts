#!/usr/bin/env python3
"""Load the archived results record from storage-only JSON sections."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SECTIONS = REPO / "results" / "analysis_sections"
SECTION_FILES = (
    "01_audit.json",
    "02_models_15.json",
    "03_models_30.json",
    "04_models_60.json",
    "05_differences_selection_sensitivity.json",
    "06_ablation_perm60.json",
    "07_importance_selected.json",
    "08_importance_boosting.json",
    "09_imbalance.json",
    "10_boosting_implementations.json",
    "11_literature_reimplementations.json",
    "12_recursive.json",
    "13_difficulty_shift_events.json",
    "14_threshold_tradeoff.json",
    "15_operating_15.json",
    "16_operating_30.json",
    "17_operating_60.json",
    "18_latency_software.json",
)


def _merge_ordered(target: dict, incoming: dict) -> None:
    """Merge nested mappings while preserving first-seen insertion order."""
    for key, value in incoming.items():
        if key not in target:
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _merge_ordered(target[key], value)
        else:
            raise ValueError(f"Duplicate archived result value at key {key!r}")


def load_archived_results() -> dict:
    """Reassemble the original top-level results mapping in canonical key order."""
    combined: dict = {}
    for name in SECTION_FILES:
        part = json.loads((SECTIONS / name).read_text(encoding="utf-8"))
        _merge_ordered(combined, part)
    # The selected 30- and 60-minute models are gradient boosting, so their
    # stored permutation-importance series are identical to the boosting series.
    # Reuse those exact values rather than duplicating them in storage.
    boost = combined["permutation_importance_boosting"]
    selected = combined["permutation_importance_selected"]
    boost["30"] = selected["30"]
    boost["60"] = selected["60"]
    return combined


def canonical_bytes() -> bytes:
    """Return the original Windows/CRLF JSON serialization byte-for-byte."""
    text = json.dumps(load_archived_results(), indent=2, default=str)
    return text.replace("\n", "\r\n").encode("utf-8")
