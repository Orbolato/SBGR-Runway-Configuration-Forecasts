"""Lightweight repository-integrity checks that do not require restricted data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SHA256 = {
    "results/analysis_results.json": "90c8792270694f222095578dd0c7621171e5387f546beb157549dbb91c890bad",
    "results/figure_data.json": "92a1f86c8b648e2b054a40d6a5804a98feede42846bbdcc4348bbd0e85b148a6",
    "results/figure_source_values.json": "723526b253b01b7554ca6df31c0e716172efc3882e085463ed45a78561259001",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_archived_results_are_exact() -> None:
    for relative, expected in EXPECTED_SHA256.items():
        path = ROOT / relative
        assert path.is_file(), f"Missing archived result: {relative}"
        assert sha256(path) == expected, f"Unexpected contents: {relative}"
        with path.open("r", encoding="utf-8") as stream:
            json.load(stream)


def test_restricted_tatic_records_are_absent() -> None:
    restricted = ROOT / "data" / "TATIC"
    if restricted.exists():
        forbidden = [p for p in restricted.rglob("*") if p.is_file() and p.suffix.lower() in {".rda", ".csv"}]
        assert not forbidden, f"Restricted TATIC data must not be committed: {forbidden}"


def test_derived_analysis_grids_are_absent() -> None:
    assert not list(ROOT.glob("analysis_grid*.csv"))


def test_citation_points_to_canonical_repository() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    canonical = "https://github.com/Orbolato/SBGR-Runway-Configuration-Forecasts"
    assert canonical in text
    assert "JATM-D-26-00770" in text
