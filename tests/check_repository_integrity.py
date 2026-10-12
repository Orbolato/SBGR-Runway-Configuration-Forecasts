#!/usr/bin/env python3
"""Lightweight integrity checks that require only the Python standard library."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

EXPECTED_SHA256 = {
    "results/analysis_results.json": "90c8792270694f222095578dd0c7621171e5387f546beb157549dbb91c890bad",
    "results/figure_data.json": "92a1f86c8b648e2b054a40d6a5804a98feede42846bbdcc4348bbd0e85b148a6",
    "results/figure_source_values.json": "723526b253b01b7554ca6df31c0e716172efc3882e085463ed45a78561259001",
}

SOURCE_DATA_PATHS = (
    "data/METAR/dataMETAR_SBGR.Rda",
    "data/SIROS/Arrivals/dataSR_arr_SBGR.Rda",
    "data/SIROS/Departures/dataSR_dep_SBGR.Rda",
    "data/TATIC/Arrivals/dataTF_arr_SBGR.Rda",
    "data/TATIC/Departures/dataTF_dep_SBGR.Rda",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    for relative, expected in EXPECTED_SHA256.items():
        path = REPO / relative
        if not path.is_file():
            raise SystemExit(f"Missing archived artifact: {relative}")
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {relative}: {actual}")

    for relative in SOURCE_DATA_PATHS:
        if (REPO / relative).exists():
            raise SystemExit(f"Source-data file must not be committed: {relative}")

    results = json.loads((REPO / "results/analysis_results.json").read_text(encoding="utf-8"))
    if results.get("software", {}).get("random_seed") != 20260720:
        raise SystemExit("Unexpected random seed in archived results")

    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    canonical = "https://github.com/Orbolato/SBGR-Runway-Configuration-Forecasts"
    if canonical not in citation:
        raise SystemExit("CITATION.cff does not contain the canonical repository URL")

    print("Repository integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
