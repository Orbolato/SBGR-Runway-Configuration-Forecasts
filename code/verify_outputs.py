#!/usr/bin/env python3
"""Check a rerun of the pipeline against the archived results of the article.

The reference is rebuilt in a temporary folder from ``results/``: Tables 3-10 and
``generated_results.tex`` with the same writer used by the pipeline, and Figures
1-7 with ``make_figures.py``. Tables and macros rebuilt this way are identical to
those in the submitted R2 manuscript. The check covers whatever is present in the
output folder:

1. ``analysis_results.json`` against ``results/analysis_results.json``: every numeric
   value, excluding wall-clock timing fields.
2. ``figure_data.json`` against ``results/figure_data.json``.
3. Tables 3-10 and ``generated_results.tex`` (exact text). Lines that differ only in
   wall-clock timing values are reported separately: the article states that fit and
   inference times are the only quantities not fixed by the random seed.
4. Figures 1-7, by rendering both PDFs and comparing pixels (requires PyMuPDF;
   skipped otherwise).

Usage:
    python code/verify_outputs.py [--out output] [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "results"
GENERATED = ["Table03_Data_Availability.tex", "Table04_Feature_Set.tex",
             "Table05_Main_Performance.tex", "Table06_Transition_Performance.tex",
             "Table07_Imbalance_Handling.tex", "Table08_Boosting_Implementations.tex",
             "Table09_Literature_Reimplementation.tex", "Table10_Operating_Conditions.tex",
             "generated_results.tex"]
# Wall-clock quantities and environment descriptors that legitimately vary by machine.
TIMING_KEYS = re.compile(r"fit_seconds|median_milliseconds_per_1000_rows|median_total_seconds|"
                         r"^/software/platform$")
TIMING_MACROS = re.compile(r"Latency|FitTime|Seconds|Millisecond", re.I)


def build_reference(folder: Path) -> None:
    """Regenerate tables, macros, and figures from the archived results."""
    from make_figures import draw_all
    from run_analysis import normalize_results, write_artifacts

    archived = json.loads((ARCHIVE / "analysis_results.json").read_text(encoding="utf-8"))
    write_artifacts(normalize_results(json.loads(json.dumps(archived))), folder)
    figure_data = ARCHIVE / "figure_data.json"
    if figure_data.exists():
        draw_all(archived, json.loads(figure_data.read_text(encoding="utf-8")), folder)


def numeric_leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from numeric_leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numeric_leaves(v, f"{path}/{i}")
    else:
        yield path, obj


def compare_json(new: Path, old: Path, tol: float) -> bool:
    a = dict(numeric_leaves(json.loads(new.read_text(encoding="utf-8"))))
    b = dict(numeric_leaves(json.loads(old.read_text(encoding="utf-8"))))
    missing = sorted(set(b) - set(a))
    extra = sorted(set(a) - set(b))
    worst, mismatches, timing = 0.0, [], 0
    for key in sorted(set(a) & set(b)):
        if TIMING_KEYS.search(key):
            timing += 1
            continue
        x, y = a[key], b[key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            diff = abs(float(x) - float(y))
            worst = max(worst, diff)
            if diff > tol:
                mismatches.append((key, y, x))
        elif x != y:
            mismatches.append((key, y, x))
    print(f"  {new.name}: {len(set(a) & set(b)) - timing} values compared, "
          f"{timing} timing values skipped, max |diff| = {worst:.3g}")
    for key, y, x in mismatches[:20]:
        print(f"    MISMATCH {key}: archived={y!r} rerun={x!r}")
    if len(mismatches) > 20:
        print(f"    ... {len(mismatches) - 20} more")
    for label, keys in (("missing", missing), ("extra", extra)):
        if keys:
            print(f"    {len(keys)} {label} keys, e.g. {keys[:3]}")
    return not mismatches and not missing


def compare_text(out: Path, reference: Path) -> bool:
    ok = True
    for name in GENERATED:
        new = out / name
        if not new.exists():
            continue
        ref = (reference / name).read_text(encoding="utf-8").splitlines()
        got = new.read_text(encoding="utf-8").splitlines()
        if ref == got:
            print(f"  IDENTICAL   {name}")
            continue
        differing = [(r, g) for r, g in zip(ref, got) if r != g]
        only_timing = len(ref) == len(got) and all(
            (name.startswith("Table08") and re.sub(r"& \d+ \\\\$", "", r) == re.sub(r"& \d+ \\\\$", "", g))
            or TIMING_MACROS.search(r) for r, g in differing)
        if only_timing:
            print(f"  TIMING ONLY {name}: {len(differing)} line(s) differ in wall-clock values")
        else:
            ok = False
            print(f"  DIFFERS     {name}")
            for r, g in differing[:5]:
                print(f"    - {r}\n    + {g}")
    return ok


def compare_figures(out: Path, reference: Path) -> bool:
    try:
        import fitz  # PyMuPDF
        import numpy as np
    except ImportError:
        print("  PyMuPDF not installed; figure comparison skipped")
        return True
    ok = True
    for i in range(1, 8):
        name = f"Figure_{i:02}.pdf"
        if not (out / name).exists() or not (reference / name).exists():
            continue
        images = []
        for path in (reference / name, out / name):
            pix = fitz.open(path)[0].get_pixmap(dpi=150, colorspace=fitz.csGRAY)
            images.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width))
        if images[0].shape != images[1].shape:
            ok = False
            print(f"  DIFFERS     {name}: page size {images[0].shape} vs {images[1].shape}")
            continue
        changed = np.mean(np.abs(images[0].astype(int) - images[1].astype(int)) > 32)
        ok &= changed < 0.001
        status = "MATCHES    " if changed < 0.001 else "DIFFERS    "
        print(f"  {status} {name}: {100 * changed:.3f}% of pixels differ at 150 dpi")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "output")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="absolute tolerance for numeric values in the JSON records")
    args = parser.parse_args()
    if not args.out.exists():
        print(f"No output folder at {args.out}; run the pipeline first.")
        return 1
    ok = True
    for name in ("analysis_results.json", "figure_data.json"):
        if (args.out / name).exists() and (ARCHIVE / name).exists():
            print("Results record" if name.startswith("analysis") else "Figure series")
            ok &= compare_json(args.out / name, ARCHIVE / name, args.tol)
    with tempfile.TemporaryDirectory(prefix="sbgr_reference_") as tmp:
        reference = Path(tmp)
        build_reference(reference)
        print("Tables and macros")
        ok &= compare_text(args.out, reference)
        print("Figures")
        ok &= compare_figures(args.out, reference)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
