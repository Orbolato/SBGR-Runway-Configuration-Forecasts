#!/usr/bin/env bash
# Regenerate every reported result. Runs the full pipeline when the restricted
# TATIC records are present; otherwise regenerates tables, macros, and figures
# from the archived results record.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

if [[ -f data/TATIC/Arrivals/dataTF_arr_SBGR.Rda && -f data/TATIC/Departures/dataTF_dep_SBGR.Rda ]]; then
  echo "Restricted TATIC records found: full rerun."
  "$PYTHON" code/run_analysis.py
  "$PYTHON" code/verify_outputs.py
else
  echo "Restricted TATIC records not found: regenerating from results/analysis_results.json."
  "$PYTHON" code/run_analysis.py --artifacts-only
  "$PYTHON" code/make_figures.py
fi
