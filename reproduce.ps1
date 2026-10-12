# Regenerate every reported result. Runs the full pipeline when the restricted
# TATIC records are present; otherwise regenerates tables, macros, and figures
# from the archived results record.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }

if ((Test-Path 'data/TATIC/Arrivals/dataTF_arr_SBGR.Rda') -and (Test-Path 'data/TATIC/Departures/dataTF_dep_SBGR.Rda')) {
    Write-Host 'Restricted TATIC records found: full rerun.'
    & $python code/run_analysis.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python code/verify_outputs.py
} else {
    Write-Host 'Restricted TATIC records not found: regenerating from results/analysis_results.json.'
    & $python code/run_analysis.py --artifacts-only
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python code/make_figures.py
}
exit $LASTEXITCODE
