# One-click run for single-cell clustering pipeline
# Usage: run from project root in PowerShell, or run with ExecutionPolicy bypass

# 1) Try to activate virtual environment if exists
$venv = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    Write-Host "Activating virtual environment..."
    & $venv
} else {
    Write-Host "No .venv found in project root; continuing with system Python."
}

# 2) Install requirements (safe to skip if already installed)
Write-Host "Ensuring Python dependencies..."
python -m pip install -r requirements.txt

# 3) Run clustering pipeline (adjust arguments if needed)
$cmd = "python clustering_pipeline.py --data Tosches_turtle.h5ad --outdir results --epochs 50 --latent 32 --batch 128 --device cpu"
Write-Host "Running pipeline: $cmd"
# Save console output to one_click_run.log and also print to console
$env:PYTHONUNBUFFERED = '1'
Write-Host "Running with PYTHONUNBUFFERED=$env:PYTHONUNBUFFERED"
Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath one_click_run.log

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline finished with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Pipeline completed successfully. Logs: one_click_run.log" -ForegroundColor Green
