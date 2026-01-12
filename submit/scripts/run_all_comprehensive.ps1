# ==============================================================================
# Single Cell Clustering Analysis - Comprehensive Run Script
# ==============================================================================
# Description: Run all experiments step by step
# Usage: .\scripts\run_all_comprehensive.ps1
# ==============================================================================

Write-Host "========================================" -ForegroundColor Green
Write-Host "Single Cell Clustering Analysis - Running" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Check Python environment
Write-Host "`n[1/4] Checking Python environment..." -ForegroundColor Cyan
python --version

# Install dependencies
Write-Host "`n[2/4] Installing dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

# Run basic task (50 points) - Autoencoder clustering
Write-Host "`n[3/4] Running basic task - Autoencoder clustering..." -ForegroundColor Cyan
Write-Host "Dataset 1: Tosches_turtle.h5ad" -ForegroundColor Yellow
python code/clustering_pipeline.py --data data/Tosches_turtle.h5ad --outdir results_turtle --epochs 50 --latent 32 --batch 128 --device cpu

Write-Host "Dataset 2: Quake_Lung.h5ad" -ForegroundColor Yellow
python code/clustering_pipeline.py --data data/Quake_Lung.h5ad --outdir results_lung --epochs 50 --latent 32 --batch 128 --device cpu

Write-Host "Dataset 3: Quake_Diaphragm.h5ad" -ForegroundColor Yellow
python code/clustering_pipeline.py --data data/Quake_Diaphragm.h5ad --outdir results_diaphragm --epochs 50 --latent 32 --batch 128 --device cpu

# Run innovative methods (50 points) - 7 algorithms multi-dataset
Write-Host "`n[4/4] Running innovative methods - 7 algorithms multi-dataset..." -ForegroundColor Cyan
python code/multi_dataset_runner.py
