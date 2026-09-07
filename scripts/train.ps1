# Model Training Pipeline for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Running LSTM and XGBoost Training Pipeline" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

python module3_workload_prediction/train.py `
  --config config/config.yaml `
  --epochs 15 `
  --batch-size 32 `
  --samples 2000

Write-Host "Training completed successfully. Models saved to models/" -ForegroundColor Green
