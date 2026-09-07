# Experiment Runner for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Executing End-to-End Comparative Scheduling Experiment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "[1/3] Running comparative evaluation against baselines..." -ForegroundColor Yellow
python evaluation/evaluate.py

Write-Host "[2/3] Generating 13 publication-quality comparison plots..." -ForegroundColor Yellow
python evaluation/plots.py

Write-Host "[3/3] Inspecting output figures in evaluation/plots/..." -ForegroundColor Green
Get-ChildItem evaluation/plots/

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Experiment Complete! Results saved to evaluation/results/" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
