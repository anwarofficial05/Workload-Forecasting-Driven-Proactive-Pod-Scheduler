<#
.SYNOPSIS
    Deploy Proactive Scheduler Web Portal to Google Cloud Run (Windows PowerShell)

.DESCRIPTION
    Builds the project Docker container and deploys it to Google Cloud Run,
    making the interactive web application publicly accessible via HTTPS.
#>

param(
    [string]$ServiceName = "proactive-scheduler-portal",
    [string]$Region = "us-central1",
    [string]$ProjectId = $env:GCP_PROJECT_ID
)

$ErrorActionPreference = "Stop"

function Get-ToolPath($name) {
    $p = Get-Command $name -ErrorAction SilentlyContinue
    if ($p) { return $p.Source }
    $local = $env:LOCALAPPDATA
    $known = @{
        "gcloud" = @(
            "$local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            "C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        )
    }
    foreach ($cand in $known[$name]) {
        if (Test-Path $cand) { return $cand }
    }
    return $name
}

$gcloud = Get-ToolPath "gcloud"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Deploying Proactive Scheduler Portal to Google Cloud Run" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not $ProjectId) {
    $currentProj = & $gcloud config get-value project 2>$null
    if ($currentProj -and $currentProj -ne "(unset)") {
        $ProjectId = $currentProj.Trim()
    } else {
        $ProjectId = Read-Host "Enter your Google Cloud Project ID"
    }
}

Write-Host "Project: $ProjectId | Region: $Region | Service: $ServiceName" -ForegroundColor Yellow
& $gcloud config set project $ProjectId

$projectRoot = Split-Path $PSScriptRoot -Parent

Write-Host ""
Write-Host "Starting Cloud Run Build & Deploy from: $projectRoot" -ForegroundColor Yellow
& $gcloud run deploy $ServiceName `
    --source $projectRoot `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --port 8000 `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 1 `
    --set-env-vars="PORT=8000"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Cloud Run Deployment Completed!" -ForegroundColor Green
Write-Host " The public HTTPS portal URL is printed above." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
