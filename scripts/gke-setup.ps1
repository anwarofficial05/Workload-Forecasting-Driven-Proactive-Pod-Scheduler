<#
.SYNOPSIS
    Google Kubernetes Engine (GKE) Standard Cluster Provisioning & Deployment Script (Windows PowerShell)

.DESCRIPTION
    Automates creating a 3-node GKE Standard cluster, fetching credentials for kubectl,
    and deploying the Proactive Pod Scheduler manifests.

.PARAMETER ProjectId
    The Google Cloud Platform Project ID.

.PARAMETER ClusterName
    The name for the GKE cluster (defaults to "proactive-scheduler-gke").

.PARAMETER Zone
    The GCP compute zone (defaults to "us-central1-a").

.PARAMETER MachineType
    The machine type for the nodes (defaults to "e2-standard-4").
#>

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$ClusterName = "proactive-scheduler-gke",
    [string]$Zone = "us-central1-a",
    [string]$MachineType = "e2-standard-4",
    [int]$NumNodes = 3
)

$ErrorActionPreference = "Stop"

# Helper to find gcloud & kubectl
function Get-ToolPath($name) {
    $p = Get-Command $name -ErrorAction SilentlyContinue
    if ($p) { return $p.Source }
    $local = $env:LOCALAPPDATA
    $known = @{
        "gcloud" = @(
            "$local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
            "C:\Program Files\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
        );
        "kubectl" = @(
            "$local\Programs\k8s-tools\kubectl.exe",
            "$local\Google\Cloud SDK\google-cloud-sdk\bin\kubectl.exe"
        )
    }
    foreach ($cand in $known[$name]) {
        if (Test-Path $cand) { return $cand }
    }
    return $name
}

$gcloud = Get-ToolPath "gcloud"
$kubectl = Get-ToolPath "kubectl"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  GKE Standard Cluster Automation (Proactive Scheduler)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Verify gcloud
try {
    & $gcloud --version | Out-Null
    Write-Host "[OK] Google Cloud SDK detected: $gcloud" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Google Cloud SDK (gcloud) is not found." -ForegroundColor Red
    exit 1
}

# Verify Project ID
if (-not $ProjectId) {
    $currentProj = & $gcloud config get-value project 2>$null
    if ($currentProj -and $currentProj -ne "(unset)") {
        $ProjectId = $currentProj.Trim()
    } else {
        $ProjectId = Read-Host "Enter your Google Cloud Project ID"
    }
}

if (-not $ProjectId) {
    Write-Host "[ERROR] Project ID is required." -ForegroundColor Red
    exit 1
}

Write-Host "Configuring active project: $ProjectId" -ForegroundColor Yellow
& $gcloud config set project $ProjectId

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Provisioning GKE Cluster: $ClusterName" -ForegroundColor Cyan
Write-Host " Zone: $Zone | Nodes: $NumNodes | Machine: $MachineType" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Provision GKE cluster
& $gcloud container clusters create $ClusterName `
    --zone $Zone `
    --num-nodes $NumNodes `
    --machine-type $MachineType `
    --enable-ip-alias `
    --scopes "https://www.googleapis.com/auth/cloud-platform" `
    --addons HorizontalPodAutoscaling,HttpLoadBalancing

Write-Host ""
Write-Host "[OK] GKE Cluster created successfully. Fetching kubectl credentials..." -ForegroundColor Green
& $gcloud container clusters get-credentials $ClusterName --zone $Zone

Write-Host ""
Write-Host "Testing Kubernetes cluster connectivity:" -ForegroundColor Yellow
& $kubectl get nodes -o wide

Write-Host ""
Write-Host "Deploying Proactive Scheduler manifests to GKE..." -ForegroundColor Yellow
$manifest = Join-Path $PSScriptRoot "..\kubernetes\gke\gke-deploy.yaml"
if (Test-Path $manifest) {
    & $kubectl apply -f $manifest
    Write-Host "[OK] Proactive Scheduler and demo workloads deployed to GKE!" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Manifest file not found at: $manifest" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " GKE Deployment Completed!" -ForegroundColor Green
Write-Host " Check pods with: kubectl get pods -n proactive-system -o wide" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
