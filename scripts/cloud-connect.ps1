<#
.SYNOPSIS
    Connect to Google Kubernetes Engine (GKE) Cluster (Windows PowerShell)

.DESCRIPTION
    Configures kubectl to target a GKE cluster using gcloud and validates connection.
#>

param(
    [string]$ClusterName = "proactive-scheduler-gke",
    [string]$Zone = "us-central1-a",
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

Write-Host "Connecting to GKE cluster '$ClusterName' in zone '$Zone'..." -ForegroundColor Cyan

$cmd = @($gcloud, "container", "clusters", "get-credentials", $ClusterName, "--zone", $Zone)
if ($ProjectId) {
    $cmd += @("--project", $ProjectId)
}

& $cmd[0] $cmd[1..($cmd.Length - 1)]

Write-Host ""
Write-Host "Verifying cluster nodes:" -ForegroundColor Yellow
& $kubectl get nodes -o wide

Write-Host ""
Write-Host "[OK] Connected to GKE! Refresh the web portal at http://localhost:8000 to see live GKE nodes." -ForegroundColor Green
