# Minikube Multi-Node Cluster Setup for Windows PowerShell
$ErrorActionPreference = "Stop"
$ClusterName = "proactive-cluster"
$Nodes = 3
$Cpus = 2
$Memory = "3072MB"

# Check and append local k8s-tools directory if present
$ToolsDir = "$env:LOCALAPPDATA\Programs\k8s-tools"
if (Test-Path $ToolsDir) {
    if ($env:Path -notlike "*$ToolsDir*") {
        $env:Path = "$ToolsDir;$env:Path"
    }
}

# Verify minikube availability
if (-not (Get-Command minikube -ErrorAction SilentlyContinue)) {
    Write-Host "minikube not detected in PATH. Running automated installer..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "install-k8s-tools.ps1")
    if (Test-Path $ToolsDir) {
        $env:Path = "$ToolsDir;$env:Path"
    }
}

# Detect container driver (Docker vs Hyper-V)
$Driver = "docker"
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker not detected on host; checking Hyper-V..." -ForegroundColor Yellow
    $Driver = "hyperv"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Starting Minikube Multi-Node Cluster: $ClusterName" -ForegroundColor Cyan
Write-Host " Nodes: $Nodes, CPUs: $Cpus, Memory: $Memory, Driver: $Driver" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

minikube start `
  -p $ClusterName `
  --nodes=$Nodes `
  --cpus=$Cpus `
  --memory=$Memory `
  --driver=$Driver

Write-Host "Enabling metrics-server addon..." -ForegroundColor Green
minikube addons enable metrics-server -p $ClusterName

Write-Host "Cluster worker node topology:" -ForegroundColor Yellow
kubectl get nodes -o wide

Write-Host "Minikube multi-node cluster is ready!" -ForegroundColor Green
