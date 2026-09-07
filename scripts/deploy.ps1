# One-Click Cluster Deployment for Windows PowerShell
$ErrorActionPreference = "Stop"
$Namespace = "proactive-system"

# Check and append local k8s-tools directory if present
$ToolsDir = "$env:LOCALAPPDATA\Programs\k8s-tools"
if (Test-Path $ToolsDir) {
    if ($env:Path -notlike "*$ToolsDir*") {
        $env:Path = "$ToolsDir;$env:Path"
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Deploying Proactive Workload-Forecasting Scheduling System " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "[1/6] Creating namespace..." -ForegroundColor Yellow
kubectl apply -f kubernetes/namespace.yaml

Write-Host "[2/6] Applying RBAC permissions..." -ForegroundColor Yellow
kubectl apply -f module5_pod_scheduling/configuration/rbac.yaml

Write-Host "[3/6] Deploying Prometheus & Grafana monitoring..." -ForegroundColor Yellow
kubectl apply -f module6_monitoring/prometheus/prometheus-deployment.yaml
kubectl apply -f module6_monitoring/grafana/grafana-deployment.yaml

Write-Host "[4/6] Deploying ConfigMaps & Services..." -ForegroundColor Yellow
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/service.yaml

Write-Host "[5/6] Deploying Workload Forecast API & Proactive Scheduler..." -ForegroundColor Yellow
kubectl apply -f module5_pod_scheduling/configuration/scheduler-deployment.yaml

Write-Host "[6/6] Deploying Demo Workload Microservice..." -ForegroundColor Yellow
kubectl apply -f kubernetes/deployment.yaml

Write-Host "Waiting for pods to stabilize in $Namespace..." -ForegroundColor Green
kubectl get pods -n $Namespace -o wide

Write-Host "Deployment complete! Verify placement with: kubectl get pods -n $Namespace -o wide" -ForegroundColor Green
