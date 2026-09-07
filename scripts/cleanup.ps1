# Teardown and cleanup script for Windows PowerShell
$Namespace = "proactive-system"

Write-Host "Tearing down proactive scheduling resources in namespace $Namespace..." -ForegroundColor Yellow
kubectl delete namespace $Namespace --ignore-not-found=true
kubectl delete clusterrole proactive-scheduler-role --ignore-not-found=true
kubectl delete clusterrolebinding proactive-scheduler-binding --ignore-not-found=true

Write-Host "Cleanup completed." -ForegroundColor Green
