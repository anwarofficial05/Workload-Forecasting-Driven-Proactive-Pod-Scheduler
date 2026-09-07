# ==============================================================================
# Helper Script: Download and Configure Minikube and Kubectl for Windows
# ==============================================================================
$ErrorActionPreference = "Stop"

$InstallDir = "$env:LOCALAPPDATA\Programs\k8s-tools"
Write-Host "Creating installation directory at: $InstallDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# 1. Download Kubectl binary
$KubectlPath = Join-Path $InstallDir "kubectl.exe"
if (-not (Test-Path $KubectlPath)) {
    Write-Host "Downloading kubectl v1.29.2..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://dl.k8s.io/release/v1.29.2/bin/windows/amd64/kubectl.exe" -OutFile $KubectlPath
    Write-Host "Downloaded kubectl successfully." -ForegroundColor Green
} else {
    Write-Host "kubectl already present." -ForegroundColor Green
}

# 2. Download Minikube binary
$MinikubePath = Join-Path $InstallDir "minikube.exe"
if (-not (Test-Path $MinikubePath)) {
    Write-Host "Downloading minikube latest..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri "https://github.com/kubernetes/minikube/releases/latest/download/minikube-windows-amd64.exe" -OutFile $MinikubePath
    Write-Host "Downloaded minikube successfully." -ForegroundColor Green
} else {
    Write-Host "minikube already present." -ForegroundColor Green
}

# 3. Add to Current Session PATH and User Environment PATH
if ($env:Path -notlike "*$InstallDir*") {
    $env:Path = "$InstallDir;$env:Path"
    $userPath = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::User)
    if ($userPath -notlike "*$InstallDir*") {
        [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$userPath", [EnvironmentVariableTarget]::User)
    }
}

Write-Host "`nTools installed successfully!" -ForegroundColor Green
Write-Host "Verifying tool versions:" -ForegroundColor Cyan
& "$KubectlPath" version --client
& "$MinikubePath" version

Write-Host "`nYou can now run:" -ForegroundColor Yellow
Write-Host "  minikube start -p proactive-cluster --nodes=3" -ForegroundColor White
