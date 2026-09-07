<#
.SYNOPSIS
    Build & Package Proactive Kubernetes Scheduler Portal for Netlify Hosting
#>

$ErrorActionPreference = "Stop"

Write-Host "Building Proactive Scheduler portal for Netlify..." -ForegroundColor Cyan

$dist = Join-Path $PSScriptRoot "..\dist"
$webStatic = Join-Path $PSScriptRoot "..\web\static"
$plotsDir = Join-Path $PSScriptRoot "..\evaluation\plots"

# Ensure directories
New-Item -ItemType Directory -Force -Path (Join-Path $dist "plots") | Out-Null

# Copy static assets
Copy-Item (Join-Path $webStatic "index.html") $dist -Force
Copy-Item (Join-Path $webStatic "styles.css") $dist -Force
Copy-Item (Join-Path $webStatic "app.js") $dist -Force

# Copy evaluation plots
Copy-Item (Join-Path $plotsDir "*.png") (Join-Path $dist "plots") -Force

# Copy Netlify redirects
"/*    /index.html   200" | Out-File -FilePath (Join-Path $dist "_redirects") -Encoding ascii

Write-Host "[SUCCESS] Netlify dist bundle generated at: $dist" -ForegroundColor Green
Write-Host "You can deploy to Netlify using:" -ForegroundColor Yellow
Write-Host "  1. Netlify CLI: 'netlify deploy --dir=dist --prod'" -ForegroundColor Yellow
Write-Host "  2. Drag and drop the 'dist' folder onto app.netlify.com/drop" -ForegroundColor Yellow
Write-Host "  3. Push to GitHub with the included 'netlify.toml'" -ForegroundColor Yellow
