#!/usr/bin/env bash
set -e

# ==============================================================================
# Build & Package Proactive Kubernetes Scheduler Portal for Netlify Hosting
# ==============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
STATIC_DIR="${PROJECT_ROOT}/web/static"
PLOTS_DIR="${PROJECT_ROOT}/evaluation/plots"

echo "Building Netlify publication bundle..."
mkdir -p "${DIST_DIR}/plots"

# Copy static website files
cp "${STATIC_DIR}/index.html" "${DIST_DIR}/"
cp "${STATIC_DIR}/styles.css" "${DIST_DIR}/"
cp "${STATIC_DIR}/app.js" "${DIST_DIR}/"

# Copy publication plots
cp "${PLOTS_DIR}"/*.png "${DIST_DIR}/plots/"

# Copy Netlify redirect rules
echo "/*    /index.html   200" > "${DIST_DIR}/_redirects"

echo "[SUCCESS] Netlify dist bundle generated at: ${DIST_DIR}"
echo "Ready for deployment to Netlify!"
