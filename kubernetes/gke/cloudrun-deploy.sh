#!/usr/bin/env bash
set -e

# ==============================================================================
# One-Click Cloud Deployment to Google Cloud Run (Public Web Application)
# ==============================================================================

SERVICE_NAME="proactive-scheduler-portal"
REGION=${GCP_REGION:-"us-central1"}
PROJECT_ID=${GCP_PROJECT_ID:-$(gcloud config get-value project 2>/dev/null || echo "my-gcp-project")}

echo "============================================================"
echo " Deploying Proactive Scheduler Portal to Google Cloud Run"
echo " Project: ${PROJECT_ID}, Region: ${REGION}, Service: ${SERVICE_NAME}"
echo "============================================================"

# Ensure gcloud is configured
if ! command -v gcloud &> /dev/null; then
    echo "Error: Google Cloud SDK (gcloud) is not installed."
    echo "Install via: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

gcloud config set project "${PROJECT_ID}"

echo "Building container and deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --set-env-vars="PORT=8000"

echo ""
echo "============================================================"
echo " Cloud Deployment Succeeded!"
echo " Access your public web application via the URL printed above."
echo "============================================================"
