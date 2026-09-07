#!/usr/bin/env bash
set -e

# ==============================================================================
# Google Kubernetes Engine (GKE) Standard Cluster Provisioning Script
# ==============================================================================

PROJECT_ID=${GCP_PROJECT_ID:-"my-gcp-project"}
CLUSTER_NAME="proactive-scheduler-gke"
ZONE="us-central1-a"
NUM_NODES=3
MACHINE_TYPE="e2-standard-4"

echo "============================================================"
echo " Provisioning GKE Standard Cluster: ${CLUSTER_NAME}"
echo " Project: ${PROJECT_ID}, Zone: ${ZONE}, Nodes: ${NUM_NODES}"
echo " Machine Type: ${MACHINE_TYPE} (4 vCPU, 16GB RAM per node)"
echo "============================================================"

# Set project
gcloud config set project "${PROJECT_ID}"

# Create GKE Standard cluster
gcloud container clusters create "${CLUSTER_NAME}" \
  --zone "${ZONE}" \
  --num-nodes "${NUM_NODES}" \
  --machine-type "${MACHINE_TYPE}" \
  --enable-ip-alias \
  --scopes "https://www.googleapis.com/auth/cloud-platform" \
  --addons HorizontalPodAutoscaling,HttpLoadBalancing

# Fetch cluster credentials for kubectl
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone "${ZONE}"

echo "Configured kubectl context for GKE cluster."
kubectl get nodes -o wide

echo "Deploying system to GKE..."
kubectl apply -f kubernetes/gke/gke-deploy.yaml
echo "GKE deployment completed successfully!"
