#!/usr/bin/env bash
set -e

NAMESPACE="proactive-system"

echo "============================================================"
echo " Deploying Proactive Workload-Forecasting Scheduling System "
echo "============================================================"

echo "[1/6] Creating namespace..."
kubectl apply -f kubernetes/namespace.yaml

echo "[2/6] Applying RBAC permissions..."
kubectl apply -f module5_pod_scheduling/configuration/rbac.yaml

echo "[3/6] Deploying Prometheus & Grafana monitoring..."
kubectl apply -f module6_monitoring/prometheus/prometheus-deployment.yaml
kubectl apply -f module6_monitoring/grafana/grafana-deployment.yaml

echo "[4/6] Deploying ConfigMaps & Services..."
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/service.yaml

echo "[5/6] Deploying Workload Forecast API & Proactive Scheduler..."
kubectl apply -f module5_pod_scheduling/configuration/scheduler-deployment.yaml

echo "[6/6] Deploying Demo Workload Microservice..."
kubectl apply -f kubernetes/deployment.yaml

echo "Waiting for pods to stabilize in ${NAMESPACE}..."
kubectl get pods -n ${NAMESPACE} -o wide

echo "Deployment complete! Verify placement with: kubectl get pods -n ${NAMESPACE} -o wide"
