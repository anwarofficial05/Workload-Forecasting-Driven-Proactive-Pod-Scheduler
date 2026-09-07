#!/usr/bin/env bash
NAMESPACE="proactive-system"

echo "Tearing down proactive scheduling resources in namespace ${NAMESPACE}..."
kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
kubectl delete clusterrole proactive-scheduler-role --ignore-not-found=true
kubectl delete clusterrolebinding proactive-scheduler-binding --ignore-not-found=true

echo "Cleanup completed."
