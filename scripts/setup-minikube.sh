#!/usr/bin/env bash
set -e

CLUSTER_NAME="proactive-cluster"
NODES=3
CPUS=2
MEMORY="3072MB"

echo "============================================================"
echo " Starting Minikube Multi-Node Cluster: ${CLUSTER_NAME}"
echo " Nodes: ${NODES}, CPUs: ${CPUS}, Memory: ${MEMORY}"
echo "============================================================"

minikube start \
  -p ${CLUSTER_NAME} \
  --nodes=${NODES} \
  --cpus=${CPUS} \
  --memory=${MEMORY} \
  --driver=docker

echo "Enabling required Minikube addons..."
minikube addons enable metrics-server -p ${CLUSTER_NAME}
minikube addons enable default-storageclass -p ${CLUSTER_NAME}

echo "Cluster worker node topology:"
kubectl get nodes -o wide

echo "Minikube multi-node cluster is ready!"
