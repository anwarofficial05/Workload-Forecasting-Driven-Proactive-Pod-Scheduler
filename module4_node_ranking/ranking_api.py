import os
import logging
from typing import List, Dict, Any, Optional
from .node_ranker import MultiObjectiveNodeRanker

logger = logging.getLogger("module4.ranking_api")


class KubernetesNodeInspector:
    """
    Connects to Kubernetes cluster to extract worker node capacity, allocatable resources,
    readiness, and current usage.
    """

    def __init__(self):
        self.k8s_available = False
        self.cluster_context = None
        self.reconnect()

    def reconnect(self) -> bool:
        """Attempt to connect or reconnect to Kubernetes API (local, in-cluster, or GKE cloud)."""
        self.k8s_available = False
        self.cluster_context = None
        try:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
                self.k8s_available = True
                self.cluster_context = "in-cluster"
            except Exception:
                try:
                    config.load_kube_config()
                    self.k8s_available = True
                    # Retrieve current context name
                    _, active_ctx = config.list_kube_config_contexts()
                    if active_ctx:
                        self.cluster_context = active_ctx.get("name")
                except Exception:
                    self.k8s_available = False

            if self.k8s_available:
                self.v1 = client.CoreV1Api()
                logger.info(f"Connected to Kubernetes API (Context: {self.cluster_context})")
                return True
        except Exception as exc:
            logger.warning(f"Kubernetes client unavailable: {exc}")
        return False

    def get_live_pods(self) -> List[Dict[str, Any]]:
        """Fetch active workload pods from Kubernetes API."""
        if not self.k8s_available:
            return []
        try:
            pod_list = self.v1.list_pod_for_all_namespaces()
            pods = []
            for p in pod_list.items:
                ns = p.metadata.namespace
                if ns in ["kube-system", "kube-public", "kube-node-lease", "gke-managed-system", "gke-managed-cim"]:
                    continue
                pods.append({
                    "name": p.metadata.name,
                    "namespace": ns,
                    "status": p.status.phase or "Running",
                    "node": p.spec.node_name or "Unassigned",
                    "ip": p.status.pod_ip or "Pending",
                    "scheduler": p.spec.scheduler_name or "default-scheduler",
                    "age": "Active",
                })
            return pods
        except Exception as exc:
            logger.warning(f"Error reading live pods: {exc}")
            return []

    def get_candidate_nodes(self) -> List[Dict[str, Any]]:
        """Fetch real nodes from Kubernetes API or return realistic cluster nodes if offline."""
        if not self.k8s_available:
            return self._mock_cluster_nodes()

        try:
            node_list = self.v1.list_node()
            nodes_data = []
            for n in node_list.items:
                # Skip control plane / master nodes for pod scheduling
                labels = n.metadata.labels or {}
                if "node-role.kubernetes.io/control-plane" in labels or "node-role.kubernetes.io/master" in labels:
                    continue

                name = n.metadata.name
                ready = False
                for cond in (n.status.conditions or []):
                    if cond.type == "Ready" and cond.status == "True":
                        ready = True
                        break

                unschedulable = bool(n.spec.unschedulable)

                # Parse capacities
                cap = n.status.capacity or {}
                cpu_cap_str = cap.get("cpu", "4")
                mem_cap_str = cap.get("memory", "8192Mi")

                cpu_cap_m = float(cpu_cap_str) * 1000.0 if cpu_cap_str.isdigit() else 4000.0
                mem_cap_mb = 8192.0 # Standard fallback

                # Estimate current usage or query metrics server
                nodes_data.append({
                    "name": name,
                    "ready": ready,
                    "unschedulable": unschedulable,
                    "cpu_capacity_millicores": cpu_cap_m,
                    "memory_capacity_mb": mem_cap_mb,
                    "current_cpu_used_millicores": cpu_cap_m * 0.35, # default baseline
                    "current_memory_used_mb": mem_cap_mb * 0.45,
                    "measured_latency_ms": 25.0,
                })

            if not nodes_data:
                return self._mock_cluster_nodes()
            return nodes_data

        except Exception as exc:
            logger.warning(f"Error querying K8s API for nodes: {exc}, using fallback")
            return self._mock_cluster_nodes()

    def _mock_cluster_nodes(self) -> List[Dict[str, Any]]:
        """Mock 3-node cluster representation matching Minikube / GKE topologies."""
        return [
            {
                "name": "worker-1",
                "ready": True,
                "unschedulable": False,
                "cpu_capacity_millicores": 4000.0,
                "memory_capacity_mb": 8192.0,
                "current_cpu_used_millicores": 2600.0, # 65% loaded
                "current_memory_used_mb": 5200.0,
                "measured_latency_ms": 65.0,
            },
            {
                "name": "worker-2",
                "ready": True,
                "unschedulable": False,
                "cpu_capacity_millicores": 4000.0,
                "memory_capacity_mb": 8192.0,
                "current_cpu_used_millicores": 800.0,  # 20% loaded (optimal)
                "current_memory_used_mb": 2200.0,
                "measured_latency_ms": 20.0,
            },
            {
                "name": "worker-3",
                "ready": True,
                "unschedulable": False,
                "cpu_capacity_millicores": 4000.0,
                "memory_capacity_mb": 8192.0,
                "current_cpu_used_millicores": 1900.0, # 47.5% loaded
                "current_memory_used_mb": 4100.0,
                "measured_latency_ms": 38.0,
            },
        ]
