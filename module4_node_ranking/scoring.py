import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger("module4.scoring")


class HeadroomScorer:
    """
    Calculates normalized CPU and Memory Headroom Scores based on
    FORECASTED future workload demand rather than just static current load.
    Higher headroom produces higher score (0.0 to 1.0).
    """

    @staticmethod
    def calculate_cpu_headroom_score(
        node_capacity_m: float,
        predicted_cpu_used_m: float,
        pod_request_m: float,
    ) -> float:
        if node_capacity_m <= 0:
            return 0.0
        # Expected remaining CPU millicores after future workload arrives and pod is placed
        remaining_m = node_capacity_m - (predicted_cpu_used_m + pod_request_m)
        headroom_ratio = remaining_m / node_capacity_m
        # Clamp strictly between 0.0 and 1.0
        return float(max(0.0, min(1.0, headroom_ratio)))

    @staticmethod
    def calculate_memory_headroom_score(
        node_capacity_mb: float,
        predicted_mem_used_mb: float,
        pod_request_mb: float,
    ) -> float:
        if node_capacity_mb <= 0:
            return 0.0
        # Expected remaining Memory MB after future workload arrives and pod is placed
        remaining_mb = node_capacity_mb - (predicted_mem_used_mb + pod_request_mb)
        headroom_ratio = remaining_mb / node_capacity_mb
        return float(max(0.0, min(1.0, headroom_ratio)))


class LatencyScorer:
    """
    Estimates expected latency impact using a documented, reproducible queuing theory proxy
    (Kleinrock's Conservation Law / M/M/1 approximation for CPU server queues):
      Expected Queuing Latency ~ Base_Latency / (1 - Predicted_Utilization)
    Nodes operating with healthy headroom prevent tail latency spikes.
    Normalized to 0.0 - 1.0 where 1.0 indicates minimum expected latency impact.
    """

    def __init__(self, base_latency_ms: float = 20.0, max_acceptable_latency_ms: float = 500.0):
        self.base_latency_ms = base_latency_ms
        self.max_acceptable_latency_ms = max_acceptable_latency_ms

    def calculate_latency_score(
        self,
        node_capacity_m: float,
        predicted_cpu_used_m: float,
        pod_request_m: float,
        node_measured_latency_ms: float = 25.0,
    ) -> float:
        if node_capacity_m <= 0:
            return 0.0

        post_utilization = (predicted_cpu_used_m + pod_request_m) / node_capacity_m
        # Cap utilization to 0.98 to avoid division by zero in queuing equation
        util_factor = min(0.98, max(0.05, post_utilization))

        # Queuing delay multiplier: 1 / (1 - rho)
        queuing_multiplier = 1.0 / (1.0 - util_factor)
        base = max(5.0, node_measured_latency_ms)
        estimated_latency = base * (0.5 + 0.5 * queuing_multiplier)

        # Normalize inversely: higher estimated latency -> lower score
        score = 1.0 - min(1.0, estimated_latency / self.max_acceptable_latency_ms)
        return float(max(0.0, min(1.0, score)))


class LoadBalanceScorer:
    """
    Measures cluster-wide load balancing impact.
    Evaluates how placing the pod on a candidate node alters the variance/dispersion
    of utilization across all worker nodes in the cluster.
    Nodes that minimize cluster load variance receive a score closer to 1.0.
    """

    @staticmethod
    def calculate_balance_score(
        candidate_node_name: str,
        all_nodes: List[Dict[str, Any]],
        pod_request_cpu_m: float,
    ) -> float:
        if not all_nodes:
            return 1.0

        # Compute projected post-placement CPU utilization for all nodes
        projected_utils = []
        for n in all_nodes:
            cap = float(n.get("cpu_capacity_millicores", 4000.0))
            pred = float(n.get("predicted_cpu_used_millicores", 1500.0))
            if n.get("name") == candidate_node_name:
                pred += pod_request_cpu_m
            util = pred / max(1.0, cap)
            projected_utils.append(util)

        # Standard deviation across cluster
        cluster_std = float(np.std(projected_utils))
        cluster_mean = float(np.mean(projected_utils))

        # Coefficient of variation (CV = std / mean) or standard deviation penalty
        # Ideal balance has std = 0.0 -> score = 1.0
        penalty = cluster_std * 2.0
        balance_score = max(0.0, 1.0 - penalty)
        return float(max(0.0, min(1.0, balance_score)))
