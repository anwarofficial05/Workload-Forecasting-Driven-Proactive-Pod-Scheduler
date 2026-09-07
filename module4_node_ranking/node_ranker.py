import logging
from typing import List, Dict, Any, Optional
from .node_filter import NodeFilter
from .scoring import HeadroomScorer, LatencyScorer, LoadBalanceScorer

logger = logging.getLogger("module4.node_ranker")


class MultiObjectiveNodeRanker:
    """
    Core Research Contribution:
    Multi-Objective Node Ranking engine combining:
    1. Predicted CPU Headroom Score (weight: 0.30)
    2. Predicted Memory Headroom Score (weight: 0.25)
    3. Expected Latency Impact Score (weight: 0.20)
    4. Cluster-wide Load Balance Score (weight: 0.25)
    Formula:
      NODE_SCORE = (w_cpu * S_cpu) + (w_mem * S_mem) + (w_lat * S_lat) + (w_bal * S_bal)
    """

    def __init__(
        self,
        cpu_weight: float = 0.30,
        memory_weight: float = 0.25,
        latency_weight: float = 0.20,
        balance_weight: float = 0.25,
        slo_latency_ms: float = 500.0,
    ):
        total = cpu_weight + memory_weight + latency_weight + balance_weight
        if abs(total - 1.0) > 1e-4:
            self.cpu_weight = cpu_weight / total
            self.memory_weight = memory_weight / total
            self.latency_weight = latency_weight / total
            self.balance_weight = balance_weight / total
        else:
            self.cpu_weight = cpu_weight
            self.memory_weight = memory_weight
            self.latency_weight = latency_weight
            self.balance_weight = balance_weight

        self.filter = NodeFilter()
        self.headroom_scorer = HeadroomScorer()
        self.latency_scorer = LatencyScorer(max_acceptable_latency_ms=slo_latency_ms)
        self.balance_scorer = LoadBalanceScorer()

    def rank_nodes(
        self,
        candidate_nodes: List[Dict[str, Any]],
        future_forecast: Optional[Dict[str, Any]] = None,
        pod_cpu_request_m: float = 250.0,
        pod_mem_request_mb: float = 512.0,
    ) -> List[Dict[str, Any]]:
        """
        Filters candidate nodes, applies forecasted workload shifts,
        computes multi-objective scores, and returns sorted ranked nodes.
        """
        # 1. Node Filtering
        eligible_nodes, filtered_out = self.filter.filter_nodes(
            candidate_nodes,
            pod_cpu_request_millicores=pod_cpu_request_m,
            pod_memory_request_mb=pod_mem_request_mb,
        )

        if not eligible_nodes:
            logger.warning("No nodes passed the eligibility filter!")
            return []

        # Extract forecasted ratios if provided
        forecast_cpu_ratio = future_forecast.get("cpu", None) if future_forecast else None
        forecast_mem_ratio = future_forecast.get("memory", None) if future_forecast else None

        # Prepare projected values for eligible nodes
        for node in eligible_nodes:
            cap_cpu = float(node.get("cpu_capacity_millicores", 4000.0))
            cap_mem = float(node.get("memory_capacity_mb", 8192.0))
            curr_cpu_m = float(node.get("current_cpu_used_millicores", 1200.0))
            curr_mem_mb = float(node.get("current_memory_used_mb", 3200.0))

            # If global forecast is provided, incorporate forecast shift
            if forecast_cpu_ratio is not None:
                # Weighted combination of node current usage and cluster predicted usage
                pred_cpu_m = max(curr_cpu_m, (forecast_cpu_ratio * cap_cpu * 0.7) + (curr_cpu_m * 0.3))
            else:
                pred_cpu_m = curr_cpu_m

            if forecast_mem_ratio is not None:
                pred_mem_mb = max(curr_mem_mb, (forecast_mem_ratio * cap_mem * 0.7) + (curr_mem_mb * 0.3))
            else:
                pred_mem_mb = curr_mem_mb

            node["predicted_cpu_used_millicores"] = round(pred_cpu_m, 2)
            node["predicted_memory_used_mb"] = round(pred_mem_mb, 2)

        # 2. Score Each Eligible Node
        scored_nodes = []
        for node in eligible_nodes:
            name = node.get("name")
            cap_cpu = float(node["cpu_capacity_millicores"])
            cap_mem = float(node["memory_capacity_mb"])
            curr_cpu_m = float(node.get("current_cpu_used_millicores", 1200.0))
            curr_mem_mb = float(node.get("current_memory_used_mb", 3200.0))
            pred_cpu_m = float(node["predicted_cpu_used_millicores"])
            pred_mem_mb = float(node["predicted_memory_used_mb"])
            meas_lat_ms = float(node.get("measured_latency_ms", 25.0))

            # 2.1 CPU Headroom Score
            s_cpu = self.headroom_scorer.calculate_cpu_headroom_score(
                cap_cpu, pred_cpu_m, pod_cpu_request_m
            )

            # 2.2 Memory Headroom Score
            s_mem = self.headroom_scorer.calculate_memory_headroom_score(
                cap_mem, pred_mem_mb, pod_mem_request_mb
            )

            # 2.3 Latency Score
            s_lat = self.latency_scorer.calculate_latency_score(
                cap_cpu, pred_cpu_m, pod_cpu_request_m, node_measured_latency_ms=meas_lat_ms
            )

            # 2.4 Load Balance Score
            s_bal = self.balance_scorer.calculate_balance_score(
                name, eligible_nodes, pod_cpu_request_m
            )

            # 2.5 Multi-Objective Composite Score
            final_score = (
                (self.cpu_weight * s_cpu)
                + (self.memory_weight * s_mem)
                + (self.latency_weight * s_lat)
                + (self.balance_weight * s_bal)
            )

            scored_nodes.append({
                "name": name,
                "current_cpu": round(curr_cpu_m / cap_cpu, 4),
                "current_memory": round(curr_mem_mb / cap_mem, 4),
                "predicted_cpu": round(pred_cpu_m / cap_cpu, 4),
                "predicted_memory": round(pred_mem_mb / cap_mem, 4),
                "cpu_headroom": round(s_cpu, 4),
                "memory_headroom": round(s_mem, 4),
                "latency_score": round(s_lat, 4),
                "balance_score": round(s_bal, 4),
                "final_score": round(final_score, 4),
            })

        # 3. Sort Descending by Final Score
        scored_nodes.sort(key=lambda x: x["final_score"], reverse=True)

        # 4. Assign Ranks (1 = best)
        for rank_idx, item in enumerate(scored_nodes):
            item["rank"] = rank_idx + 1

        # Mandatory logging format per project specification:
        # [NODE RANKING] worker-1=0.62 worker-2=0.89 worker-3=0.74
        log_parts = [f"{item['name']}={item['final_score']:.2f}" for item in scored_nodes]
        log_msg = f"[NODE RANKING] {' '.join(log_parts)}"
        print(log_msg)
        logger.info(log_msg)

        return scored_nodes
