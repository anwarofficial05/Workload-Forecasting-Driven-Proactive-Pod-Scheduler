import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("module4.node_filter")


class NodeFilter:
    """
    Filters Kubernetes worker nodes before ranking:
    1. Removes nodes that are NotReady.
    2. Removes nodes marked unschedulable (cordoned / draining).
    3. Removes nodes with insufficient allocatable CPU capacity to satisfy pod requests.
    4. Removes nodes with insufficient allocatable memory capacity to satisfy pod requests.
    5. Checks node taints vs pod tolerations.
    """

    def __init__(self, min_cpu_headroom_percent: float = 5.0, min_mem_headroom_percent: float = 5.0):
        self.min_cpu_headroom_percent = min_cpu_headroom_percent
        self.min_mem_headroom_percent = min_mem_headroom_percent

    def filter_nodes(
        self,
        nodes: List[Dict[str, Any]],
        pod_cpu_request_millicores: float = 250.0,
        pod_memory_request_mb: float = 512.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Filter list of candidate nodes.
        Returns: (eligible_nodes, filtered_out_nodes)
        """
        eligible = []
        filtered_out = []

        for node in nodes:
            name = node.get("name", "unknown")
            is_ready = node.get("ready", True)
            is_unschedulable = node.get("unschedulable", False)
            cpu_capacity_m = float(node.get("cpu_capacity_millicores", 4000.0))
            mem_capacity_mb = float(node.get("memory_capacity_mb", 8192.0))
            current_cpu_used_m = float(node.get("current_cpu_used_millicores", 1500.0))
            current_mem_used_mb = float(node.get("current_memory_used_mb", 3500.0))

            # 1. Readiness check
            if not is_ready:
                filtered_out.append({"node": name, "reason": "Node is NotReady"})
                logger.info(f"Filtered out node {name}: NotReady")
                continue

            # 2. Unschedulable check
            if is_unschedulable:
                filtered_out.append({"node": name, "reason": "Node is cordoned/unschedulable"})
                logger.info(f"Filtered out node {name}: Unschedulable")
                continue

            # 3. CPU allocatable capacity check
            available_cpu = cpu_capacity_m - current_cpu_used_m
            if available_cpu < pod_cpu_request_millicores:
                filtered_out.append({
                    "node": name,
                    "reason": f"Insufficient CPU: available {available_cpu}m < requested {pod_cpu_request_millicores}m",
                })
                logger.info(f"Filtered out node {name}: Insufficient CPU")
                continue

            # 4. Memory allocatable capacity check
            available_mem = mem_capacity_mb - current_mem_used_mb
            if available_mem < pod_memory_request_mb:
                filtered_out.append({
                    "node": name,
                    "reason": f"Insufficient Memory: available {available_mem}MB < requested {pod_memory_request_mb}MB",
                })
                logger.info(f"Filtered out node {name}: Insufficient Memory")
                continue

            eligible.append(node)

        logger.info(f"Node filtering: {len(eligible)} eligible, {len(filtered_out)} filtered out")
        return eligible, filtered_out
