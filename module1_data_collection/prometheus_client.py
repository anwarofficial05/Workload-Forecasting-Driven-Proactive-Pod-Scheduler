import logging
import time
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger("module1.prometheus_client")


class PrometheusClient:
    """
    Reusable client for querying Prometheus REST API in Kubernetes clusters.
    Supports instant queries, range queries, and cluster node/pod metrics extraction.
    """

    def __init__(self, base_url: str = "http://localhost:9090", timeout: int = 5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.query_endpoint = f"{self.base_url}/api/v1/query"
        self.query_range_endpoint = f"{self.base_url}/api/v1/query_range"

    def is_connected(self) -> bool:
        """Check if Prometheus server is reachable and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/-/healthy", timeout=self.timeout)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug(f"Prometheus health check failed at {self.base_url}: {exc}")
            return False

    def query(self, promql: str) -> Optional[List[Dict[str, Any]]]:
        """Execute instant PromQL query."""
        try:
            resp = requests.get(
                self.query_endpoint,
                params={"query": promql},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") == "success":
                    return payload.get("data", {}).get("result", [])
            logger.warning(f"Prometheus query returned status {resp.status_code}: {resp.text}")
            return None
        except Exception as exc:
            logger.warning(f"Prometheus query failed for '{promql}': {exc}")
            return None

    def query_range(
        self, promql: str, start_time: float, end_time: float, step: str = "15s"
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute range PromQL query for historical metrics."""
        try:
            resp = requests.get(
                self.query_range_endpoint,
                params={
                    "query": promql,
                    "start": start_time,
                    "end": end_time,
                    "step": step,
                },
                timeout=self.timeout * 2,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") == "success":
                    return payload.get("data", {}).get("result", [])
            return None
        except Exception as exc:
            logger.warning(f"Prometheus query_range failed for '{promql}': {exc}")
            return None

    def get_cluster_snapshot(self) -> Dict[str, Any]:
        """
        Collect aggregate cluster-wide and per-node metrics.
        Returns normalized dictionary with CPU, Memory, Request Rate, and Latency.
        """
        snapshot = {
            "timestamp": time.time(),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "request_rate": 0.0,
            "response_latency": 0.0,
            "pod_count": 0,
            "nodes": {},
        }

        # 1. Total or Average Node CPU Usage (%)
        cpu_query = '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)'
        cpu_res = self.query(cpu_query)
        if cpu_res and len(cpu_res) > 0:
            try:
                val = float(cpu_res[0]["value"][1])
                snapshot["cpu_usage"] = max(0.0, min(100.0, val))
            except (ValueError, IndexError):
                pass
        else:
            # Fallback to container CPU usage
            c_cpu = self.query('sum(rate(container_cpu_usage_seconds_total{container!=""}[1m]))')
            if c_cpu and len(c_cpu) > 0:
                try:
                    snapshot["cpu_usage"] = float(c_cpu[0]["value"][1]) * 100.0
                except (ValueError, IndexError):
                    pass

        # 2. Memory Usage (%)
        mem_query = '(sum(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / sum(node_memory_MemTotal_bytes)) * 100'
        mem_res = self.query(mem_query)
        if mem_res and len(mem_res) > 0:
            try:
                val = float(mem_res[0]["value"][1])
                snapshot["memory_usage"] = max(0.0, min(100.0, val))
            except (ValueError, IndexError):
                pass

        # 3. Request Rate (req/sec)
        req_query = 'sum(rate(http_requests_total[1m]))'
        req_res = self.query(req_query)
        if req_res and len(req_res) > 0:
            try:
                snapshot["request_rate"] = max(0.0, float(req_res[0]["value"][1]))
            except (ValueError, IndexError):
                pass

        # 4. P95 Latency (seconds)
        lat_query = 'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))'
        lat_res = self.query(lat_query)
        if lat_res and len(lat_res) > 0:
            try:
                val = float(lat_res[0]["value"][1])
                if not (val != val): # check not NaN
                    snapshot["response_latency"] = max(0.0, val)
            except (ValueError, IndexError):
                pass

        # 5. Pod Count
        pod_query = 'count(kube_pod_info{namespace="proactive-system"})'
        pod_res = self.query(pod_query)
        if pod_res and len(pod_res) > 0:
            try:
                snapshot["pod_count"] = int(float(pod_res[0]["value"][1]))
            except (ValueError, IndexError):
                pass

        # 6. Per-node utilization
        per_node_cpu = self.query('100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)')
        if per_node_cpu:
            for item in per_node_cpu:
                node = item.get("metric", {}).get("instance", "unknown")
                try:
                    snapshot["nodes"].setdefault(node, {})["cpu_usage"] = float(item["value"][1])
                except (ValueError, IndexError):
                    pass

        return snapshot
