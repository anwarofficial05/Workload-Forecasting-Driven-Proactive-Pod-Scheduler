import time
import math
import os
import psutil
from typing import Optional
from fastapi import FastAPI, Query, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

app = FastAPI(
    title="Kubernetes Workload Demo Microservice",
    description="Microservice providing configurable CPU/Memory workload and Prometheus instrumentation for proactive scheduling",
    version="1.0.0",
)

# Prometheus Metrics Instrumentation
REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests received by demo service",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
ACTIVE_REQUESTS = Gauge(
    "http_active_requests",
    "Number of active concurrent requests",
)
PROCESS_CPU_PERCENT = Gauge(
    "app_process_cpu_percent",
    "Current process CPU usage percentage",
)
PROCESS_MEMORY_MB = Gauge(
    "app_process_memory_mb",
    "Current process resident memory in megabytes",
)

# Global memory holder to simulate stateful/cache memory consumption
_MEMORY_CACHE = []


def burn_cpu(duration_ms: float):
    """Burn CPU cycles deterministically for duration_ms milliseconds."""
    if duration_ms <= 0:
        return
    end_time = time.perf_counter() + (duration_ms / 1000.0)
    # Busy-spin computation with trigonometric calculations
    val = 1.0
    while time.perf_counter() < end_time:
        val = math.sin(val) * math.cos(val) + math.sqrt(abs(val) + 1.0)


def consume_memory(megabytes: int):
    """Allocate and retain memory in megabytes to simulate memory pressure."""
    global _MEMORY_CACHE
    if megabytes > 0:
        # Allocate bytes (approx 1MB chunks)
        chunk = bytearray(1024 * 1024)
        for _ in range(min(megabytes, 256)): # Cap at 256MB to avoid OOMKill on test pods
            _MEMORY_CACHE.append(chunk)
            if len(_MEMORY_CACHE) > 500:
                _MEMORY_CACHE.pop(0)


@app.get("/")
def root(
    cpu_ms: float = Query(20.0, description="CPU burn duration in milliseconds"),
    mem_mb: int = Query(0, description="Memory consumption in megabytes"),
):
    """
    Main workload endpoint: produces measurable CPU and Memory load.
    Returns operational metadata.
    """
    start_time = time.perf_counter()
    ACTIVE_REQUESTS.inc()
    try:
        burn_cpu(cpu_ms)
        if mem_mb > 0:
            consume_memory(mem_mb)

        elapsed = time.perf_counter() - start_time
        REQUEST_COUNTER.labels(endpoint="/", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/").observe(elapsed)

        proc = psutil.Process(os.getpid())
        mem_info = proc.memory_info()
        PROCESS_MEMORY_MB.set(mem_info.rss / (1024 * 1024))
        PROCESS_CPU_PERCENT.set(proc.cpu_percent())

        return {
            "status": "healthy",
            "message": "Workload processed successfully",
            "cpu_burn_ms": cpu_ms,
            "memory_allocated_mb": mem_mb,
            "latency_ms": round(elapsed * 1000.0, 2),
            "pod_name": os.getenv("HOSTNAME", "local-pod"),
            "node_name": os.getenv("NODE_NAME", "unknown-node"),
            "timestamp": time.time(),
        }
    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    REQUEST_COUNTER.labels(endpoint="/health", status="200").inc()
    return {"status": "UP", "timestamp": time.time()}


@app.get("/workload")
def workload_trigger(
    intensity: str = Query("moderate", description="Workload intensity: low, moderate, high, burst"),
):
    """
    Preset workload generator for easy test triggers.
    """
    profiles = {
        "low": {"cpu_ms": 10.0, "mem_mb": 5},
        "moderate": {"cpu_ms": 45.0, "mem_mb": 20},
        "high": {"cpu_ms": 120.0, "mem_mb": 50},
        "burst": {"cpu_ms": 250.0, "mem_mb": 80},
    }
    profile = profiles.get(intensity.lower(), profiles["moderate"])
    return root(cpu_ms=profile["cpu_ms"], mem_mb=profile["mem_mb"])


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
