import os
import shutil
import tempfile
import pytest
import pandas as pd
import numpy as np

from module1_data_collection.prometheus_client import PrometheusClient
from module1_data_collection.data_storage import MetricsStorage
from module1_data_collection.metrics_collector import MetricsCollector


def test_prometheus_client_query_builder():
    client = PrometheusClient(base_url="http://localhost:9090", timeout=1)
    assert client.base_url == "http://localhost:9090"
    assert client.query_endpoint == "http://localhost:9090/api/v1/query"
    assert client.query_range_endpoint == "http://localhost:9090/api/v1/query_range"


def test_data_storage_initialization_and_append():
    temp_dir = tempfile.mkdtemp()
    try:
        storage = MetricsStorage(storage_dir=temp_dir, csv_filename="test_metrics.csv")
        assert os.path.exists(storage.csv_path)

        record = {
            "timestamp": 1700000000.0,
            "node_name": "test-node",
            "pod_name": "test-pod",
            "cpu_usage": 45.2,
            "memory_usage": 55.0,
            "request_rate": 1200.0,
            "response_latency": 0.045,
            "node_cpu_capacity": 4000.0,
            "node_memory_capacity": 8192.0,
            "pod_cpu_requests": 250.0,
            "pod_memory_requests": 512.0,
            "node_readiness": True,
            "pod_count": 3,
            "workload_type": "normal",
        }
        storage.append_record(record)

        df = storage.load_data()
        assert len(df) == 1
        assert df["cpu_usage"].iloc[0] == 45.2
        assert df["node_name"].iloc[0] == "test-node"
    finally:
        shutil.rmtree(temp_dir)


def test_synthetic_data_generation():
    temp_dir = tempfile.mkdtemp()
    try:
        storage = MetricsStorage(storage_dir=temp_dir, csv_filename="synth_metrics.csv")
        df = storage.generate_synthetic_workload_history(num_samples=50, nodes=["w1", "w2"])
        assert len(df) == 100 # 50 samples * 2 nodes
        assert "cpu_usage" in df.columns
        assert "memory_usage" in df.columns
        assert "request_rate" in df.columns
        assert "response_latency" in df.columns
        assert set(df["node_name"].unique()) == {"w1", "w2"}
    finally:
        shutil.rmtree(temp_dir)


def test_metrics_collector_collect_once():
    temp_dir = tempfile.mkdtemp()
    try:
        storage = MetricsStorage(storage_dir=temp_dir)
        collector = MetricsCollector(storage=storage)
        snap = collector.collect_once()
        assert "cpu_usage" in snap
        assert "memory_usage" in snap
        assert "request_rate" in snap
        assert "response_latency" in snap
        df = storage.load_data()
        assert len(df) >= 1
    finally:
        shutil.rmtree(temp_dir)
