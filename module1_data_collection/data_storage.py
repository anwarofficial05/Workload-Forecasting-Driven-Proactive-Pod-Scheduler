import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("module1.data_storage")

SCHEMA_COLUMNS = [
    "timestamp",
    "node_name",
    "pod_name",
    "cpu_usage",
    "memory_usage",
    "request_rate",
    "response_latency",
    "node_cpu_capacity",
    "node_memory_capacity",
    "pod_cpu_requests",
    "pod_memory_requests",
    "node_readiness",
    "pod_count",
    "workload_type",
]


class MetricsStorage:
    """
    Persistent storage manager for time-series cluster metrics.
    Supports CSV and Apache Parquet formats.
    """

    def __init__(self, storage_dir: str = "data", csv_filename: str = "historical_workload_metrics.csv"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.csv_path = os.path.join(self.storage_dir, csv_filename)
        self.parquet_path = os.path.join(
            self.storage_dir, csv_filename.replace(".csv", ".parquet")
        )
        self._initialize_storage()

    def _initialize_storage(self):
        """Ensure file exists with header if brand new."""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame(columns=SCHEMA_COLUMNS)
            df.to_csv(self.csv_path, index=False)
            logger.info(f"Initialized new metrics file at {self.csv_path}")

    def append_record(self, record: Dict[str, Any]):
        """Append a single structured record."""
        self.append_records([record])

    def append_records(self, records: List[Dict[str, Any]]):
        """Append batch of records to CSV (and Parquet if pyarrow available)."""
        if not records:
            return
        df = pd.DataFrame(records)
        # Ensure all schema columns exist
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        df = df[SCHEMA_COLUMNS]

        # Append to CSV
        df.to_csv(self.csv_path, mode="a", header=not os.path.exists(self.csv_path), index=False)
        try:
            # Sync to parquet
            full_df = self.load_data()
            full_df.to_parquet(self.parquet_path, index=False)
        except Exception as exc:
            logger.debug(f"Parquet export skipped: {exc}")

    def load_data(self) -> pd.DataFrame:
        """Load entire dataset as Pandas DataFrame sorted chronologically."""
        if os.path.exists(self.parquet_path):
            try:
                df = pd.read_parquet(self.parquet_path)
                return df.sort_values(by="timestamp").reset_index(drop=True)
            except Exception:
                pass

        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            df = pd.read_csv(self.csv_path)
            if not df.empty and "timestamp" in df.columns:
                df = df.sort_values(by="timestamp").reset_index(drop=True)
            return df

        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    def generate_synthetic_workload_history(
        self,
        num_samples: int = 1500,
        interval_seconds: int = 10,
        nodes: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate realistic multi-node Kubernetes workload historical dataset
        simulating diurnal traffic, sudden spikes, bursty behavior, and periodic oscillations.
        Crucial for training models and validating offline tests.
        """
        if nodes is None:
            nodes = ["worker-1", "worker-2", "worker-3"]

        np.random.seed(42)
        base_time = 1700000000.0 # Realistic epoch
        records = []

        for i in range(num_samples):
            t = base_time + (i * interval_seconds)
            # Diurnal & cyclical pattern (simulates hourly / daily cycle)
            cycle = np.sin(2 * np.pi * i / 120.0)
            burst = 0.0
            workload_type = "normal"

            # Inject periodic spikes & bursts
            if 300 <= i <= 360 or 750 <= i <= 800 or 1200 <= i <= 1260:
                burst = np.random.uniform(30.0, 50.0)
                workload_type = "spike"
            elif i % 50 in [10, 11, 12]:
                burst = np.random.uniform(15.0, 30.0)
                workload_type = "burst"

            # Aggregate request rate
            req_rate = max(50.0, 800.0 + (350.0 * cycle) + (burst * 40.0) + np.random.normal(0, 20.0))
            # Latency follows request rate with non-linear saturation
            latency = max(0.015, 0.045 + (req_rate / 15000.0) + (0.15 if burst > 0 else 0.0) + np.random.normal(0, 0.005))

            # Distribute among nodes with slight heterogeneity
            for node_idx, node_name in enumerate(nodes):
                bias = (node_idx - 1) * 5.0
                node_cpu = max(5.0, min(95.0, 35.0 + (20.0 * cycle) + burst + bias + np.random.normal(0, 3.0)))
                node_mem = max(20.0, min(90.0, 45.0 + (12.0 * cycle) + (burst * 0.4) + (bias * 0.5) + np.random.normal(0, 1.5)))

                record = {
                    "timestamp": t,
                    "node_name": node_name,
                    "pod_name": f"workload-demo-app-{node_name}-{i % 5}",
                    "cpu_usage": round(node_cpu, 2),
                    "memory_usage": round(node_mem, 2),
                    "request_rate": round(req_rate / len(nodes), 2),
                    "response_latency": round(latency, 4),
                    "node_cpu_capacity": 4000.0, # 4000m (4 vCPU)
                    "node_memory_capacity": 8192.0, # 8192 MiB
                    "pod_cpu_requests": 250.0, # 250m
                    "pod_memory_requests": 512.0, # 512 MiB
                    "node_readiness": True,
                    "pod_count": 4 + (node_idx % 3),
                    "workload_type": workload_type,
                }
                records.append(record)

        synth_df = pd.DataFrame(records)
        synth_df.to_csv(self.csv_path, index=False)
        try:
            synth_df.to_parquet(self.parquet_path, index=False)
        except Exception:
            pass
        logger.info(f"Generated {len(records)} realistic training samples to {self.csv_path}")
        return synth_df
