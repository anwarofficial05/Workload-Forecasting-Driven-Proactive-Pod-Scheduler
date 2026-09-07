import pytest
from module4_node_ranking.node_filter import NodeFilter
from module4_node_ranking.scoring import HeadroomScorer, LatencyScorer, LoadBalanceScorer
from module4_node_ranking.node_ranker import MultiObjectiveNodeRanker


def test_node_filter_readiness_and_capacity():
    nf = NodeFilter()
    candidate_nodes = [
        {
            "name": "healthy-node",
            "ready": True,
            "unschedulable": False,
            "cpu_capacity_millicores": 4000.0,
            "memory_capacity_mb": 8192.0,
            "current_cpu_used_millicores": 1000.0,
            "current_memory_used_mb": 2000.0,
        },
        {
            "name": "not-ready-node",
            "ready": False, # Should be filtered out
            "unschedulable": False,
            "cpu_capacity_millicores": 4000.0,
            "memory_capacity_mb": 8192.0,
            "current_cpu_used_millicores": 500.0,
            "current_memory_used_mb": 1000.0,
        },
        {
            "name": "cordoned-node",
            "ready": True,
            "unschedulable": True, # Should be filtered out
            "cpu_capacity_millicores": 4000.0,
            "memory_capacity_mb": 8192.0,
            "current_cpu_used_millicores": 500.0,
            "current_memory_used_mb": 1000.0,
        },
        {
            "name": "exhausted-cpu-node",
            "ready": True,
            "unschedulable": False,
            "cpu_capacity_millicores": 4000.0,
            "memory_capacity_mb": 8192.0,
            "current_cpu_used_millicores": 3900.0, # Available 100m < 250m requested
            "current_memory_used_mb": 2000.0,
        },
    ]

    eligible, filtered = nf.filter_nodes(candidate_nodes, pod_cpu_request_millicores=250.0, pod_memory_request_mb=512.0)
    assert len(eligible) == 1
    assert eligible[0]["name"] == "healthy-node"
    assert len(filtered) == 3


def test_multi_objective_scoring_bounds():
    hs = HeadroomScorer()
    # 4000m capacity, 1000m predicted, 250m pod req -> 2750m remaining -> 2750 / 4000 = 0.6875
    s_cpu = hs.calculate_cpu_headroom_score(4000.0, 1000.0, 250.0)
    assert 0.0 <= s_cpu <= 1.0
    assert abs(s_cpu - 0.6875) < 1e-4

    ls = LatencyScorer()
    s_lat = ls.calculate_latency_score(4000.0, 1000.0, 250.0)
    assert 0.0 <= s_lat <= 1.0

    lbs = LoadBalanceScorer()
    mock_nodes = [
        {"name": "n1", "cpu_capacity_millicores": 4000.0, "predicted_cpu_used_millicores": 1500.0},
        {"name": "n2", "cpu_capacity_millicores": 4000.0, "predicted_cpu_used_millicores": 1500.0},
    ]
    s_bal = lbs.calculate_balance_score("n1", mock_nodes, 250.0)
    assert 0.0 <= s_bal <= 1.0


def test_multi_objective_node_ranker_sorting_and_ranks():
    ranker = MultiObjectiveNodeRanker(
        cpu_weight=0.30,
        memory_weight=0.25,
        latency_weight=0.20,
        balance_weight=0.25,
    )

    candidate_nodes = [
        {
            "name": "worker-busy",
            "ready": True,
            "unschedulable": False,
            "cpu_capacity_millicores": 4000.0,
            "memory_capacity_mb": 8192.0,
            "current_cpu_used_millicores": 3200.0, # 80% used
            "current_memory_used_mb": 6500.0,
            "measured_latency_ms": 120.0,
        },
        {
            "name": "worker-idle",
            "ready": True,
            "unschedulable": False,
            "cpu_capacity_millicores": 4000.0,
            "memory_capacity_mb": 8192.0,
            "current_cpu_used_millicores": 600.0, # 15% used (optimal)
            "current_memory_used_mb": 1500.0,
            "measured_latency_ms": 18.0,
        },
    ]

    forecast = {"cpu": 0.50, "memory": 0.40, "request_rate": 1000.0}
    ranked = ranker.rank_nodes(candidate_nodes, future_forecast=forecast)

    assert len(ranked) == 2
    # worker-idle must have rank 1 and higher final score than worker-busy
    assert ranked[0]["name"] == "worker-idle"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["name"] == "worker-busy"
    assert ranked[1]["rank"] == 2
    assert ranked[0]["final_score"] > ranked[1]["final_score"]
