import pytest
from module5_pod_scheduling.scheduler.controller import ProactiveSchedulingController


def test_scheduler_dry_run_binding():
    controller = ProactiveSchedulingController(dry_run=True)
    # Testing binding call in dry_run / simulation mode
    success = controller.bind_pod(
        pod_name="demo-workload-pod-xyz",
        namespace="proactive-system",
        node_name="worker-2",
    )
    assert success is True


def test_scheduler_cycle_simulation():
    controller = ProactiveSchedulingController(dry_run=True)
    nodes = controller.node_inspector.get_candidate_nodes()
    assert len(nodes) >= 2
    # Ensure ranker sorts candidate nodes and top node is selected
    forecast = {"cpu": 0.65, "memory": 0.55, "request_rate": 1500.0}
    ranked = controller.node_ranker.rank_nodes(nodes, future_forecast=forecast)
    assert len(ranked) > 0
    best_node = ranked[0]["name"]
    assert best_node in [n["name"] for n in nodes]
