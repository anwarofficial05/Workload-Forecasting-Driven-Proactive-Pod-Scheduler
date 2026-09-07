"""
Module 5: Pod Scheduling
Real Kubernetes proactive pod scheduler executing multi-objective placement decisions
and actual node binding via the Kubernetes API.
"""

from .scheduler.controller import ProactiveSchedulingController

__all__ = ["ProactiveSchedulingController"]
