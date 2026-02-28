"""
compute_allocator.py — Tiered compute allocation (local vs cloud).

Decides WHERE each brain region runs based on available resources:
  - Fast path (local, <16ms): SensoryCortex, AmygdalaThalamus, MotorCortex
  - Slow path (local or cloud): Striatum training, PrefrontalCortex LLM
  - Overnight (cloud preferred): DreamLoop, Hippocampus consolidation

Monitors actual compute times and dynamically adjusts assignments.

Usage:
    from brain.scaling.compute_allocator import ComputeAllocator

    allocator = ComputeAllocator()
    plan = allocator.create_plan(available_resources)
    allocator.monitor(brain)
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ResourceProfile:
    """Available compute resources."""
    cpu_cores: int = 4
    gpu_available: bool = False
    gpu_vram_gb: float = 0.0
    ram_gb: float = 8.0
    cloud_available: bool = False
    cloud_gpu: bool = False
    network_latency_ms: float = 50.0  # Latency to cloud


@dataclass
class RegionAssignment:
    """Compute assignment for a single brain region."""
    region_name: str
    tier: str          # "fast_local", "slow_local", "slow_cloud", "overnight_cloud"
    budget_ms: float
    gpu_required: bool = False
    notes: str = ""


class ComputeAllocator:
    """
    Tiered compute allocation for brain regions.

    Assigns each region to a compute tier based on latency requirements
    and available resources.
    """

    # Regions that MUST run locally for real-time (<16ms)
    FAST_PATH_REGIONS = {"sensory_cortex", "amygdala_thalamus", "motor_cortex"}

    # Regions that benefit from GPU but can run on CPU
    GPU_PREFERRED = {"striatum", "basal_ganglia"}

    # Regions that can tolerate high latency
    CLOUD_OK = {"prefrontal_cortex", "hippocampus"}

    # Overnight-only regions
    OVERNIGHT = {"dream_loop"}

    def __init__(self):
        self._timing_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._budget_violations: Dict[str, int] = defaultdict(int)
        self._current_plan: Dict[str, RegionAssignment] = {}

    def create_plan(
        self,
        resources: Optional[ResourceProfile] = None,
    ) -> Dict[str, RegionAssignment]:
        """
        Create a compute allocation plan based on available resources.

        Returns a dict mapping region names to their assignments.
        """
        if resources is None:
            resources = ResourceProfile()

        plan = {}

        # ── Fast path: always local ───────────────────────────────────
        for region in self.FAST_PATH_REGIONS:
            plan[region] = RegionAssignment(
                region_name=region,
                tier="fast_local",
                budget_ms=16.7,
                notes="Must maintain 60fps, always local",
            )

        # ── GPU-preferred: local GPU or local CPU ─────────────────────
        for region in self.GPU_PREFERRED:
            if resources.gpu_available:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="slow_local",
                    budget_ms=100.0,
                    gpu_required=True,
                    notes=f"Using local GPU ({resources.gpu_vram_gb:.1f}GB VRAM)",
                )
            elif resources.cloud_gpu and resources.network_latency_ms < 100:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="slow_cloud",
                    budget_ms=200.0,
                    gpu_required=True,
                    notes=f"Offloaded to cloud GPU (latency ~{resources.network_latency_ms:.0f}ms)",
                )
            else:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="slow_local",
                    budget_ms=500.0,
                    notes="CPU-only, budget relaxed",
                )

        # ── Cloud-OK: local or cloud depending on availability ────────
        for region in self.CLOUD_OK:
            if resources.cloud_available and resources.network_latency_ms < 200:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="slow_cloud",
                    budget_ms=1000.0,
                    notes="Offloaded to cloud (latency acceptable)",
                )
            else:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="slow_local",
                    budget_ms=1000.0,
                    notes="Running locally",
                )

        # ── Overnight: cloud if available ─────────────────────────────
        for region in self.OVERNIGHT:
            if resources.cloud_gpu:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="overnight_cloud",
                    budget_ms=float("inf"),
                    gpu_required=True,
                    notes="Overnight processing on cloud GPU",
                )
            else:
                plan[region] = RegionAssignment(
                    region_name=region,
                    tier="overnight_local",
                    budget_ms=float("inf"),
                    notes="Overnight processing on local CPU",
                )

        self._current_plan = plan
        return plan

    def record_timing(self, region_name: str, elapsed_ms: float) -> None:
        """Record actual compute time for a region."""
        self._timing_history[region_name].append(elapsed_ms)

        # Check budget violations
        if region_name in self._current_plan:
            budget = self._current_plan[region_name].budget_ms
            if elapsed_ms > budget:
                self._budget_violations[region_name] += 1

    def time_region(self, region_name: str):
        """Context manager to time a region's processing."""
        return _RegionTimer(self, region_name)

    def stats(self) -> Dict[str, Any]:
        result = {}
        for region, history in self._timing_history.items():
            if history:
                times = list(history)
                result[region] = {
                    "avg_ms": round(np.mean(times), 2),
                    "p95_ms": round(np.percentile(times, 95), 2),
                    "max_ms": round(max(times), 2),
                    "violations": self._budget_violations.get(region, 0),
                    "tier": self._current_plan.get(region, RegionAssignment(region, "unknown", 0)).tier,
                }
        return result


class _RegionTimer:
    """Context manager for timing brain region processing."""

    def __init__(self, allocator: ComputeAllocator, region_name: str):
        self._allocator = allocator
        self._region = region_name
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._allocator.record_timing(self._region, elapsed_ms)
