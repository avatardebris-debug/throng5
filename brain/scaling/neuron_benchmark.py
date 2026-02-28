"""
neuron_benchmark.py — Throng 2 vs Throng 5 neuron budget comparison.

Counts the total "neurons" (parameters) used across all brain regions
and compares efficiency metrics against the Throng 2 baseline.

Metrics tracked:
  - Total parameters (weights + biases)
  - Active parameters (non-zero after pruning)
  - Parameters per action (how efficient is the action selection)
  - Replay buffer memory usage
  - Heuristic table size

Usage:
    from brain.scaling.neuron_benchmark import NeuronBenchmark

    bench = NeuronBenchmark(brain)
    report = bench.full_report()
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class NeuronBenchmark:
    """
    Count and compare neuron/parameter budgets across brain regions.
    """

    # Throng 2 baseline (from original tabular Q-learning implementation)
    THRONG2_BASELINE = {
        "total_params": 2048,          # Q-table slots: 128 states × 16 actions
        "replay_buffer_items": 1000,
        "heuristics": 0,
        "regions": 1,                  # Single monolithic agent
        "games_supported": 1,
    }

    def __init__(self, brain):
        self.brain = brain

    def count_params(self) -> Dict[str, int]:
        """Count parameters per brain region."""
        counts = {}

        # Striatum DQN
        s = self.brain.striatum
        counts["striatum"] = (
            s._W1.size + s._b1.size + s._W2.size + s._b2.size
        )

        # Amygdala threat model
        a = self.brain.amygdala
        counts["amygdala"] = (
            a._W1.size + a._b1.size + a._W2.size + a._b2.size
        )

        # Motor Cortex (heuristic table counts as params)
        counts["motor_cortex"] = len(self.brain.motor._heuristics) * 2

        # Basal Ganglia (SNN params if present)
        counts["basal_ganglia"] = 0  # Placeholder until SNN is integrated

        # Hippocampus (replay buffer)
        hippo = self.brain.hippocampus
        transitions = hippo._transitions
        if transitions:
            sample = transitions[0]
            if len(sample) >= 5:
                per_item = len(np.asarray(sample[0]).flatten()) * 2 + 3
            else:
                per_item = 100
            counts["hippocampus_buffer"] = len(transitions) * per_item
        else:
            counts["hippocampus_buffer"] = 0

        # Prefrontal (hypothesis storage)
        counts["prefrontal"] = len(self.brain.prefrontal._strategies) * 50

        # Sensory Cortex (no learned params, just adapter)
        counts["sensory_cortex"] = 0

        return counts

    def active_params(self) -> Dict[str, int]:
        """Count non-zero (active) parameters per region."""
        counts = {}

        s = self.brain.striatum
        counts["striatum"] = (
            np.count_nonzero(s._W1) + np.count_nonzero(s._b1)
            + np.count_nonzero(s._W2) + np.count_nonzero(s._b2)
        )

        a = self.brain.amygdala
        counts["amygdala"] = (
            np.count_nonzero(a._W1) + np.count_nonzero(a._b1)
            + np.count_nonzero(a._W2) + np.count_nonzero(a._b2)
        )

        counts["motor_cortex"] = len(self.brain.motor._heuristics) * 2
        return counts

    def memory_usage_kb(self) -> Dict[str, float]:
        """Estimate memory usage in KB per region."""
        usage = {}

        s = self.brain.striatum
        usage["striatum_weights"] = round(
            (s._W1.nbytes + s._b1.nbytes + s._W2.nbytes + s._b2.nbytes) / 1024, 2
        )
        usage["striatum_buffer"] = round(
            len(s._replay) * 200 / 1024, 2  # ~200 bytes per transition estimate
        )

        a = self.brain.amygdala
        usage["amygdala_weights"] = round(
            (a._W1.nbytes + a._b1.nbytes + a._W2.nbytes + a._b2.nbytes) / 1024, 2
        )

        usage["hippocampus_buffer"] = round(
            len(self.brain.hippocampus._transitions) * 200 / 1024, 2
        )

        usage["motor_heuristics"] = round(
            len(self.brain.motor._heuristics) * 16 / 1024, 2
        )

        usage["total_kb"] = round(sum(usage.values()), 2)
        return usage

    def efficiency_metrics(self) -> Dict[str, float]:
        """Compute efficiency ratios."""
        total_params = sum(self.count_params().values())
        active = sum(self.active_params().values())

        return {
            "total_params": total_params,
            "active_params": active,
            "sparsity": round(1 - active / max(total_params, 1), 4),
            "params_per_action": round(total_params / max(self.brain.n_actions, 1), 1),
            "params_per_feature": round(total_params / max(self.brain.n_features, 1), 1),
            "steps_per_param": round(
                self.brain._step_count / max(total_params, 1), 4
            ),
        }

    def vs_throng2(self) -> Dict[str, Any]:
        """
        Compare against Throng 2 baseline.

        Returns improvement/degradation ratios.
        """
        t5_params = sum(self.count_params().values())
        t2_params = self.THRONG2_BASELINE["total_params"]

        return {
            "throng2_params": t2_params,
            "throng5_params": t5_params,
            "param_ratio": round(t5_params / max(t2_params, 1), 2),
            "throng2_regions": self.THRONG2_BASELINE["regions"],
            "throng5_regions": 7,
            "throng2_games": self.THRONG2_BASELINE["games_supported"],
            "throng5_games": "unlimited (via ROM factory)",
            "throng5_advantages": [
                "Multi-game support via UniversalAdapter",
                f"18 swappable RL algorithms (vs 1)",
                "Overnight dream consolidation",
                "Distributed compute support",
                f"7 specialized brain regions (vs 1 monolithic agent)",
                "Curiosity-driven exploration",
                "Serializable checkpoints",
            ],
        }

    def full_report(self) -> Dict[str, Any]:
        """Generate comprehensive neuron budget report."""
        return {
            "params_by_region": self.count_params(),
            "active_params": self.active_params(),
            "memory_kb": self.memory_usage_kb(),
            "efficiency": self.efficiency_metrics(),
            "vs_throng2": self.vs_throng2(),
        }
