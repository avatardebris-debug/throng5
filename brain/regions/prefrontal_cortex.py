"""
prefrontal_cortex.py — Strategic Planning & LLM Integration Region.

Responsible for:
  - Long-term strategy generation via LLM (Tetra)
  - Synthesizing reports from all brain regions
  - Hypothesis generation and testing
  - Cross-region coordination
  - Overnight dream analysis and heuristic generation

Operates on the SLOW path. Halted by Amygdala during emergencies.
During overnight loop, this is the primary active region.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion


class PrefrontalCortex(BrainRegion):
    """
    Strategic planning and LLM integration brain region.

    Receives structured reports from all regions, synthesizes them,
    and produces high-level strategy directives.

    Halted during emergencies; resumes when threat subsides.
    """

    def __init__(
        self,
        bus: MessageBus,
        llm_client=None,
        llm_cooldown_steps: int = 100,
    ):
        super().__init__("prefrontal_cortex", bus)
        self._llm_client = llm_client
        self._llm_cooldown = llm_cooldown_steps
        self._steps_since_llm = 0

        # Collected region reports
        self._region_reports: Dict[str, Dict] = {}

        # Strategy history
        self._strategies: deque = deque(maxlen=50)
        self._active_hypothesis: Optional[Dict] = None

        # Heuristics generated from overnight analysis
        self._generated_heuristics: List[Dict] = []

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize information from all regions and produce strategy.

        Expected inputs:
            region_reports: Dict[str, Dict] — reports from all regions
            episode_count: int
            avg_reward: float
        """
        # Collect region reports
        reports = inputs.get("region_reports", {})
        self._region_reports.update(reports)

        # Process incoming messages
        messages = self.receive(max_messages=10)
        for msg in messages:
            if msg.msg_type == "perception":
                self._region_reports["sensory"] = msg.payload
            elif msg.msg_type == "threat_assessment":
                self._region_reports["amygdala"] = msg.payload

        self._steps_since_llm += 1

        # Check if we should query LLM for strategy
        strategy = None
        episode_count = inputs.get("episode_count", 0)
        avg_reward = inputs.get("avg_reward", 0.0)

        if self._should_query_llm(episode_count, avg_reward):
            strategy = self._generate_strategy(episode_count, avg_reward)
            if strategy:
                self._strategies.append(strategy)
                # Send strategy to Striatum
                self.send(
                    target="striatum",
                    msg_type="strategy",
                    payload=strategy,
                )

        return {
            "strategy": strategy,
            "active_hypothesis": self._active_hypothesis,
            "strategies_generated": len(self._strategies),
        }

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Process overnight dream analysis and generate heuristics.

        Expected experience:
            episode_summaries: List[Dict] — from Hippocampus
            dream_results: List[Dict] — from overnight dream loop
        """
        summaries = experience.get("episode_summaries", [])
        dreams = experience.get("dream_results", [])

        heuristics_generated = 0

        if summaries and self._llm_client is not None:
            # In overnight mode: LLM processes summaries and generates heuristics
            # This is where the heavy LLM processing happens (acceptable latency)
            pass

        return {"heuristics_generated": heuristics_generated}

    def _should_query_llm(self, episode_count: int, avg_reward: float) -> bool:
        """Determine if we should query the LLM for new strategy."""
        if self._llm_client is None:
            return False
        if self._steps_since_llm < self._llm_cooldown:
            return False
        return True

    def _generate_strategy(self, episode_count: int, avg_reward: float) -> Optional[Dict]:
        """
        Generate a strategy using LLM analysis.

        In real operation, this builds a prompt from region reports
        and queries Tetra. For now, returns a simple reward-based heuristic.
        """
        self._steps_since_llm = 0

        if self._llm_client is None:
            # No LLM available — generate simple heuristic strategy
            return {
                "type": "heuristic",
                "description": f"Reward-based at ep={episode_count}, avg={avg_reward:.2f}",
                "action_bias": None,  # No bias without LLM analysis
            }

        # Full LLM integration happens in Phase 4 (overnight loop)
        return None

    def request_algorithm_review(
        self,
        meta_report: Dict[str, Any],
        plateau_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request LLM re-evaluation of algorithm selection on plateau.

        Triggered by MetaController when performance plateaus:
        - Formats a structured prompt with current learner stats
        - If LLM available, queries Tetra for recommendation
        - If no LLM, uses rule-based fallback (always re-probe after plateau)

        Returns:
            action: "re_probe" | "switch" | "continue"
            suggestion: Optional learner name
            reason: Explanation
        """
        learner_stats = meta_report.get("learners", {})
        current = meta_report.get("locked_learner", "unknown")
        mode = meta_report.get("mode", "unknown")

        # Build prompt for LLM
        prompt_data = {
            "current_algorithm": current,
            "mode": mode,
            "learner_win_rates": {
                name: stats.get("mean_reward", 0)
                for name, stats in learner_stats.items()
            },
            "plateau": plateau_info or {},
        }

        if self._llm_client is not None:
            # Real LLM path — format prompt and query Tetra
            prompt = (
                f"Performance plateaued using algorithm '{current}'.\n"
                f"Win rates: {prompt_data['learner_win_rates']}\n"
                f"Plateau info: {plateau_info}\n"
                f"Should we re-run the probe phase? Suggest alternatives."
            )
            try:
                response = self._llm_client.query(prompt)
                return {
                    "action": "re_probe",
                    "suggestion": None,
                    "reason": response,
                    "source": "llm",
                }
            except Exception:
                pass  # Fall through to rule-based

        # Rule-based fallback: always recommend re-probe on plateau
        result = {
            "action": "re_probe",
            "suggestion": None,
            "reason": f"Plateau detected for '{current}'. "
                      f"Rule-based fallback: re-probe all algorithms.",
            "source": "rule_based",
        }

        # Publish on bus for MetaController
        self.send(
            target="striatum",
            msg_type="algorithm_review",
            payload=result,
        )

        self._strategies.append({
            "type": "algorithm_review",
            "result": result,
        })

        return result

    def report(self) -> Dict[str, Any]:
        base = super().report()
        return {
            **base,
            "strategies_generated": len(self._strategies),
            "has_llm": self._llm_client is not None,
            "active_hypothesis": self._active_hypothesis is not None,
            "heuristics_generated": len(self._generated_heuristics),
            "regions_reporting": list(self._region_reports.keys()),
        }
