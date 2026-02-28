"""
dream_loop.py — Overnight Dream/Replay/Consolidation Orchestrator.

The DreamLoop runs when the agent is NOT actively playing.
It processes the day's experiences through three phases:

  Phase A: Replay Consolidation
    - Hippocampus replay scheduler selects high-priority transitions
    - Striatum trains on replay batches (offline learning)
    - Amygdala re-evaluates threat estimates with updated knowledge

  Phase B: Dream Simulation
    - Basal Ganglia WorldModel generates imagined trajectories
    - AmygdalaThalamus assesses danger in dream scenarios
    - Hippocampus stores dream results for future replay

  Phase C: Heuristic Compilation
    - HeuristicGenerator extracts fast-path rules from successful patterns
    - Motor Cortex installs new heuristics for real-time execution
    - PrefrontalCortex synthesizes overnight findings into strategy report

Usage:
    from brain.overnight.dream_loop import DreamLoop
    from brain.orchestrator import WholeBrain

    brain = WholeBrain(...)
    # ... run training episodes ...

    dream = DreamLoop(brain)
    report = dream.run(
        n_replay_cycles=50,
        n_dream_steps=20,
        generate_heuristics=True,
    )
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from brain.overnight.replay_scheduler import ReplayScheduler
from brain.overnight.heuristic_generator import HeuristicGenerator
from brain.telemetry.session_logger import SessionLogger


class DreamLoop:
    """
    Overnight dream/replay/consolidation loop.

    Orchestrates offline learning across all brain regions using
    the day's collected experiences.
    """

    def __init__(
        self,
        brain,  # WholeBrain instance (avoid circular import)
        logger: Optional[SessionLogger] = None,
    ):
        self.brain = brain
        self.logger = logger or brain.logger

        # Hippocampus owns the ReplayScheduler now (no standalone one)
        self.heuristic_gen = HeuristicGenerator()

        self._total_cycles = 0
        self._total_dreams = 0

    def run(
        self,
        n_replay_cycles: int = 50,
        n_dream_steps: int = 20,
        generate_heuristics: bool = True,
        max_time_seconds: float = 3600.0,
    ) -> Dict[str, Any]:
        """
        Run the overnight loop.

        Args:
            n_replay_cycles: Number of replay batches to process
            n_dream_steps: Number of dream simulation steps per dream
            generate_heuristics: Whether to produce Motor Cortex heuristics
            max_time_seconds: Hard time limit for the entire loop

        Returns:
            Comprehensive report of overnight processing
        """
        start_time = time.time()
        report = {
            "phase_a_replay": {},
            "phase_b_dreams": {},
            "phase_c_heuristics": {},
            "total_time": 0.0,
        }

        if self.logger:
            self.logger.milestone("overnight", "DreamLoop started")

        # ── Phase A: Replay Consolidation ─────────────────────────────
        replay_report = self._phase_a_replay(n_replay_cycles, start_time, max_time_seconds)
        report["phase_a_replay"] = replay_report

        if self.logger:
            self.logger.event("overnight", "replay_complete",
                f"Replayed {replay_report.get('batches_processed', 0)} batches",
                data=replay_report)

        # ── Phase B: Dream Simulation ─────────────────────────────────
        if time.time() - start_time < max_time_seconds * 0.7:  # Leave time for Phase C
            dream_report = self._phase_b_dreams(n_dream_steps, start_time, max_time_seconds)
            report["phase_b_dreams"] = dream_report

            if self.logger:
                self.logger.event("overnight", "dreams_complete",
                    f"Ran {dream_report.get('dreams_completed', 0)} dream simulations",
                    data=dream_report)

        # ── Phase C: Heuristic Compilation ────────────────────────────
        if generate_heuristics and time.time() - start_time < max_time_seconds * 0.9:
            heuristic_report = self._phase_c_heuristics()
            report["phase_c_heuristics"] = heuristic_report

            if self.logger:
                self.logger.event("overnight", "heuristics_complete",
                    f"Generated {heuristic_report.get('new_heuristics', 0)} heuristics",
                    data=heuristic_report)

        elapsed = time.time() - start_time
        report["total_time"] = round(elapsed, 2)
        self._total_cycles += 1

        if self.logger:
            self.logger.milestone("overnight",
                f"DreamLoop complete: {elapsed:.1f}s, "
                f"replay={replay_report.get('batches_processed', 0)}, "
                f"dreams={report['phase_b_dreams'].get('dreams_completed', 0)}, "
                f"heuristics={report['phase_c_heuristics'].get('new_heuristics', 0)}")

        return report

    # ── Phase A: Replay ───────────────────────────────────────────────

    def _phase_a_replay(
        self, n_cycles: int, start_time: float, max_time: float,
    ) -> Dict[str, Any]:
        """
        Replay high-priority transitions through the Striatum.

        Pulls prioritized batches from Hippocampus's ReplayScheduler
        and trains brain regions on them.
        """
        hippocampus = self.brain.hippocampus
        striatum = self.brain.striatum
        amygdala = self.brain.amygdala

        batches_processed = 0
        total_loss = 0.0

        for cycle in range(n_cycles):
            if time.time() - start_time > max_time * 0.5:
                break

            # Pull prioritized batch from Hippocampus
            batch = hippocampus.get_replay_batch(batch_size=64)
            if not batch:
                break

            # Train Striatum on replay batch
            for trans in batch:
                if len(trans) >= 5:
                    state, action, reward, next_state, done = trans
                    metrics = striatum.learn({
                        "state": state,
                        "action": action,
                        "reward": reward,
                        "next_state": next_state,
                        "done": done,
                    })
                    total_loss += metrics.get("loss", 0.0)

                    # Also update amygdala threat estimates
                    amygdala.learn({
                        "X": np.array([state]),
                        "y": np.array([1.0 if done and reward < 0 else 0.0]),
                        "epochs": 1,
                    })

            batches_processed += 1

        # Age priorities for next overnight cycle
        hippocampus.age_replay_priorities()

        return {
            "batches_processed": batches_processed,
            "avg_loss": round(total_loss / max(batches_processed * 64, 1), 5),
            "scheduler_stats": hippocampus._replay_scheduler.stats(),
        }

    # ── Phase B: Dreams ───────────────────────────────────────────────

    def _phase_b_dreams(
        self, n_steps: int, start_time: float, max_time: float,
    ) -> Dict[str, Any]:
        """
        Run dream simulations using the WorldModel.

        Start from high-priority or edge-case states,
        simulate forward using the WorldModel, and assess results.
        """
        hippocampus = self.brain.hippocampus
        basal_ganglia = self.brain.basal_ganglia
        amygdala = self.brain.amygdala
        striatum = self.brain.striatum

        # Get edge case starting states
        edge_cases = hippocampus.get_edge_cases(n=10)
        dreams_completed = 0
        dream_trajectories = []

        for trans in edge_cases:
            if time.time() - start_time > max_time * 0.7:
                break

            if len(trans) < 5:
                continue

            start_state = trans[0]

            # Dream: simulate forward using greedy policy
            trajectory = {
                "states": [start_state.tolist()],
                "actions": [],
                "rewards": [],
                "total_reward": 0.0,
            }

            dream_state = np.asarray(start_state, dtype=np.float32)

            for step in range(n_steps):
                # Select action using current policy (greedy, no exploration)
                q_values = striatum._forward(dream_state)
                action = int(np.argmax(q_values))

                # "Imagine" next state via simple prediction
                # (Full WorldModel integration would use DreamerEngine here)
                # For now: perturb state based on action embedding
                noise = np.random.randn(len(dream_state)).astype(np.float32) * 0.05
                predicted_next = dream_state + noise
                predicted_reward = float(np.random.randn() * 0.3)

                trajectory["actions"].append(action)
                trajectory["rewards"].append(predicted_reward)
                trajectory["total_reward"] += predicted_reward
                trajectory["states"].append(predicted_next.tolist())

                dream_state = predicted_next

            dream_trajectories.append(trajectory)
            dreams_completed += 1

            # Store dream in hippocampus
            hippocampus.store_dream(trajectory)

        # Assess dream results via amygdala
        if dream_trajectories:
            dream_summaries = []
            for traj in dream_trajectories:
                dream_summaries.append({
                    "total_predicted_reward": traj["total_reward"],
                    "worst_step_reward": min(traj["rewards"]) if traj["rewards"] else 0,
                    "steps": len(traj["actions"]),
                })

            # Amygdala processes dream results for threat assessment
            amygdala.process({
                "features": None,
                "dream_results": dream_summaries,
                "step": 0,
            })

        self._total_dreams += dreams_completed

        return {
            "dreams_completed": dreams_completed,
            "total_dreams_ever": self._total_dreams,
            "dream_trajectories": len(dream_trajectories),
        }

    # ── Phase C: Heuristics ───────────────────────────────────────────

    def _phase_c_heuristics(self) -> Dict[str, Any]:
        """
        Generate heuristics from replay and dream analysis.

        Extract fast-path rules and install them in Motor Cortex.
        """
        hippocampus = self.brain.hippocampus
        motor = self.brain.motor

        # Process replay transitions for heuristic extraction
        buffer = hippocampus._transitions
        if buffer:
            states = [np.asarray(t[0]) for t in buffer if len(t) >= 5]
            actions = [t[1] for t in buffer if len(t) >= 5]
            rewards = [t[2] for t in buffer if len(t) >= 5]

            new_heuristics = self.heuristic_gen.process_replay_batch(
                states=states[-500:],  # Last 500 transitions
                actions=actions[-500:],
                rewards=rewards[-500:],
            )
        else:
            new_heuristics = 0

        # Process dream trajectories
        dream_results = list(hippocampus._dream_results)
        if dream_results:
            dream_heuristics = self.heuristic_gen.process_dream_results(dream_results)
            new_heuristics += dream_heuristics

        # Install in Motor Cortex
        heuristic_table = self.heuristic_gen.export_for_motor_cortex()
        motor.install_heuristics(heuristic_table)

        return {
            "new_heuristics": new_heuristics,
            "total_heuristics": len(heuristic_table),
            "heuristic_stats": self.heuristic_gen.stats(),
            "installed_in_motor": len(heuristic_table),
        }

    # ── State ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "total_cycles": self._total_cycles,
            "total_dreams": self._total_dreams,
            "replay_scheduler": self.brain.hippocampus._replay_scheduler.stats(),
            "heuristic_generator": self.heuristic_gen.stats(),
        }
