"""
hippocampus.py — Memory, Replay, and Dream/Sleep Consolidation Region.

Responsible for:
  - Storing episode transitions in prioritized replay buffer
  - Scheduling replay batches for Striatum training
  - Running dream/sleep consolidation (overnight loop entry point)
  - Generating compressed episode summaries for Prefrontal Cortex

Operates primarily on the SLOW path and overnight loop.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion
from brain.overnight.replay_scheduler import ReplayScheduler
from brain.learning.elite_replay import EliteReplayBuffer
from brain.learning.her_replay import HERReplayBuffer
from brain.planning.go_explore import GoExploreArchive


class Hippocampus(BrainRegion):
    """
    Memory and replay brain region.

    Stores transitions, prioritizes by surprise/reward, and feeds
    replay batches to the Striatum and other learning regions.
    """

    def __init__(
        self,
        bus: MessageBus,
        buffer_size: int = 50000,
        batch_size: int = 64,
        priority_alpha: float = 0.6,
    ):
        super().__init__("hippocampus", bus)
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._priority_alpha = priority_alpha

        # Transition storage: (state, action, reward, next_state, done, priority)
        self._transitions: deque = deque(maxlen=buffer_size)
        self._priorities: deque = deque(maxlen=buffer_size)

        # Episode memory: compressed episode summaries
        self._episode_summaries: deque = deque(maxlen=1000)

        # Dream results storage
        self._dream_results: deque = deque(maxlen=200)

        # Edge cases for targeted replay
        self._edge_cases: deque = deque(maxlen=500)

        # Overnight replay scheduler (priority queue for DreamLoop)
        self._replay_scheduler = ReplayScheduler(max_items=buffer_size)

        # Stats
        self._total_stored = 0
        self._total_replayed = 0

        # ── Elite Replay Buffer ───────────────────────────────────────
        # Keeps top-3 episodes by reward; samples at 80% → 20% decay schedule
        self.elite: EliteReplayBuffer = EliteReplayBuffer(
            top_k=3,
            start_fraction=0.80,
            floor_fraction=0.20,
            halflife=200.0,
        )
        # Buffer for current in-progress episode transitions
        self._current_episode_transitions: List[Tuple] = []
        self._current_episode_reward: float = 0.0

        # ── Hindsight Experience Replay (HER) ─────────────────────────
        # Active for sparse-reward envs (total_reward ≤ 0 at done).
        # Relabels failed trajectories with future-state hindsight goals.
        self.her: HERReplayBuffer = HERReplayBuffer(
            k=4,                        # future goals per transition
            sparse_reward_threshold=0.0,# activate when episode reward ≤ 0
            goal_reward=1.0,            # reward given to relabelled successes
        )
        # External ref to striatum for HER injection (set by orchestrator)
        self._striatum_ref = None

        # ── Go-Explore State Archive ─────────────────────────────────────
        # Cell-based archive of interesting states. on sparse episodes,
        # select_return_state() gives a good re-entry point for next episode.
        self.go_explore: GoExploreArchive = GoExploreArchive(
            max_cells=5000,
            resolution=0.10,
            min_steps_before_return=5,
        )
        self._go_explore_return_state: Optional[np.ndarray] = None

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a transition and optionally produce a replay batch.

        Expected inputs:
            state: np.ndarray
            action: int
            reward: float
            next_state: np.ndarray
            done: bool
            td_error: Optional[float] — for priority calculation
            is_edge_case: bool — near-death, recovery, etc.
        """
        state = inputs.get("state")
        action = inputs.get("action")
        reward = inputs.get("reward", 0.0)
        next_state = inputs.get("next_state")
        done = inputs.get("done", False)
        td_error = inputs.get("td_error", 0.0)
        is_edge_case = inputs.get("is_edge_case", False)

        if state is not None and next_state is not None:
            # Calculate priority
            priority = (abs(td_error) + 1e-5) ** self._priority_alpha
            if abs(reward) > 1.0:
                priority *= 2.0
            if is_edge_case:
                priority *= 3.0

            self._transitions.append((
                np.asarray(state, dtype=np.float32),
                action,
                reward,
                np.asarray(next_state, dtype=np.float32),
                done,
            ))
            self._priorities.append(priority)
            self._total_stored += 1

            # Go-Explore: track every state visit
            try:
                ep_step = len(self._current_episode_transitions)
                self.go_explore.add_state(state, reward=reward, step=ep_step)
            except Exception:
                pass

            # Track current episode for elite buffer
            self._current_episode_transitions.append((
                np.asarray(state, dtype=np.float32),
                action,
                reward,
                np.asarray(next_state, dtype=np.float32),
                done,
            ))
            self._current_episode_reward += reward

            # Feed overnight replay scheduler
            self._replay_scheduler.add(
                idx=self._total_stored - 1,
                td_error=td_error,
                reward=reward,
                is_edge_case=is_edge_case,
            )

            if is_edge_case:
                self._edge_cases.append(self._total_stored - 1)

        # End of episode: store summary + try to add to elite archive
        if done:
            self._store_episode_summary(inputs)
            # Offer this episode to the elite archive
            if self._current_episode_transitions:
                self.elite.try_add_episode(
                    transitions=self._current_episode_transitions,
                    total_reward=self._current_episode_reward,
                    label="agent",
                )
                # ── HER: relabel sparse/failed episodes ──────────────
                if self.her.should_relabel(self._current_episode_reward):
                    relabelled = self.her.relabel_episode(
                        self._current_episode_transitions,
                        total_reward=self._current_episode_reward,
                    )
                    for (rs, ra, rr, rs2, rdone) in relabelled:
                        if self._striatum_ref is not None:
                            # Push directly into striatum's n-step buffer
                            self._striatum_ref._nstep.push(
                                rs, ra, rr, rs2, rdone, near_death=False,
                            )
                        else:
                            # Fallback: push to own transitions deque
                            self._transitions.append((rs, ra, rr, rs2, rdone))
                            self._priorities.append(1.0)
                            self._total_stored += 1

                # ── Go-Explore: pick return state for next ep ─────────
                if self.her.should_relabel(self._current_episode_reward):
                    ret_state, _ = self.go_explore.select_return_state(strategy="mixed")
                    self._go_explore_return_state = ret_state  # orchestrator may use
                else:
                    self._go_explore_return_state = None

            # Reset episode accumulator
            self._current_episode_transitions = []
            self._current_episode_reward = 0.0

        return {
            "buffer_size": len(self._transitions),
            "total_stored": self._total_stored,
            "go_explore_cells": len(self.go_explore),
        }

    def set_striatum_ref(self, striatum) -> None:
        """Wire reference to Striatum for direct HER injection. Call from orchestrator."""
        self._striatum_ref = striatum

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate a prioritized replay batch and send to Striatum.

        Called periodically (not every step) to avoid overwhelming the bus.
        """
        if len(self._transitions) < self._batch_size:
            return {"replayed": 0}

        # Sample batch with priority weighting
        batch = self._sample_batch()

        # Send replay to Striatum
        self.send(
            target="striatum",
            msg_type="replay_batch",
            payload={"batch": batch},
        )

        self._total_replayed += len(batch)

        return {
            "replayed": len(batch),
            "total_replayed": self._total_replayed,
            "buffer_size": len(self._transitions),
        }

    def _sample_batch(self) -> List[Tuple]:
        """Sample a prioritized batch of transitions."""
        n = len(self._transitions)
        priorities = np.array(list(self._priorities), dtype=np.float32)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(n, size=min(self._batch_size, n), replace=False, p=probabilities)
        return [self._transitions[i] for i in indices]

    def _store_episode_summary(self, inputs: Dict[str, Any]) -> None:
        """Store a compressed summary of the completed episode."""
        summary = {
            "total_reward": inputs.get("episode_reward", 0.0),
            "steps": inputs.get("episode_steps", 0),
            "final_state": inputs.get("state"),
        }
        self._episode_summaries.append(summary)

    # ── Dream/Overnight API ───────────────────────────────────────────

    def store_dream(self, dream_result: Dict[str, Any]) -> None:
        """Store results from a dream simulation for overnight processing."""
        self._dream_results.append(dream_result)

    def get_edge_cases(self, n: int = 10) -> List[Tuple]:
        """Get recent edge-case transitions for targeted dream replay."""
        indices = list(self._edge_cases)[-n:]
        cases = []
        for idx in indices:
            if idx < len(self._transitions):
                cases.append(self._transitions[idx])
        return cases

    def get_episode_summaries(self, n: int = 50) -> List[Dict]:
        """Get recent episode summaries for Prefrontal analysis."""
        return list(self._episode_summaries)[-n:]

    def get_replay_batch(self, batch_size: int = 64) -> List[Tuple]:
        """
        Get a prioritized replay batch for overnight DreamLoop.

        Uses the ReplayScheduler for priority-aware selection.
        Returns actual transitions (not just indices).
        """
        indices = self._replay_scheduler.schedule_batch(batch_size)
        batch = []
        for idx in indices:
            if idx < len(self._transitions):
                batch.append(self._transitions[idx])
        self._total_replayed += len(batch)
        return batch

    def age_replay_priorities(self) -> None:
        """Age all replay priorities (called each overnight cycle)."""
        self._replay_scheduler.age_all()

    def report(self) -> Dict[str, Any]:
        base = super().report()
        return {
            **base,
            "buffer_size": len(self._transitions),
            "total_stored": self._total_stored,
            "total_replayed": self._total_replayed,
            "edge_cases": len(self._edge_cases),
            "episode_summaries": len(self._episode_summaries),
            "dream_results": len(self._dream_results),
            "replay_scheduler": self._replay_scheduler.stats(),
            "elite": self.elite.report(),
        }
