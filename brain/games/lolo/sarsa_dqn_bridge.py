"""
sarsa_dqn_bridge.py — Dual-process bridge: SARSA (solver) + DQN (habit).

System 1 (DQN): Fast, generalizes to unseen states via neural network.
System 2 (SARSA): Slower but reliable, exact Q-table lookup.

Decision logic:
  1. DQN queries Q-values from compressed state
  2. If DQN is confident (Q-spread > threshold) → use DQN
  3. If DQN uncertain AND SARSA has this state → use SARSA
  4. When SARSA solves → feed (state, action) pairs to DQN's replay

This ensures:
  - Existing knowledge is used immediately (SARSA)
  - New puzzles get solved quickly (SARSA fallback)
  - DQN continuously improves (distillation from SARSA)
  - Eventually DQN handles everything (generalization)
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.games.lolo.lolo_compressed_state import LoloCompressedState


class SarsaDQNBridge:
    """
    Dual-process bridge between SARSA (tabular solver) and DQN (habit network).

    Usage:
        bridge = SarsaDQNBridge()
        bridge.load_sarsa("brain/games/lolo/sarsa_qtable.npy")
        bridge.load_dqn("brain/games/lolo/dqn_weights.pt")

        # Each step:
        action, source = bridge.select_action(sim)

        # After each step:
        bridge.observe(sim, action, reward, done)
    """

    def __init__(
        self,
        n_actions: int = 6,
        confidence_threshold: float = 0.3,
        distill_batch_size: int = 64,
        distill_interval: int = 100,
    ):
        self.n_actions = n_actions
        self.confidence_threshold = confidence_threshold
        self.distill_batch_size = distill_batch_size
        self.distill_interval = distill_interval

        # Components
        self._sarsa_qtable: Optional[dict] = None
        self._dqn = None
        self._encoder = LoloCompressedState()

        # State tracking
        self._step_count = 0
        self._sarsa_used = 0
        self._dqn_used = 0
        self._last_source = "none"  # "dqn" | "sarsa" | "random"

        # Distillation queue: (compressed_state, best_action)
        self._distill_queue: deque = deque(maxlen=10000)
        self._distill_count = 0

        # Episode tracking for SARSA solution recording
        self._episode_states: List[np.ndarray] = []
        self._episode_actions: List[int] = []
        self._episode_rewards: List[float] = []

    # ── Load / Save ──────────────────────────────────────────────────

    def load_sarsa(self, path: str) -> bool:
        """Load SARSA Q-table from .npy file."""
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True).item()
            self._sarsa_qtable = data
            print(f"  Bridge: loaded SARSA Q-table ({len(data)} states)")
            return True
        except Exception as e:
            print(f"  Bridge: failed to load SARSA: {e}")
            return False

    def load_dqn(self, path: str) -> bool:
        """Load DQN weights from .pt file."""
        try:
            from brain.games.lolo.lolo_dqn_learner import LoloDQNLearner
            self._dqn = LoloDQNLearner(n_actions=self.n_actions)
            self._dqn.load(path)
            self._dqn.epsilon = 0.02  # Near-greedy for inference
            print(f"  Bridge: loaded DQN weights from {path}")
            return True
        except Exception as e:
            print(f"  Bridge: failed to load DQN: {e}")
            return False

    def has_sarsa(self) -> bool:
        return self._sarsa_qtable is not None and len(self._sarsa_qtable) > 0

    def has_dqn(self) -> bool:
        return self._dqn is not None

    # ── Action Selection ─────────────────────────────────────────────

    def select_action(self, sim) -> Tuple[int, str]:
        """
        Select action using the dual-process system.

        Returns:
            (action, source) where source is "dqn", "sarsa", or "random"
        """
        compressed = self._encoder.encode_from_sim(sim)
        compressed_key = self._encoder.encode_key(sim)
        return self._select_action_impl(compressed, compressed_key)

    def select_action_from_features(self, compressed: np.ndarray) -> Tuple[int, str]:
        """
        Select action from pre-computed 84-dim features (for ROM usage).

        Args:
            compressed: 84-dim float array from encode_from_ram()

        Returns:
            (action, source) where source is "dqn", "sarsa", or "random"
        """
        compressed_key = self._encoder.quantize(compressed)
        return self._select_action_impl(compressed, compressed_key)

    def _select_action_impl(self, compressed: np.ndarray,
                            compressed_key: tuple) -> Tuple[int, str]:
        """Core action selection logic."""
        # Try DQN first
        dqn_q = None
        dqn_confidence = 0.0
        if self._dqn is not None:
            dqn_q = self._dqn.get_q_values(compressed)
            q_range = float(dqn_q.max() - dqn_q.min())
            dqn_confidence = q_range

        # Try SARSA
        sarsa_q = None
        if self._sarsa_qtable is not None and compressed_key in self._sarsa_qtable:
            sarsa_q = self._sarsa_qtable[compressed_key]

        # Decision logic
        if dqn_q is not None and dqn_confidence > self.confidence_threshold:
            action = int(np.argmax(dqn_q))
            self._dqn_used += 1
            self._last_source = "dqn"
            return action, "dqn"

        if sarsa_q is not None:
            action = int(np.argmax(sarsa_q))
            self._sarsa_used += 1
            self._last_source = "sarsa"
            self._distill_queue.append((compressed.copy(), sarsa_q.copy()))
            return action, "sarsa"

        if dqn_q is not None:
            action = int(np.argmax(dqn_q))
            self._dqn_used += 1
            self._last_source = "dqn"
            return action, "dqn"

        # Neither available → random
        action = np.random.randint(self.n_actions)
        self._last_source = "random"
        return action, "random"

    def get_action_values(self, sim) -> Optional[np.ndarray]:
        """Get blended action values for the striatum."""
        compressed = self._encoder.encode_from_sim(sim)
        compressed_key = self._encoder.encode_key(sim)

        dqn_q = None
        if self._dqn is not None:
            dqn_q = self._dqn.get_q_values(compressed)

        sarsa_q = None
        if self._sarsa_qtable is not None and compressed_key in self._sarsa_qtable:
            sarsa_q = self._sarsa_qtable[compressed_key]

        if dqn_q is not None and sarsa_q is not None:
            # Blend: average both (SARSA is more reliable but DQN generalizes)
            return 0.6 * sarsa_q + 0.4 * dqn_q
        elif sarsa_q is not None:
            return sarsa_q
        elif dqn_q is not None:
            return dqn_q
        return None

    # ── Learning ─────────────────────────────────────────────────────

    def observe(self, sim, action: int, reward: float, done: bool):
        """
        Observe a transition for episode tracking and distillation.

        Call this after each step.
        """
        compressed = self._encoder.encode_from_sim(sim)
        self._observe_impl(compressed, action, reward, done)

    def observe_from_features(self, compressed: np.ndarray, action: int,
                               reward: float, done: bool):
        """Observe from pre-computed features (for ROM usage)."""
        self._observe_impl(compressed, action, reward, done)

    def _observe_impl(self, compressed: np.ndarray, action: int,
                       reward: float, done: bool):
        """Core observe logic."""
        self._step_count += 1

        # Track episode for SARSA solution recording
        self._episode_states.append(compressed.copy())
        self._episode_actions.append(action)
        self._episode_rewards.append(reward)

        if done:
            # If episode was successful, queue ALL state-action pairs for DQN
            total_reward = sum(self._episode_rewards)
            if total_reward > 0:  # Positive reward = solved
                for s, a in zip(self._episode_states, self._episode_actions):
                    q_target = np.zeros(self.n_actions, dtype=np.float32)
                    q_target[a] = 1.0
                    self._distill_queue.append((s, q_target))

            self._episode_states = []
            self._episode_actions = []
            self._episode_rewards = []

        # Periodic distillation from queue into DQN
        if (self._step_count % self.distill_interval == 0
                and len(self._distill_queue) >= self.distill_batch_size
                and self._dqn is not None):
            self._run_distillation()

    def _run_distillation(self):
        """Train DQN on queued SARSA observations."""
        import torch

        batch_size = min(self.distill_batch_size, len(self._distill_queue))

        # Sample from queue
        indices = np.random.choice(len(self._distill_queue), batch_size, replace=False)
        items = [self._distill_queue[i] for i in indices]
        states = np.array([x[0] for x in items], dtype=np.float32)
        targets = np.array([x[1] for x in items], dtype=np.float32)

        device = self._dqn.device
        s = torch.tensor(states, device=device)
        t = torch.tensor(targets, device=device)

        predicted = self._dqn.q_net(s)
        loss = torch.nn.functional.mse_loss(predicted, t)

        self._dqn.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._dqn.q_net.parameters(), 10.0)
        self._dqn.optimizer.step()

        self._distill_count += 1

    # ── Live SARSA Solving ───────────────────────────────────────────

    def sarsa_solve(self, sim, max_episodes: int = 50,
                    max_steps: int = 500) -> dict:
        """
        Have SARSA actively solve a puzzle, recording the solution
        for DQN distillation.

        This is the "System 2" fallback: brain doesn't know this puzzle,
        so SARSA brute-forces it and DQN learns from watching.

        Returns:
            {"solved": bool, "episodes": int, "solution_length": int,
             "distilled_pairs": int}
        """
        from brain.games.lolo.lolo_sarsa_learner import LoloSarsaLearner

        # Create a fresh SARSA with existing Q-table for warm start
        solver = LoloSarsaLearner(n_actions=self.n_actions)
        if self._sarsa_qtable:
            solver.q_table = dict(self._sarsa_qtable)  # Copy existing knowledge
        solver.set_simulator(sim)

        initial_state = sim.save()
        solved = False
        solution_actions = []
        solution_states = []
        distilled = 0

        for ep in range(max_episodes):
            sim.load(initial_state)
            solver.set_simulator(sim)
            ep_states = []
            ep_actions = []

            for step in range(max_steps):
                compressed = self._encoder.encode_from_sim(sim)
                result = solver.step(reward=0.0 if step == 0 else
                                     (10.0 if sim.won else -0.01),
                                     done=sim.won or step == max_steps - 1)
                action = result["action"]

                ep_states.append(compressed.copy())
                ep_actions.append(action)

                obs, reward, done, info = sim.step(action)

                if done:
                    # Final step for SARSA
                    solver.step(reward=reward, done=True)
                    break

            if sim.won:
                solved = True
                solution_states = ep_states
                solution_actions = ep_actions

                # Queue all solution pairs for DQN distillation
                for s, a in zip(solution_states, solution_actions):
                    q_target = np.zeros(self.n_actions, dtype=np.float32)
                    q_target[a] = 1.0
                    self._distill_queue.append((s, q_target))
                    distilled += 1
                break

        # Update our SARSA Q-table with new knowledge
        if self._sarsa_qtable is None:
            self._sarsa_qtable = {}
        self._sarsa_qtable.update(solver.q_table)

        return {
            "solved": solved,
            "episodes": ep + 1,
            "solution_length": len(solution_actions) if solved else 0,
            "distilled_pairs": distilled,
            "sarsa_states": len(solver.q_table),
        }

    # ── Reporting ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        total_actions = self._dqn_used + self._sarsa_used
        return {
            "steps": self._step_count,
            "dqn_used": self._dqn_used,
            "sarsa_used": self._sarsa_used,
            "dqn_pct": (self._dqn_used / max(total_actions, 1)),
            "sarsa_pct": (self._sarsa_used / max(total_actions, 1)),
            "distill_queue": len(self._distill_queue),
            "distill_count": self._distill_count,
            "sarsa_states": len(self._sarsa_qtable) if self._sarsa_qtable else 0,
            "has_dqn": self._dqn is not None,
            "last_source": self._last_source,
        }
