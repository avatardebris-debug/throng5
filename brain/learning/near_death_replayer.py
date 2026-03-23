"""
near_death_replayer.py — Counterfactual replay from near-death bottleneck states.

Concept
-------
When the elite replay buffer is triggered, instead of sampling random transitions
from a top episode, this module:

  1.  Picks a random elite episode.
  2.  Scans backward from the episode end to find the LAST "near-death" state
      (a state where reward was very negative, or the step immediately before done=True).
  3.  At the near-death state, checks every alternative action the agent DIDN'T take
      using the current DQN Q-values.  If a better action exists (higher Q or non-negative
      predicted next-state reward) that's the "escape action".
  4.  Builds a replay batch of two layers:
        Layer A — all transitions from ep start → near-death  (played-through memory)
        Layer B — a synthetic counterfactual transition:
                  (near_death_state, escape_action, +bonus_reward, predicted_next, False)
  5.  Returns this batch for immediate training, at inflated priority.

The "backward search" terminates as soon as an escape action is found at the
near-death state.  If the Q-network can't distinguish a better action (all Q-values
equal) the module walks backward one more step and tries again (up to max_walk steps).

Why this helps
--------------
* The agent relives the entire run to the bottleneck — re-consolidating the
  successful path into weights.
* The synthetic transition at the bottleneck teaches "if you do X instead,
  you would have survived" — explicit counterfactual signal.
* Combined with elite sampling this is the CPU-only equivalent of Go-Explore:
  "return to the promising state and try something different".

Usage
-----
    replayer = NearDeathReplayer(world_model=brain.basal_ganglia._world_model)
    batch = replayer.build_replay_batch(elite_buf, striatum)
    # inject batch into striatum replay
    for t in batch:
        striatum._replay.append(t)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class NearDeathReplayer:
    """
    Counterfactual replay from near-death bottleneck states in elite episodes.

    Parameters
    ----------
    world_model : optional
        If provided (WorldModel instance from basal_ganglia), used to simulate
        the next state under the escape action for a more accurate counterfactual.
        Falls back to copying the near-death next_state if absent.
    near_death_reward_thresh : float
        Transitions with reward below this are classified as near-death.
        Default -0.5.  For CartPole/Atari where done=True signals death,
        also the step immediately before done=True is flagged.
    bonus_reward : float
        Synthetic reward given to the counterfactual escape transition.
        Should be comfortably positive to override the near-death signal.
        Default +1.0.
    max_walk : int
        Maximum number of steps to walk backward from the end when searching
        for an escape action.  Default 10.
    pre_death_window : int
        Number of transitions BEFORE the near-death state to include in the
        replay batch (in addition to the full run).  Provides denser gradient
        signal around the bottleneck.  Default 5.
    """

    def __init__(
        self,
        world_model=None,
        near_death_reward_thresh: float = -0.5,
        bonus_reward: float = 1.0,
        max_walk: int = 10,
        pre_death_window: int = 5,
    ):
        self._wm = world_model
        self.near_death_thresh = near_death_reward_thresh
        self.bonus_reward = bonus_reward
        self.max_walk = max_walk
        self.pre_death_window = pre_death_window

        # Stats
        self._attempts = 0
        self._escapes_found = 0
        self._batches_built = 0

    # ── Public API ────────────────────────────────────────────────────

    def build_replay_batch(
        self,
        elite_buf,           # EliteReplayBuffer
        striatum,            # Striatum (for Q-values)
        world_model=None,    # override model
    ) -> List[Transition]:
        """
        Select random elite episode, find near-death state, build replay batch.

        Returns a list of (s, a, r, s', done) transitions ready to append
        directly into striatum._replay or pass to _train_batch_direct().

        Returns [] if the elite buffer is empty or no valid episode found.
        """
        if elite_buf is None or elite_buf.is_empty:
            return []

        # 1. Pick a random elite episode
        ep = random.choice(elite_buf._archive)
        transitions = ep.transitions
        if len(transitions) < 3:
            return []

        self._attempts += 1

        # 2. Find the last near-death state (scan backward)
        critical_idx = self._find_critical_state(transitions)
        if critical_idx is None:
            # No near-death state, use last step as the critical point
            critical_idx = len(transitions) - 1

        # 3. Backward search: find an escape action at or before critical_idx
        escape_found, escape_idx, escape_action = self._backward_search(
            transitions, critical_idx, striatum,
        )
        if escape_found:
            self._escapes_found += 1

        # 4. Build replay batch
        batch = self._build_batch(
            transitions, escape_idx, escape_action, escape_found, striatum, world_model or self._wm,
        )
        self._batches_built += 1
        return batch

    def build_replay_batch_direct(
        self,
        elite_buf,
        striatum,
        world_model=None,
    ) -> int:
        """
        Build batch and inject directly into striatum._replay.
        Returns number of transitions injected.
        """
        batch = self.build_replay_batch(elite_buf, striatum, world_model)
        for t in batch:
            striatum._replay.append(t)
        return len(batch)

    # ── Near-death detection ──────────────────────────────────────────

    def _find_critical_state(self, transitions: List[Transition]) -> Optional[int]:
        """
        Scan backward to find the last near-death state.

        A transition is near-death if:
          - reward < near_death_thresh, OR
          - done=True (the terminal state — the death step itself)

        Returns the index into transitions, or None if none found.
        """
        # Prefer the step BEFORE done=True (the last action before death)
        for i in range(len(transitions) - 1, -1, -1):
            s, a, r, s2, done = transitions[i]
            if done:
                # Return the step just before the death, not the death itself
                return max(0, i - 1)

        # Fallback: scan for large negative rewards
        for i in range(len(transitions) - 1, -1, -1):
            s, a, r, s2, done = transitions[i]
            if r < self.near_death_thresh:
                return i

        return None

    # ── Backward search ───────────────────────────────────────────────

    def _backward_search(
        self,
        transitions: List[Transition],
        start_idx: int,
        striatum,
    ) -> Tuple[bool, int, int]:
        """
        Walk backward from start_idx looking for a state where an alternative
        action has a higher Q-value (or at least different Q-value) than what
        was taken.

        Returns (escape_found, best_idx, escape_action).
        escape_action is the alternative action to take at best_idx.
        """
        n_actions = striatum.n_actions

        for walk in range(self.max_walk):
            idx = max(0, start_idx - walk)
            state, action_taken, reward, next_state, done = transitions[idx]

            # Get Q-values for this state
            try:
                q_values = striatum._forward(
                    np.asarray(state, dtype=np.float32)
                )
            except Exception:
                continue

            if q_values is None or len(q_values) < n_actions:
                continue

            # Check if any action is strictly better than what was taken
            taken_q = float(q_values[action_taken]) if action_taken < len(q_values) else -999.0
            best_alt_q = float('-inf')
            best_alt_action = -1

            for a in range(n_actions):
                if a == action_taken:
                    continue
                if float(q_values[a]) > best_alt_q:
                    best_alt_q = float(q_values[a])
                    best_alt_action = a

            # Accept if the alternative action has a meaningfully higher Q-value
            if best_alt_action >= 0 and best_alt_q > taken_q + 0.01:
                return True, idx, best_alt_action

            # If all Q-values are identical (random init), pick a random alternative
            if all(abs(float(q_values[a]) - float(q_values[0])) < 1e-6 for a in range(n_actions)):
                alt_actions = [a for a in range(n_actions) if a != action_taken]
                if alt_actions:
                    return True, idx, random.choice(alt_actions)

        # No escape found — use the critical state with random alternative
        state, action_taken, reward, next_state, done = transitions[start_idx]
        alt_actions = [a for a in range(n_actions) if a != action_taken]
        if alt_actions:
            return False, start_idx, random.choice(alt_actions)

        return False, start_idx, 0

    # ── Batch construction ────────────────────────────────────────────

    def _build_batch(
        self,
        transitions: List[Transition],
        escape_idx: int,
        escape_action: int,
        escape_found: bool,
        striatum,
        world_model,
    ) -> List[Transition]:
        """
        Construct the full replay batch:
          - Layer A: all transitions from ep start → escape_idx (the full run-up)
          - Layer B: extra pre-death window around escape_idx (denser signal)
          - Layer C: synthetic counterfactual at escape_idx with escape_action
        """
        batch: List[Transition] = []

        # Layer A: full play-through up to escape point
        for t in transitions[:escape_idx + 1]:
            batch.append(t)

        # Layer B: the pre-death window is already included in A, but
        # duplicate the last `pre_death_window` steps to over-represent them
        window_start = max(0, escape_idx - self.pre_death_window + 1)
        for t in transitions[window_start:escape_idx + 1]:
            batch.append(t)

        # Layer C: synthetic counterfactual transition
        state, action_taken, reward, next_state, done = transitions[escape_idx]

        # Predict next state under escape action using world model if available
        predicted_next = self._predict_next_state(
            state, escape_action, next_state, world_model, striatum,
        )

        # Reward: bonus if escape was Q-guided, smaller bonus if random
        synth_reward = self.bonus_reward if escape_found else self.bonus_reward * 0.5

        synthetic: Transition = (
            np.asarray(state, dtype=np.float32),
            escape_action,
            float(synth_reward),
            np.asarray(predicted_next, dtype=np.float32),
            False,  # Not done — the escape action avoids death
        )
        # Add synthetic transition multiple times for emphasis
        n_repeats = 3 if escape_found else 1
        for _ in range(n_repeats):
            batch.append(synthetic)

        return batch

    def _predict_next_state(
        self,
        state: np.ndarray,
        action: int,
        fallback_next: np.ndarray,
        world_model,
        striatum,
    ) -> np.ndarray:
        """
        Use the world model to predict next state under the escape action.
        Falls back to the original next_state if model unavailable.
        """
        if world_model is None:
            return fallback_next

        try:
            # WorldModel.predict(state, action) returns (next_state, reward)
            result = world_model.predict(state, action)
            if isinstance(result, tuple) and len(result) >= 1:
                predicted = np.asarray(result[0], dtype=np.float32)
                if predicted.shape == state.shape:
                    return predicted
        except Exception:
            pass

        return fallback_next

    # ── Stats ─────────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        escape_rate = (
            self._escapes_found / self._attempts if self._attempts > 0 else 0.0
        )
        return {
            "attempts": self._attempts,
            "escapes_found": self._escapes_found,
            "batches_built": self._batches_built,
            "escape_rate": round(escape_rate, 3),
        }

    def __repr__(self) -> str:
        return (
            f"NearDeathReplayer("
            f"attempts={self._attempts}, "
            f"escapes={self._escapes_found}, "
            f"rate={self._escapes_found/max(1,self._attempts):.2f})"
        )
