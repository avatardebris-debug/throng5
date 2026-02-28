"""
rehearsal_loop.py — 3-tier progressive validation with 4 operating modes.

The Rehearsal Loop is an active pre-training system that pauses the game
at bottleneck states and systematically finds, validates, and stores
action sequences that overcome them.

3 Validation Tiers:
    Compressed: DQN inference rollouts (1000 trials, cheapest)
    World Model: forward simulation (100 trials, medium cost)
    Real Execution: save/load state cycles (10 trials, always 10 for robustness)

4 Operating Modes:
    Advance:  Always pause → 3-tier → execute → pause → repeat
    Frontier: Play from start; on death → Advance at death point
    Stuck:    10 failures → train flanking areas before/after
    Free Run: Play normally, log stuck points for LLM review

Usage:
    from brain.rehearsal.rehearsal_loop import RehearsalLoop

    loop = RehearsalLoop(brain)
    report = loop.run_advance(features, env)
    report = loop.run_frontier(env)
    report = loop.run_stuck(features, env)
    report = loop.run_free(env, max_episodes=100)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from brain.rehearsal.bottleneck_tracker import BottleneckTracker
from brain.rehearsal.action_chain import ActionChainStore, TIER_ORDER


class RehearsalLoop:
    """
    Active rehearsal system for mastering game bottlenecks.

    Pauses the game, runs cheap-to-expensive validation tiers,
    and promotes successful action chains through confidence levels.
    """

    def __init__(
        self,
        brain,
        success_threshold: float = 0.60,
        compressed_trials: int = 1000,
        worldmodel_trials: int = 100,
        real_trials: int = 10,          # Always 10, even on success
        max_chain_length: int = 200,    # Max actions per chain
    ):
        self.brain = brain
        self.success_threshold = success_threshold
        self.compressed_trials = compressed_trials
        self.worldmodel_trials = worldmodel_trials
        self.real_trials = real_trials
        self.max_chain_length = max_chain_length

        self.tracker = BottleneckTracker()
        self.chain_store = ActionChainStore()

        # Decaying reliance: rehearse less as we learn more
        self._total_rehearsals: int = 0
        self._total_proven: int = 0

        # Stats
        self._advance_count: int = 0
        self._frontier_count: int = 0
        self._stuck_count: int = 0
        self._free_run_count: int = 0

    @property
    def rehearsal_probability(self) -> float:
        """Probability of rehearsing (decays as more chains are proven)."""
        if self._total_proven == 0:
            return 1.0
        ratio = self._total_proven / max(self.tracker._n_seen, 1)
        return max(0.10, 1.0 - ratio * 2)

    # ══════════════════════════════════════════════════════════════════
    # MODE 1: ADVANCE — pause, validate, execute, repeat
    # ══════════════════════════════════════════════════════════════════

    def run_advance(
        self,
        features: np.ndarray,
        env,
        policy_fn: Optional[Callable] = None,
        success_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Full 3-tier validation at current state.

        1. Save state
        2. Compressed: 1000 DQN rollouts → find chains with >60% success
        3. World Model: 100 trials with best chains → confirm >60%
        4. Real: 10 save/load cycles (always 10) → confirm >60%
        5. Store proven chain

        Args:
            features: Current state features
            env: Environment with save_state/load_state support
            policy_fn: Action selection function (default: brain's policy)
            success_fn: Custom success checker (default: survive N steps)

        Returns:
            Report with chain info and promotion result.
        """
        self._advance_count += 1
        self._total_rehearsals += 1

        if policy_fn is None:
            policy_fn = self._default_policy

        if success_fn is None:
            success_fn = self._default_success_check

        report = {
            "mode": "advance",
            "state_hash": 0,
            "tiers_completed": [],
            "final_tier": None,
            "chain_length": 0,
            "success": False,
        }

        # ── Tier 1: Compressed Model ─────────────────────────────────
        compressed_result = self._compressed_probe(
            features, policy_fn, success_fn, self.compressed_trials,
        )
        report["compressed"] = compressed_result
        report["tiers_completed"].append("compressed")

        if compressed_result["success_rate"] < self.success_threshold:
            report["final_tier"] = "compressed_failed"
            return report

        best_chain = compressed_result["best_chain"]
        state_hash = self.chain_store.store(
            features, best_chain,
            tier="compressed",
            success_rate=compressed_result["success_rate"],
            trials=self.compressed_trials,
            step=self._total_rehearsals,
        )
        report["state_hash"] = state_hash

        # ── Tier 2: World Model ──────────────────────────────────────
        wm_result = self._worldmodel_validate(
            features, best_chain, success_fn, self.worldmodel_trials,
        )
        report["worldmodel"] = wm_result
        report["tiers_completed"].append("worldmodel")

        if wm_result["success_rate"] < self.success_threshold:
            report["final_tier"] = "worldmodel_failed"
            return report

        self.chain_store.promote(
            state_hash, "worldmodel",
            success_rate=wm_result["success_rate"],
            trials=self.worldmodel_trials,
        )

        # ── Tier 3: Real Execution (always 10 save/load cycles) ──────
        real_result = self._real_validate(
            features, best_chain, env, self.real_trials,
        )
        report["real"] = real_result
        report["tiers_completed"].append("real")

        if real_result["success_rate"] < self.success_threshold:
            report["final_tier"] = "real_failed"
            return report

        self.chain_store.promote(
            state_hash, "proven",
            success_rate=real_result["success_rate"],
            trials=self.real_trials,
        )
        self._total_proven += 1

        report["final_tier"] = "proven"
        report["chain_length"] = len(best_chain)
        report["success"] = True

        # Record success in tracker
        self.tracker.record_success(features)

        return report

    # ══════════════════════════════════════════════════════════════════
    # MODE 2: FRONTIER — play from start, Advance on death
    # ══════════════════════════════════════════════════════════════════

    def run_frontier(
        self,
        env,
        max_episodes: int = 50,
        max_steps_per_episode: int = 10000,
    ) -> Dict[str, Any]:
        """
        Play from start using proven chains. On death → Advance mode.

        Returns:
            Report with frontier progress and rehearsal results.
        """
        self._frontier_count += 1
        report = {
            "mode": "frontier",
            "episodes": 0,
            "farthest_step": 0,
            "deaths": 0,
            "rehearsals_triggered": 0,
            "chains_proven": 0,
        }

        for ep in range(max_episodes):
            obs = env.reset()
            action = 0
            ep_step = 0
            features = self._get_features(obs)

            for step in range(max_steps_per_episode):
                # Check if we have a proven chain for this state
                chain = self.chain_store.recall(features)
                if chain and chain.is_proven and chain.actions:
                    # Execute proven chain
                    for chain_action in chain.actions:
                        obs, reward, done, info = env.step(chain_action)
                        ep_step += 1
                        if done:
                            break
                    features = self._get_features(obs)
                    if done:
                        break
                    continue

                # Normal policy action
                action = self._default_policy(features)
                obs, reward, done, info = env.step(action)
                features = self._get_features(obs)
                ep_step += 1

                if done:
                    break

            report["episodes"] = ep + 1
            report["farthest_step"] = max(report["farthest_step"], ep_step)

            if done and reward <= 0:
                # Death — enter Advance mode at this state
                report["deaths"] += 1
                self.tracker.record_death(features, {"episode": ep, "step": ep_step})

                # Save state and rehearse
                if hasattr(env, 'save_state') and env.supports_save_state:
                    advance_result = self.run_advance(features, env)
                    report["rehearsals_triggered"] += 1
                    if advance_result.get("success"):
                        report["chains_proven"] += 1
            else:
                self.tracker.record_success(features)

        return report

    # ══════════════════════════════════════════════════════════════════
    # MODE 3: STUCK — 10 failures → train flanking areas
    # ══════════════════════════════════════════════════════════════════

    def run_stuck(
        self,
        features: np.ndarray,
        env,
        max_iterations: int = 5,
    ) -> Dict[str, Any]:
        """
        After 10 failures at a frontier, train on flanking areas.

        Alternates between the state just BEFORE and the stuck state
        itself until progress or budget exhausted.
        """
        self._stuck_count += 1
        state_hash = self.tracker._hash_state(features)
        before_hash, after_hash = self.tracker.get_flanking_states(state_hash)

        report = {
            "mode": "stuck",
            "stuck_hash": state_hash,
            "before_hash": before_hash,
            "after_hash": after_hash,
            "iterations": 0,
            "before_results": [],
            "stuck_results": [],
            "resolved": False,
        }

        for iteration in range(max_iterations):
            report["iterations"] = iteration + 1

            # Train on state just before the stuck point
            if before_hash is not None:
                before_features = self.tracker.get_state_features(before_hash)
                if before_features is not None:
                    before_result = self.run_advance(before_features, env)
                    report["before_results"].append(before_result.get("success", False))

            # Train on the stuck state itself
            stuck_result = self.run_advance(features, env)
            report["stuck_results"].append(stuck_result.get("success", False))

            if stuck_result.get("success"):
                report["resolved"] = True
                break

        # If still stuck → flag for LLM review
        if not report["resolved"]:
            self.tracker._stuck_points.append({
                "state_hash": state_hash,
                "mode": "stuck_unresolved",
                "iterations": max_iterations,
                "step": self._total_rehearsals,
            })

        return report

    # ══════════════════════════════════════════════════════════════════
    # MODE 4: FREE RUN — play normally, log stuck points
    # ══════════════════════════════════════════════════════════════════

    def run_free(
        self,
        env,
        max_episodes: int = 100,
        max_steps_per_episode: int = 10000,
        death_threshold: int = 5,
    ) -> Dict[str, Any]:
        """
        Play normally using all proven chains. Does NOT pause.
        Logs stuck points when death count at a location exceeds threshold.

        Stuck points are stored for:
        - LLM review
        - Future Advance runs
        - Human review
        """
        self._free_run_count += 1
        report = {
            "mode": "free_run",
            "episodes": 0,
            "total_steps": 0,
            "total_reward": 0.0,
            "deaths": 0,
            "new_stuck_points": [],
            "chains_used": 0,
        }

        death_locations: Dict[int, int] = {}

        for ep in range(max_episodes):
            obs = env.reset()
            action = 0
            ep_reward = 0.0
            features = self._get_features(obs)

            for step in range(max_steps_per_episode):
                # Check for proven chain
                chain = self.chain_store.recall(features)
                if chain and chain.is_proven and chain.actions:
                    report["chains_used"] += 1
                    for chain_action in chain.actions:
                        obs, reward, done, info = env.step(chain_action)
                        ep_reward += reward
                        if done:
                            break
                    features = self._get_features(obs)
                    if done:
                        break
                    continue

                action = self._default_policy(features)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                features = self._get_features(obs)

                if done:
                    break

            report["episodes"] = ep + 1
            report["total_reward"] += ep_reward
            report["total_steps"] += step + 1

            if done and reward <= 0:
                report["deaths"] += 1
                state_hash = self.tracker.record_death(features, {
                    "episode": ep, "reward": ep_reward
                })

                death_locations[state_hash] = death_locations.get(state_hash, 0) + 1

                # Flag as potential stuck point if above threshold
                if death_locations[state_hash] == death_threshold:
                    stuck_info = {
                        "state_hash": state_hash,
                        "death_count": death_locations[state_hash],
                        "mode": "free_run_flagged",
                        "step": self._total_rehearsals,
                        "avg_reward": ep_reward,
                    }
                    report["new_stuck_points"].append(stuck_info)
            else:
                self.tracker.record_success(features)

        return report

    # ══════════════════════════════════════════════════════════════════
    # TIER IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════

    def _compressed_probe(
        self,
        features: np.ndarray,
        policy_fn: Callable,
        success_fn: Callable,
        n_trials: int,
    ) -> Dict[str, Any]:
        """
        Tier 1: Fast DQN rollouts from the given state.

        No environment needed — pure inference. Rolls out the policy
        forward using predicted next-states (or Q-value sampling).
        """
        successes = 0
        best_chain: List[int] = []
        best_score = -float("inf")

        for trial in range(n_trials):
            # Simulate a rollout from the current features
            sim_features = features.copy()
            chain: List[int] = []
            trajectory_reward = 0.0

            for step in range(self.max_chain_length):
                action = policy_fn(sim_features)
                chain.append(action)

                # Cheap forward prediction: perturb features based on action
                # This is intentionally rough — compressed tier is fast, not accurate
                noise = np.random.randn(len(sim_features)).astype(np.float32) * 0.05
                sim_features = sim_features + noise
                trajectory_reward += np.random.randn() * 0.1

                # Use world model prediction if available
                if self.brain.basal_ganglia._world_model is not None:
                    try:
                        pred_next, pred_reward = self.brain.basal_ganglia._world_model.predict(
                            sim_features, action,
                        )
                        sim_features = pred_next
                        trajectory_reward += pred_reward
                    except Exception:
                        pass

            # Check success
            trial_success = success_fn(chain, trajectory_reward)
            if trial_success:
                successes += 1
                if trajectory_reward > best_score:
                    best_score = trajectory_reward
                    best_chain = chain.copy()

        success_rate = successes / max(n_trials, 1)
        return {
            "success_rate": round(success_rate, 3),
            "best_chain": best_chain,
            "best_score": round(best_score, 4),
            "trials": n_trials,
            "successes": successes,
        }

    def _worldmodel_validate(
        self,
        features: np.ndarray,
        chain: List[int],
        success_fn: Callable,
        n_trials: int,
    ) -> Dict[str, Any]:
        """
        Tier 2: Validate chain using the WorldModel.

        More expensive than compressed but uses a trained simulation.
        """
        world_model = self.brain.basal_ganglia._world_model
        if world_model is None:
            return {"success_rate": 0.0, "trials": 0, "reason": "no_world_model"}

        successes = 0

        for trial in range(n_trials):
            sim_features = features.copy()
            trajectory_reward = 0.0

            for action in chain:
                try:
                    next_features, pred_reward = world_model.predict(sim_features, action)
                    sim_features = next_features
                    trajectory_reward += pred_reward
                except Exception:
                    break

            if success_fn(chain, trajectory_reward):
                successes += 1

        success_rate = successes / max(n_trials, 1)
        return {
            "success_rate": round(success_rate, 3),
            "trials": n_trials,
            "successes": successes,
        }

    def _real_validate(
        self,
        features: np.ndarray,
        chain: List[int],
        env,
        n_trials: int,
    ) -> Dict[str, Any]:
        """
        Tier 3: Validate chain with real save/load execution.

        ALWAYS runs exactly n_trials (10) cycles, even if succeeding.
        This provides robustness against stochastic timing.
        """
        if not (hasattr(env, 'save_state') and hasattr(env, 'load_state')):
            return {"success_rate": 0.0, "trials": 0, "reason": "no_save_state"}

        # Save the current state
        saved_state = env.save_state()
        if saved_state is None:
            return {"success_rate": 0.0, "trials": 0, "reason": "save_failed"}

        successes = 0

        # Always run exactly n_trials (10) for robustness
        for trial in range(n_trials):
            # Reload the saved state
            env.load_state(saved_state)

            total_reward = 0.0
            died = False

            for action in chain:
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done and reward <= 0:
                    died = True
                    break
                if done:
                    break

            if not died:
                successes += 1

        # Restore original state
        env.load_state(saved_state)

        success_rate = successes / max(n_trials, 1)
        return {
            "success_rate": round(success_rate, 3),
            "trials": n_trials,
            "successes": successes,
        }

    # ══════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════

    def _default_policy(self, features: np.ndarray) -> int:
        """Get action from brain's current policy."""
        try:
            if self.brain.striatum._torch_dqn is not None:
                action, _ = self.brain.striatum._torch_dqn.select_action(
                    features, explore=True,
                )
                return action
            else:
                q_vals = self.brain.striatum._forward(features)
                return int(np.argmax(q_vals))
        except Exception:
            return np.random.randint(self.brain.striatum._n_actions)

    def _default_success_check(
        self,
        chain: List[int],
        total_reward: float,
    ) -> bool:
        """Default success: survived the full chain without negative reward."""
        return len(chain) >= 10 and total_reward >= 0

    def _get_features(self, obs: np.ndarray) -> np.ndarray:
        """Extract features from observation using Sensory Cortex."""
        try:
            result = self.brain.sensory.process({"obs": obs, "action": 0, "reward": 0, "done": False})
            return result.get("features", obs)
        except Exception:
            return np.asarray(obs, dtype=np.float32).flatten()

    # ══════════════════════════════════════════════════════════════════
    # REPORTING
    # ══════════════════════════════════════════════════════════════════

    def report(self) -> Dict[str, Any]:
        return {
            "total_rehearsals": self._total_rehearsals,
            "total_proven": self._total_proven,
            "rehearsal_probability": round(self.rehearsal_probability, 3),
            "mode_counts": {
                "advance": self._advance_count,
                "frontier": self._frontier_count,
                "stuck": self._stuck_count,
                "free_run": self._free_run_count,
            },
            "bottlenecks": self.tracker.report(),
            "chains": self.chain_store.report(),
        }
