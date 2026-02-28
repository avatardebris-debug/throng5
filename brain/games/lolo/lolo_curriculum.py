"""
lolo_curriculum.py — Tiered curriculum training for Adventures of Lolo.

Orchestrates the graduated training pipeline:
  1. Generate puzzles at current tier
  2. Train agent through LoloSimulator (fast)
  3. Track success rate → advance tier when mastered
  4. Apply 3-tier validation: compressed → world model → real NES

Usage:
    curriculum = LoloCurriculum(brain, generator, adapter)
    curriculum.train_tier(tier=1, n_episodes=500)
    if curriculum.should_advance():
        curriculum.advance()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.games.lolo.lolo_adapter import LoloAdapter
from brain.games.lolo.lolo_generator import LoloPuzzleGenerator, TIER_CONFIG
from brain.games.lolo.lolo_simulator import Action, LoloSimulator


class LoloCurriculum:
    """
    Tiered curriculum with graduated difficulty.

    Each tier must reach >60% success rate before advancing.
    Training uses the fast LoloSimulator (no NES emulation needed).
    """

    ADVANCE_THRESHOLD = 0.6  # 60% success to advance
    MAX_STEPS_PER_EPISODE = 500
    DEAD_END_CHECK_INTERVAL = 20  # Check for dead-end every N steps
    MAX_FAILURES_BEFORE_FLAG = 3  # Flag puzzle after N consecutive failures

    def __init__(
        self,
        brain,
        generator: Optional[LoloPuzzleGenerator] = None,
        adapter: Optional[LoloAdapter] = None,
        seed: int = 42,
    ):
        self.brain = brain
        self.gen = generator or LoloPuzzleGenerator(seed=seed)
        self.adapter = adapter or LoloAdapter(feature_dim=84)

        # Stats per tier
        self.tier_stats: Dict[int, Dict[str, Any]] = {}
        self._current_tier = 1
        self._total_episodes = 0
        self._total_steps = 0

        # Flagged puzzles — puzzles that consistently fail (for review)
        self.flagged_puzzles: List[Dict[str, Any]] = []
        self._dead_ends_detected = 0

    @property
    def current_tier(self) -> int:
        return self._current_tier

    def train_tier(
        self,
        tier: Optional[int] = None,
        n_episodes: int = 100,
        max_steps: int = None,
        verbose: bool = False,
        fixed_puzzles: int = 0,  # 0=random each time, >0=pre-generate and cycle
    ) -> Dict[str, Any]:
        """
        Train agent on n_episodes puzzles at this tier.

        Args:
            fixed_puzzles: If >0, pre-generate this many puzzles and cycle
                          through them. Enables learners to memorize and
                          generalize from repeated exposure.

        Returns tier results: success_rate, avg_reward, avg_steps.
        """
        tier = tier or self._current_tier
        max_steps = max_steps or self.MAX_STEPS_PER_EPISODE

        results = {
            "tier": tier,
            "tier_name": TIER_CONFIG.get(tier, {}).get("name", "unknown"),
            "episodes": 0,
            "successes": 0,
            "deaths": 0,
            "timeouts": 0,
            "dead_ends": 0,
            "flagged": 0,
            "total_reward": 0.0,
            "total_steps": 0,
        }

        # Pre-generate fixed puzzle bank if requested
        puzzle_bank = None
        if fixed_puzzles > 0:
            puzzle_bank = []
            for _ in range(fixed_puzzles):
                sim = self.gen.generate(tier)
                if sim is not None:
                    puzzle_bank.append(sim.save())
            if verbose:
                print(f"  Puzzle bank: {len(puzzle_bank)} fixed puzzles", flush=True)

        for ep in range(n_episodes):
            # Generate or reload a puzzle
            if puzzle_bank:
                # Cycle through fixed puzzles
                pid = ep % len(puzzle_bank)
                state = puzzle_bank[pid]
                sim = self.gen.generate(tier)
                if sim is None:
                    continue
                sim.load(state)
                # Tell fast learner which puzzle this is
                if hasattr(self.brain, 'set_puzzle_id'):
                    self.brain.set_puzzle_id(pid)
            else:
                sim = self.gen.generate(tier)
                if sim is None:
                    continue

            # Save initial state for flagging
            initial_state = sim.save()

            success, reward, steps, info = self._run_episode(sim, max_steps)

            results["episodes"] += 1
            results["total_reward"] += reward
            results["total_steps"] += steps
            self._total_episodes += 1
            self._total_steps += steps

            if success:
                results["successes"] += 1
            elif info.get("dead_end"):
                results["dead_ends"] += 1
                self._dead_ends_detected += 1
            elif info.get("death"):
                results["deaths"] += 1
            else:
                results["timeouts"] += 1
                # Flag puzzles that time out — may be too hard or unsolvable
                self._maybe_flag_puzzle(initial_state, tier, info)

            if verbose and (ep + 1) % 50 == 0:
                sr = results["successes"] / results["episodes"]
                print(f"  Tier {tier} ep {ep + 1}/{n_episodes}: "
                      f"success={sr:.1%}, deaths={results['deaths']}")

        # Compute summary stats
        n = max(1, results["episodes"])
        results["success_rate"] = round(results["successes"] / n, 3)
        results["avg_reward"] = round(results["total_reward"] / n, 2)
        results["avg_steps"] = round(results["total_steps"] / n, 1)

        self.tier_stats[tier] = results
        return results

    def _run_episode(
        self,
        sim: LoloSimulator,
        max_steps: int,
    ) -> Tuple[bool, float, int, Dict[str, Any]]:
        """Run one episode through the brain. Returns (success, total_reward, steps, last_info)."""
        # Detect fast learner — bypass adapter for speed
        is_fast = hasattr(self.brain, 'q_table')

        if is_fast:
            # Give fast learner direct simulator access for state reading
            if hasattr(self.brain, 'set_simulator'):
                self.brain.set_simulator(sim)
            features = None  # Not used — fast learner reads sim directly
        else:
            features = self.adapter.reset(sim)

        total_reward = 0.0
        prev_action = 0
        prev_reward = 0.0  # Per-step reward for Q-learning
        last_info = {}

        for step in range(max_steps):
            # ── Dead-end detection (every N steps) ────────────────────
            if (step > 0
                    and step % self.DEAD_END_CHECK_INTERVAL == 0
                    and sim.is_dead_end()):
                last_info["dead_end"] = True
                last_info["dead_end_step"] = step
                self.brain.step(
                    features,
                    prev_action=prev_action,
                    reward=-1.0,
                    done=True,
                )
                return False, total_reward - 1.0, step, last_info

            # Get action from brain — pass per-step reward, not accumulated
            result = self.brain.step(
                features,
                prev_action=prev_action,
                reward=prev_reward,
                done=False,
            )
            action = result["action"]

            # Map brain action (0-17) to Lolo action (0-5)
            lolo_action = action if is_fast else self._map_action(action)

            # Step simulator — bypass adapter for fast learner
            if is_fast:
                raw_obs, reward, done, info = sim.step(lolo_action)
                features = None  # Not used
            else:
                features, reward, done, info = self.adapter.step(lolo_action)

            total_reward += reward
            prev_reward = reward  # Store per-step reward for next iteration
            prev_action = action
            last_info = info

            if done:
                self.brain.step(
                    features,
                    prev_action=prev_action,
                    reward=reward,
                    done=True,
                )
                return sim.won, total_reward, step + 1, info

        # Timeout
        self.brain.step(features, prev_action=prev_action, reward=0, done=True)
        return False, total_reward, max_steps, last_info

    def _maybe_flag_puzzle(
        self,
        initial_state: Dict[str, Any],
        tier: int,
        info: Dict[str, Any],
    ) -> None:
        """Flag puzzles that consistently fail for human review."""
        # Only flag if we have room (cap at 50 to avoid memory bloat)
        if len(self.flagged_puzzles) >= 50:
            return

        self.flagged_puzzles.append({
            "tier": tier,
            "initial_state": initial_state,
            "reason": "timeout",
            "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))},
            "episode": self._total_episodes,
        })

    def _map_action(self, brain_action: int) -> int:
        """
        Map brain's 18-action space to Lolo's 6 actions.

        Simple modular mapping:
          0 → NOOP, 1 → UP, 2 → DOWN, 3 → LEFT, 4 → RIGHT, 5 → SHOOT
          6+ → wraps around
        """
        return brain_action % 6

    def should_advance(self) -> bool:
        """Has current tier been mastered (>60% success)?"""
        stats = self.tier_stats.get(self._current_tier)
        if stats is None:
            return False
        return stats["success_rate"] >= self.ADVANCE_THRESHOLD

    def advance(self) -> int:
        """Advance to next tier. Returns new tier."""
        self._current_tier = min(self._current_tier + 1, 7)
        self.gen.advance_tier()
        return self._current_tier

    def run_full_curriculum(
        self,
        episodes_per_tier: int = 500,
        max_tiers: int = 7,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full graduated curriculum from tier 1 to max_tiers.

        Advances when success rate > 60%.
        """
        all_results = {}
        start_time = time.time()

        for tier in range(1, max_tiers + 1):
            if verbose:
                desc = TIER_CONFIG.get(tier, {}).get("desc", "")
                print(f"\n{'='*60}")
                print(f"TIER {tier}: {desc}")
                print(f"{'='*60}")

            results = self.train_tier(
                tier=tier,
                n_episodes=episodes_per_tier,
                verbose=verbose,
            )
            all_results[f"tier_{tier}"] = results

            if verbose:
                print(f"  Result: {results['success_rate']:.1%} success, "
                      f"avg reward={results['avg_reward']:.1f}")

            if results["success_rate"] >= self.ADVANCE_THRESHOLD:
                if verbose:
                    print(f"  [OK] Mastered tier {tier}! Advancing...")
                self.advance()
            else:
                if verbose:
                    print(f"  [--] Not mastered yet ({results['success_rate']:.1%} < "
                          f"{self.ADVANCE_THRESHOLD:.0%}). Stopping.")
                break

        elapsed = time.time() - start_time
        return {
            "tiers_completed": self._current_tier - 1,
            "highest_tier_attempted": max(all_results.keys()) if all_results else 0,
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "elapsed_seconds": round(elapsed, 1),
            "tier_results": all_results,
        }

    def report(self) -> Dict[str, Any]:
        return {
            "current_tier": self._current_tier,
            "total_episodes": self._total_episodes,
            "total_steps": self._total_steps,
            "dead_ends_detected": self._dead_ends_detected,
            "flagged_puzzles": len(self.flagged_puzzles),
            "tier_stats": {
                tier: {
                    "success_rate": s["success_rate"],
                    "avg_reward": s["avg_reward"],
                    "episodes": s["episodes"],
                }
                for tier, s in self.tier_stats.items()
            },
            "generator": self.gen.report(),
        }

    def validate_solvable(
        self,
        sim: LoloSimulator,
        max_episodes: int = 1000,
        max_steps: int = 300,
    ) -> bool:
        """
        Quick solvability check using TabQ.

        Runs a fast Q-learner on the puzzle. If it solves it at least once
        in max_episodes, the puzzle is solvable. Otherwise flag it.

        Returns True if solvable.
        """
        from brain.games.lolo.lolo_fast_learner import LoloFastLearner

        state = sim.save()
        validator = LoloFastLearner(n_actions=6, lr=0.3, epsilon_decay=0.995)
        validator.set_puzzle_id(0)

        for ep in range(max_episodes):
            sim.load(state)
            validator.set_simulator(sim)
            reward = 0.0

            for step in range(max_steps):
                result = validator.step(reward=reward, done=False)
                action = result["action"]
                _, reward, done, _ = sim.step(action)
                if done:
                    validator.step(reward=reward, done=True)
                    break
            else:
                validator.step(reward=-1.0, done=True)

            if sim.won:
                sim.load(state)  # Restore original state
                return True

        sim.load(state)  # Restore original state
        return False
