"""
lolo_gan_trainer.py — Multi-stage GAN training loop for Lolo puzzles.

Pipeline:
  1. GAN generates puzzle
  2. SARSA gets 5 tries → Solved? → graded bank
  3. Unsolved accumulate until 100
  4. SARSA gets 25 tries → Solved? → hard bank
  5. PPO+HER gets 1000 eps → Solved? → expert bank
  6. Still unsolved → flagged as unsolvable → negative GAN signal

The GAN learns from labeled results:
  - "Good" puzzles: solved in 5–25 tries (sweet spot)
  - "Bad" puzzles: unsolvable or solved on first try (too easy)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.games.lolo.lolo_gan import LoloGAN, GRID_CELLS, N_CHANNELS
from brain.games.lolo.lolo_sarsa_learner import LoloSarsaLearner
from brain.games.lolo.lolo_simulator import Enemy, EnemyType, LoloSimulator
from brain.games.lolo.lolo_generator import LoloPuzzleGenerator


class PuzzleRecord:
    """Tracks a generated puzzle through the validation pipeline."""
    __slots__ = ("grid_probs", "sim_state", "tier", "stage",
                 "sarsa_tries", "sarsa_wins", "ppo_result", "label")

    def __init__(self, grid_probs: np.ndarray, sim_state: dict, tier: int):
        self.grid_probs = grid_probs      # (143, 9) for GAN training
        self.sim_state = sim_state        # sim.save() for replay
        self.tier = tier
        self.stage = 1                    # 1=quick, 2=retry, 3=ppo
        self.sarsa_tries = 0
        self.sarsa_wins = 0
        self.ppo_result = None            # "solved" | "unsolvable" | None
        self.label = None                 # "good" | "too_easy" | "unsolvable"


class GanTrainingLoop:
    """
    Multi-stage GAN training orchestrator.

    Generates puzzles, validates through 3 stages, feeds results back
    to the GAN for adversarial training.
    """

    STAGE1_TRIES = 5
    STAGE2_TRIES = 25
    STAGE3_EPISODES = 1000
    UNSOLVED_BATCH_SIZE = 100
    MAX_STEPS_PER_EPISODE = 300

    def __init__(
        self,
        gan: Optional[LoloGAN] = None,
        sarsa: Optional[LoloSarsaLearner] = None,
        tier: int = 1,
        seed_count: int = 500,
    ):
        self.gan = gan or LoloGAN()
        self.sarsa = sarsa or LoloSarsaLearner(n_actions=6, epsilon_decay=0.999)
        self.tier = tier
        self.seed_count = seed_count

        # Puzzle banks
        self.graded_bank: List[PuzzleRecord] = []    # Solved in 5 tries
        self.hard_bank: List[PuzzleRecord] = []      # Solved in 25 tries
        self.expert_bank: List[PuzzleRecord] = []    # Solved by PPO+HER
        self.unsolvable: List[PuzzleRecord] = []     # Confirmed unsolvable
        self.unsolved_queue: List[PuzzleRecord] = [] # Awaiting retry

        # Stats
        self._total_generated = 0
        self._total_validated = 0
        self._gan_train_steps = 0
        self._seeded = False

    def seed_with_random(self, n: int = 500, tier: int = 1) -> Dict[str, int]:
        """
        Seed the GAN discriminator with randomly generated puzzles.

        Uses a decaying trial schedule: early puzzles get many SARSA
        episodes (200) so it can learn the game, later puzzles need
        fewer because Q-values are built up.

        Schedule: 200 → -10 per puzzle to 100, -5 to 50, -1 to 5
        """
        from brain.games.lolo.lolo_generator import LoloPuzzleGenerator

        gen = LoloPuzzleGenerator(seed=42)
        good_grids = []
        bad_grids = []
        solved = 0
        unsolvable_count = 0
        current_tries = 200

        for i in range(n):
            sim = gen.generate(tier=tier, max_attempts=500)
            if sim is None:
                continue

            grid_probs = self._sim_to_gan_grid(sim)
            sim_state = sim.save()

            won = self._try_sarsa(sim, sim_state, current_tries)

            if won:
                good_grids.append(grid_probs)
                self.gan.add_solved(grid_probs)  # Feed to generator bank
                solved += 1
            else:
                bad_grids.append(grid_probs)
                unsolvable_count += 1

            # Decay trial count
            if current_tries > 100:
                current_tries = max(100, current_tries - 10)
            elif current_tries > 50:
                current_tries = max(50, current_tries - 5)
            elif current_tries > 5:
                current_tries = max(5, current_tries - 1)

            # Train GAN discriminator periodically
            if len(good_grids) >= 10 and len(bad_grids) >= 5:
                self.gan.train_step(good_grids[-10:], bad_grids[-5:])
                self._gan_train_steps += 1

            # Progress reporting
            if (i + 1) % 50 == 0:
                rate = solved / (i + 1)
                print(f"  Seed [{i+1}/{n}] solved={solved} ({rate:.0%}) "
                      f"tries={current_tries} Q={len(self.sarsa.q_table)}",
                      flush=True)

        # Final discriminator training
        if good_grids and bad_grids:
            self.gan.train_step(good_grids, bad_grids)
            self._gan_train_steps += 1

        # ── Pre-train generator from solved puzzles ──
        if self.gan.solved_bank:
            n_epochs = min(100, max(20, len(self.gan.solved_bank) * 3))
            print(f"  Pre-training generator on {len(self.gan.solved_bank)} "
                  f"solved puzzles ({n_epochs} epochs)...", flush=True)
            pt_result = self.gan.pretrain_from_solved(epochs=n_epochs, batch_size=16)
            print(f"  Pre-train done: loss={pt_result['pretrain_loss']:.4f}, "
                  f"steps={pt_result['steps']}", flush=True)

        self._seeded = True
        return {"seeded": n, "solved": solved, "unsolvable": unsolvable_count,
                "gan_steps": self._gan_train_steps,
                "final_tries": current_tries}

    def run(
        self,
        n_puzzles: int = 100,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Main training loop: generate n puzzles through the full pipeline.
        """
        t0 = time.time()

        # Snapshot bank sizes so we return only THIS run's counts
        graded_before = len(self.graded_bank)
        hard_before = len(self.hard_bank)
        expert_before = len(self.expert_bank)
        unsolvable_before = len(self.unsolvable)
        generated_before = self._total_generated

        for i in range(n_puzzles):
            # ── Generate puzzle ──
            sim = self.gan.generate(tier=self.tier, temperature=0.8)
            if sim is None:
                continue

            grid_probs = self._sim_to_gan_grid(sim)
            sim_state = sim.save()
            record = PuzzleRecord(grid_probs, sim_state, self.tier)
            self._total_generated += 1

            # ── Stage 1: Quick SARSA (5 tries) ──
            won = self._try_sarsa(sim, sim_state, self.STAGE1_TRIES)
            record.sarsa_tries = self.STAGE1_TRIES
            record.sarsa_wins = 1 if won else 0

            if won:
                record.label = "good"
                record.stage = 1
                self.graded_bank.append(record)
            else:
                self.unsolved_queue.append(record)

            # ── Stage 2: Batch retry when 100 unsolved ──
            if len(self.unsolved_queue) >= self.UNSOLVED_BATCH_SIZE:
                self._run_stage2(verbose)

            # ── Train GAN periodically ──
            if (i + 1) % 20 == 0:
                self._train_gan_from_banks()

            if verbose and (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  GAN [{i+1}/{n_puzzles}] "
                      f"graded={len(self.graded_bank)} "
                      f"hard={len(self.hard_bank)} "
                      f"unsolved={len(self.unsolved_queue)} "
                      f"unsolvable={len(self.unsolvable)} "
                      f"| {elapsed:.1f}s", flush=True)

        # Process remaining unsolved
        if self.unsolved_queue:
            self._run_stage2(verbose)

        elapsed = time.time() - t0

        # Return only THIS run's contribution (delta from snapshot)
        run_graded = len(self.graded_bank) - graded_before
        run_hard = len(self.hard_bank) - hard_before
        run_expert = len(self.expert_bank) - expert_before
        run_unsolvable = len(self.unsolvable) - unsolvable_before
        run_generated = self._total_generated - generated_before

        return {
            "generated": run_generated,
            "graded": run_graded,
            "hard": run_hard,
            "expert": run_expert,
            "solved": run_graded + run_hard + run_expert,
            "unsolvable": run_unsolvable,
            "unsolved_remaining": len(self.unsolved_queue),
            "gan_train_steps": self._gan_train_steps,
            "elapsed": round(elapsed, 1),
            "gan": self.gan.report(),
        }

    def _run_stage2(self, verbose: bool = False) -> None:
        """Stage 2: retry unsolved puzzles with 25 SARSA tries each."""
        batch = self.unsolved_queue[:self.UNSOLVED_BATCH_SIZE]
        self.unsolved_queue = self.unsolved_queue[self.UNSOLVED_BATCH_SIZE:]
        still_unsolved = []

        for record in batch:
            sim = self._make_sim_from_state(record.sim_state)
            if sim is None:
                continue

            won = self._try_sarsa(sim, record.sim_state, self.STAGE2_TRIES)
            record.sarsa_tries += self.STAGE2_TRIES
            record.stage = 2

            if won:
                record.sarsa_wins += 1
                record.label = "good"  # Solved in 25 — sweet spot!
                self.hard_bank.append(record)
            else:
                still_unsolved.append(record)

        # Stage 3: PPO+HER for remaining
        if still_unsolved:
            self._run_stage3(still_unsolved, verbose)

    def _run_stage3(self, records: List[PuzzleRecord], verbose: bool = False) -> None:
        """Stage 3: PPO+HER validation for stubborn puzzles."""
        for record in records:
            record.stage = 3

            sim = self._make_sim_from_state(record.sim_state)
            if sim is None:
                record.label = "unsolvable"
                record.ppo_result = "error"
                self.unsolvable.append(record)
                continue

            won = self._try_ppo_her(sim, record.sim_state, self.STAGE3_EPISODES)

            if won:
                record.ppo_result = "solved"
                record.label = "good"
                self.expert_bank.append(record)
            else:
                record.ppo_result = "unsolvable"
                record.label = "unsolvable"
                self.unsolvable.append(record)
                self._total_validated += 1

    def _try_sarsa(
        self, sim: LoloSimulator, state: dict, n_tries: int,
    ) -> bool:
        """Run SARSA for n_tries episodes on the same puzzle. Returns True if ever won."""
        for ep in range(n_tries):
            sim.load(state)
            self.sarsa.set_simulator(sim)
            self.sarsa.set_puzzle_id(hash(str(state)) % 10000)

            reward = 0.0
            for step in range(self.MAX_STEPS_PER_EPISODE):
                result = self.sarsa.step(reward=reward, done=False)
                _, reward, done, _ = sim.step(result["action"])
                if done:
                    self.sarsa.step(reward=reward, done=True)
                    break
            else:
                self.sarsa.step(reward=-1.0, done=True)

            if sim.won:
                return True
        return False

    def _try_ppo_her(
        self, sim: LoloSimulator, state: dict, n_episodes: int,
    ) -> bool:
        """Run PPO+HER for n_episodes. Returns True if ever solved."""
        # Import inline to avoid circular deps if PPO not available
        try:
            from brain.games.lolo.lolo_ppo_her import PPO_HER_Learner
        except ImportError:
            # Fallback: use extended SARSA (more tries)
            return self._try_sarsa(sim, state, n_episodes)

        learner = PPO_HER_Learner()
        for ep in range(n_episodes):
            sim.load(state)
            learner.set_simulator(sim)
            learner.set_puzzle_id(0)

            reward = 0.0
            for step in range(self.MAX_STEPS_PER_EPISODE):
                result = learner.step(reward=reward, done=False)
                _, reward, done, _ = sim.step(result["action"])
                if done:
                    learner.step(reward=reward, done=True)
                    break
            else:
                learner.step(reward=-1.0, done=True)
            learner.end_episode()

            if sim.won:
                return True
        return False

    def _train_gan_from_banks(self) -> None:
        """Train GAN using labeled puzzle banks."""
        good = []
        bad = []

        # Good: recently solved puzzles (graded + hard)
        for record in (self.graded_bank[-20:] + self.hard_bank[-10:]):
            good.append(record.grid_probs)

        # Bad: unsolvable puzzles
        for record in self.unsolvable[-10:]:
            bad.append(record.grid_probs)

        if good and bad:
            self.gan.train_step(good, bad)
            self._gan_train_steps += 1

    def _make_sim_from_state(self, state: dict) -> Optional[LoloSimulator]:
        """Reconstruct a simulator from saved state."""
        try:
            grid = state["grid"].copy()
            enemies = []
            for ed in state.get("enemies", []):
                enemies.append(Enemy(
                    etype=EnemyType(ed["etype"]),
                    row=ed["row"], col=ed["col"],
                ))
            magic = set(tuple(p) for p in state.get("magic_shot_hearts", []))
            sim = LoloSimulator(grid, enemies, magic)
            sim.load(state)
            return sim
        except Exception:
            return None

    def _sim_to_gan_grid(self, sim: LoloSimulator) -> np.ndarray:
        """Convert a LoloSimulator grid to (143, 9) one-hot format."""
        grid_probs = np.zeros((GRID_CELLS, N_CHANNELS), dtype=np.float32)

        tile_to_chan = {
            0: 0,  # EMPTY
            1: 1,  # ROCK
            2: 2,  # TREE
            3: 3,  # HEART
            4: 4,  # EMERALD
            5: 5,  # CHEST
            6: 6,  # EXIT
            7: 7,  # WATER
            16: 0, # PLAYER → EMPTY (player placed separately)
        }

        for r in range(sim.GRID_H):
            for c in range(sim.GRID_W):
                idx = r * sim.GRID_W + c
                tile = int(sim.grid[r, c])
                chan = tile_to_chan.get(tile, 0)
                grid_probs[idx, chan] = 1.0

        # Mark enemy positions
        for enemy in sim.enemies:
            idx = enemy.row * sim.GRID_W + enemy.col
            grid_probs[idx, :] = 0.0
            grid_probs[idx, 8] = 1.0  # ENEMY channel

        return grid_probs

    def report(self) -> Dict[str, Any]:
        return {
            "total_generated": self._total_generated,
            "graded_bank": len(self.graded_bank),
            "hard_bank": len(self.hard_bank),
            "expert_bank": len(self.expert_bank),
            "unsolvable": len(self.unsolvable),
            "unsolved_queue": len(self.unsolved_queue),
            "gan_train_steps": self._gan_train_steps,
            "seeded": self._seeded,
        }
