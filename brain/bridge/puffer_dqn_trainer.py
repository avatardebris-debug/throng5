"""
Vectorized DQN pretraining using PufferLib rollouts + Throng5 RainbowDQN.

PuffeRL trains PPO policies; this module uses PufferLib's vector env for fast
parallel stepping while keeping RainbowDQN checkpoints compatible with Striatum.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from brain.bridge.checkpoint import save_bridge_checkpoint
from brain.bridge.puffer_feature_env import make_feature_atari_env
from brain.config import ABSTRACT_VEC_SIZE
from brain.learning.torch_dqn import RainbowDQN, TORCH_AVAILABLE


@dataclass
class TrainConfig:
    game_id: str = "ALE/Breakout-v5"
    total_steps: int = 500_000
    num_envs: int = 32  # batched select_actions_batch; cap if falling back to serial
    num_workers: int = 1
    batch_size: int = 32
    train_interval: int = 4
    log_interval: int = 10_000
    seed: int = 42
    device: Optional[str] = None
    hidden_sizes: tuple = (256, 256, 128)
    lr: float = 6.25e-5
    export_path: str = "checkpoints/puffer_pretrain.pt"
    use_puffer_vector: bool = True
    replay_capacity: int = 100_000


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class _ReplayBuffer:
    """Simple replay for RainbowDQN.train_step (Striatum normally owns storage)."""

    def __init__(self, capacity: int, batch_size: int):
        self._buf: Deque[Transition] = deque(maxlen=capacity)
        self.batch_size = batch_size

    def push(self, transition: Transition) -> None:
        self._buf.append(transition)

    def sample(self) -> Optional[List[Transition]]:
        if len(self._buf) < self.batch_size:
            return None
        return random.sample(self._buf, self.batch_size)

    def __len__(self) -> int:
        return len(self._buf)


class PufferDQNTrainer:
    """Pretrain RainbowDQN with optional PufferLib vectorization."""

    def __init__(self, config: TrainConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for PufferDQNTrainer")
        self.config = config
        self._adapter_probe = None
        self._n_actions: Optional[int] = None

    def _probe_actions(self) -> int:
        if self._n_actions is not None:
            return self._n_actions
        from brain.bridge.puffer_feature_env import FeatureAtariEnv

        env = FeatureAtariEnv(game_id=self.config.game_id)
        self._n_actions = int(env.action_space.n)
        env.close()
        return self._n_actions

    def _make_vecenv(self):
        import pufferlib.vector

        cfg = self.config
        backend = (
            pufferlib.vector.Serial
            if cfg.num_workers <= 1
            else pufferlib.vector.Multiprocessing
        )
        workers = max(1, cfg.num_workers)
        batch = max(workers, min(cfg.batch_size, cfg.num_envs))
        return pufferlib.vector.make(
            make_feature_atari_env,
            num_envs=cfg.num_envs,
            num_workers=workers,
            batch_size=batch,
            backend=backend,
            env_kwargs={"game_id": cfg.game_id},
            seed=cfg.seed,
        )

    def train(self) -> Dict[str, Any]:
        cfg = self.config
        n_actions = self._probe_actions()
        dqn = RainbowDQN(
            n_features=ABSTRACT_VEC_SIZE,
            n_actions=n_actions,
            hidden_sizes=cfg.hidden_sizes,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )

        use_vector = cfg.use_puffer_vector
        vecenv = None
        if use_vector:
            try:
                vecenv = self._make_vecenv()
            except ImportError as e:
                print(f"[puffer_pretrain] PufferLib unavailable ({e}); using serial fallback.")
                use_vector = False
                if cfg.num_envs > 1:
                    print(
                        f"[puffer_pretrain] Serial backend uses one env; "
                        f"ignoring num_envs={cfg.num_envs}."
                    )
                    cfg.num_envs = 1

        t0 = time.time()
        replay = _ReplayBuffer(cfg.replay_capacity, cfg.batch_size)

        if use_vector and vecenv is not None:
            stats = self._train_vector(dqn, vecenv, replay)
            vecenv.close()
        else:
            stats = self._train_serial(dqn, replay)

        export = Path(cfg.export_path)
        save_bridge_checkpoint(dqn, export, game_id=cfg.game_id, extra=stats)
        stats["export_path"] = str(export)
        stats["elapsed_sec"] = round(time.time() - t0, 2)
        stats["sps"] = round(stats["steps"] / max(stats["elapsed_sec"], 1e-6), 1)
        return stats

    def _maybe_train(self, dqn: RainbowDQN, replay: _ReplayBuffer, losses: list) -> None:
        batch = replay.sample()
        if batch is None:
            return
        out = dqn.train_step(batch=batch)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.get("loss", 0.0)
        if loss:
            losses.append(float(loss))

    def _train_vector(self, dqn: RainbowDQN, vecenv, replay: _ReplayBuffer) -> Dict[str, Any]:
        cfg = self.config
        vecenv.async_reset(seed=cfg.seed)
        obs, _, terminals, truncations, _, _, _ = vecenv.recv()
        n_agents = vecenv.num_agents
        prev_obs = obs.copy()
        episode_returns = np.zeros(n_agents, dtype=np.float64)
        ep_count = 0
        losses = []

        for step in range(1, cfg.total_steps + 1):
            if hasattr(dqn, "select_actions_batch"):
                actions, _ = dqn.select_actions_batch(obs, explore=True)
            else:
                actions = np.zeros(n_agents, dtype=np.int64)
                for i in range(n_agents):
                    a, _ = dqn.select_action(obs[i], explore=True)
                    actions[i] = a

            vecenv.send(actions)
            next_obs, rewards, terminals, truncations, _, _, _ = vecenv.recv()
            dones = np.logical_or(terminals, truncations)

            for i in range(n_agents):
                replay.push((
                    prev_obs[i].copy(), int(actions[i]), float(rewards[i]),
                    next_obs[i].copy(), bool(dones[i]),
                ))
                episode_returns[i] += rewards[i]
                if dones[i]:
                    ep_count += 1
                    episode_returns[i] = 0.0

            if step % cfg.train_interval == 0:
                self._maybe_train(dqn, replay, losses)

            if step % cfg.log_interval == 0:
                avg_loss = float(np.mean(losses[-100:])) if losses else 0.0
                print(
                    f"[puffer_pretrain] step {step}/{cfg.total_steps} "
                    f"updates={dqn._total_updates} loss={avg_loss:.4f} eps={ep_count}"
                )

            prev_obs = next_obs.copy()
            if step >= cfg.total_steps:
                break

        return {
            "steps": step,
            "episodes": ep_count,
            "total_updates": dqn._total_updates,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "num_envs": n_agents,
            "backend": "puffer_vector",
        }

    def _train_serial(self, dqn: RainbowDQN, replay: _ReplayBuffer) -> Dict[str, Any]:
        from brain.bridge.puffer_feature_env import FeatureAtariEnv

        cfg = self.config
        env = FeatureAtariEnv(game_id=cfg.game_id)
        obs, _ = env.reset(seed=cfg.seed)
        ep_reward = 0.0
        ep_count = 0
        losses = []

        for step in range(1, cfg.total_steps + 1):
            action, _ = dqn.select_action(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push((obs.copy(), action, reward, next_obs.copy(), done))
            ep_reward += reward

            if step % cfg.train_interval == 0:
                self._maybe_train(dqn, replay, losses)

            if done:
                ep_count += 1
                ep_reward = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs

            if step % cfg.log_interval == 0:
                print(
                    f"[puffer_pretrain] serial step {step}/{cfg.total_steps} "
                    f"updates={dqn._total_updates}"
                )

        env.close()
        return {
            "steps": cfg.total_steps,
            "episodes": ep_count,
            "total_updates": dqn._total_updates,
            "mean_loss": float(np.mean(losses)) if losses else 0.0,
            "num_envs": 1,
            "backend": "serial",
        }
