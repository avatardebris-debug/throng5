"""Gymnasium-compatible Mega Man 2 env via nes-py."""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np

from brain.environments.nes.action_map import N_ACTIONS, action_to_bitmask
from brain.environments.nes.feature_adapter import NESAdapter
from brain.environments.nes.paths import DEFAULT_ROM, DEFAULT_STATE
try:
    import gymnasium as gym

    Box = gym.spaces.Box
    Discrete = gym.spaces.Discrete
except ImportError:
    from brain.environments.nes.spaces import Box, Discrete

try:
    import nes_py

    NES_PY_AVAILABLE = True
except ImportError:
    NES_PY_AVAILABLE = False


class MegaManEnv:
    """Mega Man 2 with saved input sequence fast-forward on reset."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        rom_path: str = DEFAULT_ROM,
        state_path: str = DEFAULT_STATE,
        n_features: int = 84,
        render: bool = False,
        max_steps: int = 4500,
        death_penalty: float = -10.0,
        progress_reward_scale: float = 0.01,
        frame_skip: int = 4,
    ):
        if not NES_PY_AVAILABLE:
            raise ImportError("nes-py not installed. Run: pip install nes-py")
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")

        self.rom_path = rom_path
        self.state_path = state_path
        self.n_features = n_features
        self.render_mode = "human" if render else "rgb_array"
        self.max_steps = max_steps
        self.death_penalty = death_penalty
        self.progress_reward_scale = progress_reward_scale
        self.frame_skip = max(1, frame_skip)

        self._start_sequence: list = []
        seq_path = state_path.replace(
            "megaman2_stage_start.pkl", "megaman2_input_sequence.pkl"
        )
        if os.path.exists(seq_path):
            with open(seq_path, "rb") as f:
                saved = pickle.load(f)
            self._start_sequence = saved.get("sequence", [])
            print(
                f"[MegaManEnv] Loaded input sequence: "
                f"{len(self._start_sequence)} frames to stage start"
            )
        elif state_path and os.path.exists(state_path):
            print("[MegaManEnv] WARNING: RAM state restore is unreliable in nes-py.")
            print("  Run tools/record_start_sequence.py to record menu navigation.")
        else:
            print("[MegaManEnv] No start sequence — starting from title screen.")
            print("  Run tools/record_start_sequence.py to record menu navigation.")

        self._env = nes_py.NESEnv(rom_path)
        self._adapter = NESAdapter(n_features=n_features)
        self.observation_space = Box(
            low=-5.0, high=5.0, shape=(n_features,), dtype=np.float32,
        )
        self.action_space = Discrete(N_ACTIONS)

        self._steps = 0
        self._prev_x = 0
        self._lives = 3
        self._done = False
        self._last_raw = None

    def reset(self) -> np.ndarray:
        obs = self._env.reset()
        if self._start_sequence:
            for attempt in range(3):
                obs = self._env.reset()
                aborted = False
                for bitmask in self._start_sequence:
                    obs, _, done, _ = self._env.step(bitmask)
                    if done:
                        aborted = True
                        break
                if not aborted:
                    break

        self._steps = 0
        self._prev_x = int(self._env.ram[0xEA])
        self._lives = 3
        self._done = False
        return self._adapter.observe_and_extract(obs)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._done:
            return self._adapter.last_features, 0.0, True, {}

        bitmask = action_to_bitmask(action)
        obs = None
        env_done = False
        for _ in range(self.frame_skip):
            obs, _, env_done, _ = self._env.step(bitmask)
            if env_done:
                break
        self._last_raw = obs
        self._steps += 1

        reward = 0.0
        try:
            curr_x = int(self._env.ram[0xEA])
            dx = curr_x - self._prev_x
            if dx > 128:
                dx -= 256
            if dx < -128:
                dx += 256
            reward += dx * self.progress_reward_scale
            self._prev_x = curr_x
        except Exception:
            pass

        if bitmask & 0x80:
            reward += 0.002
        if bitmask & 0x01:
            reward += 0.001
        if bitmask & 0x02:
            reward += 0.001
        reward -= 0.003

        death = False
        try:
            lives_now = int(self._env.ram[0xA1BB] & 0x0F)
            death = lives_now < self._lives
            self._lives = lives_now
            if death:
                reward += self.death_penalty
        except Exception:
            pass

        done = env_done or self._steps >= self.max_steps or death
        features = self._adapter.observe_and_extract(obs)
        return features, float(reward), done, {
            "x_pos": self._prev_x,
            "lives": self._lives,
            "steps": self._steps,
        }

    def render(self) -> Optional[np.ndarray]:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    @property
    def n_actions(self) -> int:
        return N_ACTIONS
