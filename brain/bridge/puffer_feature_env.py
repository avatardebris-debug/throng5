"""
Gymnasium env that exposes Throng5's 84-dim abstract Atari features.

Used with PufferLib vectorization so rollouts match WholeBrain / Striatum input shape.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from brain.config import ABSTRACT_VEC_SIZE
from brain.environments.atari_adapter import AtariAdapter


class FeatureAtariEnv(gym.Env):
    """
    Thin Gym wrapper around ``AtariAdapter`` (RAM + abstract feature protocol).

    Observation: (84,) float32 — same vector Striatum expects with ``n_features=84``.
    """

    metadata = {"render_modes": []}

    def __init__(self, game_id: str = "ALE/Breakout-v5", render_mode: Optional[str] = None):
        super().__init__()
        self.game_id = game_id
        self._adapter = AtariAdapter(game_id=game_id, render_mode=render_mode)
        self.observation_space = gym.spaces.Box(
            low=-5.0, high=5.0, shape=(ABSTRACT_VEC_SIZE,), dtype=np.float32,
        )
        self.action_space = self._adapter.env.action_space
        self.num_agents = 1
        self._last_action = 0

    def _features(self, action: int) -> np.ndarray:
        af = self._adapter.get_abstract_features(action=action)
        return af.to_vector().astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._last_action = 0
        self._adapter.reset(seed=seed)
        return self._features(action=0), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        act = int(action)
        _, reward, done, info = self._adapter.step(act)
        self._last_action = act
        return self._features(action=act), float(reward), bool(done), False, info

    def close(self) -> None:
        self._adapter.env.close()


def make_feature_atari_env(
    game_id: str = "ALE/Breakout-v5",
    buf=None,
    seed: int = 0,
    **kwargs,
):
    """Factory for ``pufferlib.vector.make`` (wraps with GymnasiumPufferEnv)."""
    import pufferlib.emulation

    env = FeatureAtariEnv(game_id=game_id, **kwargs)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)
