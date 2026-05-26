"""Random-projection feature extractor for NES pixel frames."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from brain.environments.nes.action_map import N_ACTIONS, action_to_bitmask
from brain.environments.nes.paths import DEFAULT_ROM, DEFAULT_STATE


class NESAdapter:
    """
    Converts raw NES pixel frame (240×256×3) → (n_features,) float32 vector.
  Also supports the legacy observe / make_features brain pipeline API.
    """

    def __init__(self, n_features: int = 84):
        self.n_features = n_features
        self.last_features = np.zeros(n_features, dtype=np.float32)

        rng = np.random.RandomState(42)
        pixel_flat = 240 * 256 * 3
        self._proj = rng.randn(pixel_flat, n_features).astype(np.float32)
        self._proj /= np.sqrt(pixel_flat)

        self._mean = np.zeros(n_features, dtype=np.float32)
        self._var = np.ones(n_features, dtype=np.float32)
        self._n = 0
        self._pending_obs: Any = None

    def observe_and_extract(self, obs: np.ndarray) -> np.ndarray:
        raw = np.asarray(obs, dtype=np.float32).flatten()
        p = self._proj.shape[0]
        if len(raw) < p:
            raw = np.pad(raw, (0, p - len(raw)))
        else:
            raw = raw[:p]

        feat = raw @ self._proj
        self._n += 1
        alpha = max(0.01, 1.0 / self._n)
        self._mean += alpha * (feat - self._mean)
        self._var = (1 - alpha) * self._var + alpha * (feat - self._mean) ** 2
        feat = (feat - self._mean) / (np.sqrt(self._var) + 1e-8)

        self.last_features = feat.astype(np.float32)
        return self.last_features

    def observe(self, obs: Any, info: Optional[Dict] = None) -> None:
        self._pending_obs = obs

    def make_features(self, action: int = 0) -> np.ndarray:
        if self._pending_obs is None:
            return self.last_features
        return self.observe_and_extract(self._pending_obs)

    def action_to_buttons(self, action: int) -> int:
        return action_to_bitmask(action)

    def get_game_info(self) -> Dict[str, Any]:
        return {}

    def stats(self) -> Dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_actions": N_ACTIONS,
            "n_obs": self._n,
        }

    @staticmethod
    def create_env(
        rom_path: str = DEFAULT_ROM,
        state_path: str = DEFAULT_STATE,
        render: bool = False,
        n_features: int = 84,
    ):
        """Factory for ``MegaManEnv`` (ROM + feature adapter wiring)."""
        from brain.environments.nes.megaman_env import MegaManEnv

        return MegaManEnv(
            rom_path=rom_path,
            state_path=state_path,
            n_features=n_features,
            render=render,
        )
