"""
nes_adapter.py — NES/SNES adapter via gym-retro.

Wraps retro environments (NES, SNES, Genesis) into the brain's
abstract feature interface alongside the existing Atari adapter.

Designed to work via the ROMAdapterFactory, which auto-detects
the platform and creates the correct adapter.

Note: Requires `gym-retro` (pip install gym-retro).
This is an optional dependency — the brain works without it.

Usage:
    from brain.environments.nes_adapter import NESAdapter

    adapter = NESAdapter("SuperMarioBros-Nes")
    adapter.observe(obs)
    features = adapter.make_features(action=2)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class NESAdapter:
    """
    Environment adapter for NES/SNES/Genesis ROMs via gym-retro.

    Handles:
    - Button mapping (NES has different controls than Atari)
    - Pixel observation preprocessing (224×240 → features)
    - RAM state extraction (when available)
    - ROM-specific info parsing (score, lives, stage)
    """

    def __init__(
        self,
        game: str = "SuperMarioBros-Nes",
        target_dim: int = 84,
        use_ram: bool = False,
    ):
        self.game = game
        self.target_dim = target_dim
        self.use_ram = use_ram

        self._obs_buffer: Optional[np.ndarray] = None
        self._info: Dict[str, Any] = {}
        self._step_count = 0

        # Random projection for pixel observations
        # NES pixels are typically 224×240×3 = 161280
        rng = np.random.RandomState(42)
        pixel_flat = 224 * 240 * 3
        self._projection = rng.randn(pixel_flat, target_dim).astype(np.float32)
        self._projection /= np.sqrt(pixel_flat)

        # Running normalization stats
        self._mean = np.zeros(target_dim, dtype=np.float32)
        self._var = np.ones(target_dim, dtype=np.float32)
        self._n_obs = 0

        # NES button mapping
        # gym-retro uses: B, null, SELECT, START, UP, DOWN, LEFT, RIGHT, A
        self._button_names = [
            "B", "null", "SELECT", "START",
            "UP", "DOWN", "LEFT", "RIGHT", "A",
        ]
        self.n_buttons = 9  # NES has 9 buttons

    def observe(self, obs: Any, info: Optional[Dict] = None) -> None:
        """Store the latest observation and info dict."""
        self._obs_buffer = obs
        self._info = info or {}
        self._step_count += 1

    def make_features(self, action: int = 0) -> np.ndarray:
        """Convert current NES observation into abstract features."""
        if self._obs_buffer is None:
            return np.zeros(self.target_dim, dtype=np.float32)

        raw = np.asarray(self._obs_buffer, dtype=np.float32).flatten()

        # Project pixels down to target_dim
        if len(raw) > self._projection.shape[0]:
            raw = raw[:self._projection.shape[0]]
        elif len(raw) < self._projection.shape[0]:
            raw = np.pad(raw, (0, self._projection.shape[0] - len(raw)))

        features = raw @ self._projection

        # Normalize
        self._n_obs += 1
        alpha = max(0.01, 1.0 / self._n_obs)
        self._mean = (1 - alpha) * self._mean + alpha * features
        self._var = (1 - alpha) * self._var + alpha * (features - self._mean) ** 2
        features = (features - self._mean) / (np.sqrt(self._var) + 1e-8)

        return features.astype(np.float32)

    def action_to_buttons(self, action: int) -> np.ndarray:
        """
        Convert a discrete action index to NES button presses.

        Maps a single action integer to the 9-button binary array
        that gym-retro expects. Uses a simple mapping:
            0: NOOP, 1: RIGHT, 2: RIGHT+A, 3: RIGHT+B,
            4: LEFT, 5: DOWN, 6: UP, 7: A, 8: B,
            9: RIGHT+A+B (run+jump), ...
        """
        buttons = np.zeros(self.n_buttons, dtype=np.int8)

        mappings = {
            0: [],                        # NOOP
            1: [7],                       # RIGHT
            2: [7, 8],                    # RIGHT + A (run right)
            3: [7, 0],                    # RIGHT + B
            4: [6],                       # LEFT
            5: [6, 8],                    # LEFT + A (run left)
            6: [5],                       # DOWN
            7: [4],                       # UP
            8: [8],                       # A (jump)
            9: [0],                       # B (action)
            10: [7, 8, 0],                # RIGHT + A + B (run + jump)
            11: [6, 8, 0],                # LEFT + A + B (run + jump left)
        }

        if action in mappings:
            for btn in mappings[action]:
                buttons[btn] = 1
        elif action < self.n_buttons:
            buttons[action] = 1

        return buttons

    def get_game_info(self) -> Dict[str, Any]:
        """Extract game state from info dict (ROM-specific)."""
        return {
            "lives": self._info.get("lives", 0),
            "score": self._info.get("score", 0),
            "stage": self._info.get("level", self._info.get("stage", 0)),
            "x_position": self._info.get("x", self._info.get("xscrollLo", 0)),
            "raw_info": self._info,
        }

    @staticmethod
    def create_env(game: str, state: Optional[str] = None):
        """
        Create a gym-retro environment.

        Returns the env or None if retro is not installed.
        """
        try:
            import retro
            kwargs = {"game": game}
            if state:
                kwargs["state"] = state
            return retro.make(**kwargs)
        except ImportError:
            return None
        except Exception:
            return None

    def stats(self) -> Dict[str, Any]:
        return {
            "game": self.game,
            "steps": self._step_count,
            "n_buttons": self.n_buttons,
        }
