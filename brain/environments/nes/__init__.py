"""NES environment support (nes-py + feature adapter)."""

from brain.environments.nes.action_map import N_ACTIONS, action_to_bitmask
from brain.environments.nes.feature_adapter import NESAdapter
from brain.environments.nes.megaman_env import MegaManEnv
from brain.environments.nes.paths import DEFAULT_ROM, DEFAULT_STATE

__all__ = [
    "N_ACTIONS",
    "DEFAULT_ROM",
    "DEFAULT_STATE",
    "MegaManEnv",
    "NESAdapter",
    "action_to_bitmask",
]
