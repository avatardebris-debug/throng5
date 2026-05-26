"""
nes_adapter.py — Backward-compatible re-exports for NES / Mega Man 2.

Implementation lives in brain.environments.nes.*
"""

from brain.environments.nes import (
    DEFAULT_ROM,
    DEFAULT_STATE,
    N_ACTIONS,
    MegaManEnv,
    NESAdapter,
    action_to_bitmask,
)

# Legacy alias used by ROM factory
_ACTION_MAP = {i: action_to_bitmask(i) for i in range(N_ACTIONS)}


def create_megaman_env(**kwargs) -> MegaManEnv:
    """Create a ready-to-train MegaManEnv."""
    return MegaManEnv(**kwargs)


__all__ = [
    "N_ACTIONS",
    "DEFAULT_ROM",
    "DEFAULT_STATE",
    "MegaManEnv",
    "NESAdapter",
    "create_megaman_env",
    "action_to_bitmask",
]
