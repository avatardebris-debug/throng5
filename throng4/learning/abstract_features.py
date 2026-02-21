"""
Abstract Feature Layer — Two-layer portable state representation.

Every environment adapter maps its raw state into:
  - core:     fixed CORE_SIZE vector (universal concepts, always populated)
  - ext:      up to EXT_MAX adapter-specific extra values (zero-padded)
  - ext_mask: binary gate, 1.0 where ext slot is active

The full NN input vector is:
  phi = [core | ext * ext_mask | ext_mask | one_hot_action]

The mask travels with the data so the network learns which ext slots
to attend to vs. ignore — no architectural change needed.

Core schema (CORE_SIZE = 20):
  [0]  agent_x           normalized 0–1
  [1]  agent_y           normalized 0–1
  [2]  target_x          primary reward source / goal x
  [3]  target_y          primary reward source / goal y
  [4]  threat_x          nearest danger x
  [5]  threat_y          nearest danger y
  [6]  threat_proximity  0=adjacent danger, 1=far
  [7]  reward_proximity  0=adjacent reward, 1=far
  [8]  agent_vx          signed velocity x, normalized –1–1
  [9]  agent_vy          signed velocity y, normalized –1–1
  [10] target_vx         moving target velocity x
  [11] target_vy         moving target velocity y
  [12] resource_level    lives/health/ammo, 0–1
  [13] density_field     obstacle/threat density scalar 0–1
  [14] episode_progress  steps / max_steps, 0–1
  [15] context_0         adapter-defined general context slot
  [16] context_1
  [17] context_2
  [18] context_3
  [19] context_4
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


# ------------------------------------------------------------------
# Dimensions
# ------------------------------------------------------------------

CORE_SIZE = 20   # Universal slots — never changes across adapters
EXT_MAX   = 32   # Max extension slots — padded/masked when unused

# Total flat vector size (excluding one-hot action):
#   core + gated_ext + mask  ← DEFAULT CHOICE (84 dims)
ABSTRACT_VEC_SIZE = CORE_SIZE + EXT_MAX + EXT_MAX  # = 84

# Ablation baseline: core + gated_ext only, no explicit mask (52 dims)
# Use for transfer speed comparison; default to 84-dim with mask.
ABSTRACT_VEC_SIZE_NO_MASK = CORE_SIZE + EXT_MAX  # = 52

# Ext noise: add this to AgentConfig.ext_noise_std (default 0.02) to
# apply Gaussian noise to ext block during training so the model doesn't
# over-rely on any one adapter-specific slot.  Disabled at eval time.
EXT_NOISE_DEFAULT = 0.02

# Core slot indices (use these as named constants in adapters)
IDX_AGENT_X        = 0
IDX_AGENT_Y        = 1
IDX_TARGET_X       = 2
IDX_TARGET_Y       = 3
IDX_THREAT_X       = 4
IDX_THREAT_Y       = 5
IDX_THREAT_PROX    = 6
IDX_REWARD_PROX    = 7
IDX_AGENT_VX       = 8
IDX_AGENT_VY       = 9
IDX_TARGET_VX      = 10
IDX_TARGET_VY      = 11
IDX_RESOURCE       = 12
IDX_DENSITY        = 13
IDX_EPISODE_PROG   = 14
IDX_CONTEXT_0      = 15
IDX_CONTEXT_1      = 16
IDX_CONTEXT_2      = 17
IDX_CONTEXT_3      = 18
IDX_CONTEXT_4      = 19

# Human-readable names for blind logging
CORE_NAMES = [
    "agent_x", "agent_y", "target_x", "target_y",
    "threat_x", "threat_y", "threat_prox", "reward_prox",
    "agent_vx", "agent_vy", "target_vx", "target_vy",
    "resource", "density", "ep_progress",
    "ctx0", "ctx1", "ctx2", "ctx3", "ctx4",
]


# ------------------------------------------------------------------
# Container
# ------------------------------------------------------------------

@dataclass
class AbstractFeature:
    """
    Two-layer abstract feature for a single (state, action) pair.

    core     : shape (CORE_SIZE,)  — always fully populated
    ext      : shape (EXT_MAX,)    — zero in unused slots
    ext_mask : shape (EXT_MAX,)    — 1.0 where slot is active, 0.0 elsewhere
    """
    core:     np.ndarray
    ext:      np.ndarray = field(default_factory=lambda: np.zeros(EXT_MAX, dtype=np.float32))
    ext_mask: np.ndarray = field(default_factory=lambda: np.zeros(EXT_MAX, dtype=np.float32))

    def to_vector(self) -> np.ndarray:
        """
        Flat NN input: [core | gated_ext | mask]
        Shape: (ABSTRACT_VEC_SIZE,) = (84,)
        """
        gated = self.ext * self.ext_mask
        return np.concatenate([self.core, gated, self.ext_mask]).astype(np.float32)

    def with_action(self, one_hot_action: np.ndarray) -> np.ndarray:
        """
        Append a one-hot action vector for use as phi(s,a).
        Shape: (ABSTRACT_VEC_SIZE + len(one_hot_action),)
        """
        return np.concatenate([self.to_vector(), one_hot_action]).astype(np.float32)

    def blind_log_str(self, action_name: str = "?", reward: float = 0.0,
                      step: int = 0) -> str:
        """
        Blind semantic log line — uses abstract field names only.
        Safe to send to Tetra without revealing game identity.
        """
        c = self.core
        ext_active = int(self.ext_mask.sum())
        return (
            f"Step {step:03d} | act:{action_name:>6} | "
            f"agent:({c[IDX_AGENT_X]:.2f},{c[IDX_AGENT_Y]:.2f}) "
            f"target:({c[IDX_TARGET_X]:.2f},{c[IDX_TARGET_Y]:.2f}) "
            f"threat_prox:{c[IDX_THREAT_PROX]:.2f} "
            f"reward_prox:{c[IDX_REWARD_PROX]:.2f} "
            f"rsrc:{c[IDX_RESOURCE]:.2f} "
            f"density:{c[IDX_DENSITY]:.2f} | "
            f"reward:{reward} | ext_slots:{ext_active}"
        )


# ------------------------------------------------------------------
# Helpers for adapters
# ------------------------------------------------------------------

def make_ext(values: list, max_size: int = EXT_MAX
             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (ext, ext_mask) from a list of float values.
    Pads with zeros to max_size. Mask is 1.0 for populated slots.

    Usage in an adapter:
        ext, mask = make_ext([col_h0, col_h1, col_h2, bumpiness])
    """
    n = min(len(values), max_size)
    ext  = np.zeros(max_size, dtype=np.float32)
    mask = np.zeros(max_size, dtype=np.float32)
    ext[:n]  = np.array(values[:n], dtype=np.float32)
    mask[:n] = 1.0
    return ext, mask


def empty_core() -> np.ndarray:
    """Return a zeroed core vector. Adapters fill slots by index."""
    return np.zeros(CORE_SIZE, dtype=np.float32)
