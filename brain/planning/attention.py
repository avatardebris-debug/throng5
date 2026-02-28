"""
attention.py — RAM-guided attention for feature extraction.

Without attention, the CNN processes the entire 84x84 frame equally.
But ~95% of pixels are irrelevant background. This module uses
RAM entity positions to focus attention on relevant regions.

Produces an attention mask that tells the CNN where to look:
  - Player position → high attention
  - Enemy positions → high attention (threat)
  - Item positions → medium attention (goal)
  - Background → low attention

Usage:
    attention = RAMAttention(screen_w=160, screen_h=210, obs_w=84, obs_h=84)
    mask = attention.compute_mask(ram, mapper)
    weighted_obs = observation * mask
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RAMAttention:
    """
    Generates spatial attention masks from RAM entity positions.

    Converts RAM-discovered entity positions into a 2D attention map
    that can be applied to visual observations (pixel frames) to
    help the CNN focus on relevant game elements.
    """

    def __init__(
        self,
        screen_w: int = 160,
        screen_h: int = 210,
        obs_w: int = 84,
        obs_h: int = 84,
        base_attention: float = 0.2,
        entity_radius: int = 12,
    ):
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._obs_w = obs_w
        self._obs_h = obs_h
        self._base = base_attention
        self._radius = entity_radius

        # Entity attention weights by category
        self._weights = {
            "player": 1.0,
            "enemy": 0.9,
            "item": 0.8,
            "goal": 1.0,
            "obstacle": 0.5,
            "entity": 0.6,
        }

        # Precompute Gaussian kernel
        self._kernel = self._make_gaussian(entity_radius)

    def _make_gaussian(self, radius: int) -> np.ndarray:
        """Create a 2D Gaussian kernel for attention spread."""
        size = radius * 2 + 1
        x = np.arange(size) - radius
        kernel_1d = np.exp(-(x ** 2) / (2 * (radius / 2) ** 2))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / kernel_2d.max()

    def compute_mask(
        self,
        entities: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Compute attention mask from entity list.

        entities: list of {"x": int, "y": int, "category": str}

        Returns: (obs_h, obs_w) float32 array in [base, 1.0]
        """
        mask = np.full((self._obs_h, self._obs_w), self._base, dtype=np.float32)

        for entity in entities:
            x = entity.get("x")
            y = entity.get("y")
            if x is None or y is None:
                continue

            category = entity.get("category", "entity")
            weight = self._weights.get(category, 0.5)

            # Convert screen coordinates to observation coordinates
            ox = int(x * self._obs_w / self._screen_w)
            oy = int(y * self._obs_h / self._screen_h)

            # Apply Gaussian attention at this position
            r = self._radius
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    py = oy + dy
                    px = ox + dx
                    if 0 <= py < self._obs_h and 0 <= px < self._obs_w:
                        ki = dy + r
                        kj = dx + r
                        if ki < self._kernel.shape[0] and kj < self._kernel.shape[1]:
                            val = self._kernel[ki, kj] * weight
                            mask[py, px] = max(mask[py, px], val)

        return mask

    def compute_mask_from_ram(
        self,
        ram: np.ndarray,
        mapper=None,
    ) -> np.ndarray:
        """
        Compute attention mask directly from RAM + semantic mapper.

        Uses the mapper's entity groups to find positions.
        """
        if mapper is None:
            return np.ones((self._obs_h, self._obs_w), dtype=np.float32)

        ram = np.asarray(ram, dtype=np.uint8).flatten()
        entities = []

        for group in mapper.get_entity_groups():
            if len(group["bytes"]) >= 2:
                x_addr = group["bytes"][0]
                y_addr = group["bytes"][1]
                if x_addr < len(ram) and y_addr < len(ram):
                    entities.append({
                        "x": int(ram[x_addr]),
                        "y": int(ram[y_addr]),
                        "category": "player" if "player" in group["type"] else "entity",
                    })

        # Also add attention to subgoal-relevant positions
        for item in mapper.get_subgoal_bytes():
            addr = item["addr"]
            if addr < len(ram) and ram[addr] > 0:
                # State flag is active — draw attention to it
                # (We don't know its position, so add a diffuse beacon)
                pass

        return self.compute_mask(entities)

    def apply(
        self, observation: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        """Apply attention mask to an observation."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 3:
            # Multi-channel (H, W, C) — broadcast mask
            return obs * mask[:, :, np.newaxis]
        return obs * mask

    def report(self) -> Dict[str, Any]:
        return {
            "obs_size": (self._obs_h, self._obs_w),
            "base_attention": self._base,
            "entity_radius": self._radius,
            "categories": list(self._weights.keys()),
        }
