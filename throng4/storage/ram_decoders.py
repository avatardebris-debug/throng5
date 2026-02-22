"""
throng4/storage/ram_decoders.py
================================
Semantic RAM decoders for ALE games.

Each decoder extracts named, interpretable features from the 128-byte
ALE RAM observation, enabling Tetra to reason about game state rather
than anonymous byte values.

RAM addresses are documented from:
  - Go-Explore paper (Ecoffet et al. 2019): x/y/room/lives for Montezuma
  - ALE community RAM maps: Breakout, Pong, SpaceInvaders

Usage
-----
    decoder = get_decoder("ALE/MontezumaRevenge-v5")
    if decoder:
        state = decoder.decode(ram_obs)
        # state = {"player_x": 42, "player_y": 118, "room": 1,
        #          "lives": 4, "key_collected": False, "score": 0,
        #          "player_x_norm": 0.26, ...}

Each field comes in raw + _norm (0.0–1.0) form.
Unknown/unverified bytes are flagged with "_confidence": "estimated".
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────

class BaseDecoder:
    """
    Minimal base: subclasses implement `decode(ram)`.
    Raw RAM is uint8 numpy array of length 128.
    """
    game_id: str = ""
    confidence: str = "high"   # "high" | "estimated"

    def decode(self, ram: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def feature_names(self) -> list[str]:
        """Return list of named features this decoder produces."""
        return list(self.decode(np.zeros(128, dtype=np.uint8)).keys())


# ─────────────────────────────────────────────────────────────────────
# Montezuma's Revenge
# ─────────────────────────────────────────────────────────────────────

class MontezumaDecoder(BaseDecoder):
    """
    RAM decoder for ALE/MontezumaRevenge-v5.

    Addresses sourced from:
      - Ecoffet et al. (2019) "Go-Explore: a New Approach for Hard
        Exploration Problems", github.com/uber-research/go-explore
      - ALE community annotations

    Verified addresses (confidence=high):
        42  player_x       (0-159, pixel column)
        43  player_y       (0-255, pixel row; valid range ~8-228)
         3  room           (0-23 room index, 24 rooms in game)
        58  lives          (0-5, starts at 5)

    Estimated addresses (confidence=estimated — may need calibration):
        66  items_carried  (bitmask: bit0=key, bit1=torch, bit2=sword)
        19  score_hi       (BCD high byte)
        20  score_lo       (BCD low byte)
    """

    game_id = "ALE/MontezumaRevenge-v5"
    confidence = "mixed"

    # Documented rooms (room_id → description for Tetra context)
    ROOM_NAMES = {
        0: "start_room",
        1: "first_passage",
        2: "skull_room",
        3: "key_room",
        4: "lower_passage",
        5: "rope_room",
        6: "treasure_room",
    }

    def decode(self, ram: np.ndarray) -> dict[str, Any]:
        ram = np.asarray(ram, dtype=np.uint8)

        player_x   = int(ram[42])          # 0-159
        player_y   = int(ram[43])          # 0-255 (valid ~8-228)
        room       = int(ram[3])           # 0-23
        lives      = int(ram[58])          # 0-5
        items      = int(ram[66])          # bitmask (estimated)
        key_bit    = bool(items & 0x01)    # bit 0 = key carried

        # BCD score decode (estimated addresses)
        score_hi = int(ram[19])
        score_lo = int(ram[20])
        score = (((score_hi >> 4) & 0xF) * 10000 +
                  (score_hi & 0xF)        * 1000 +
                  ((score_lo >> 4) & 0xF) * 100 +
                  (score_lo & 0xF)        * 10)

        room_name = self.ROOM_NAMES.get(room, f"room_{room}")

        return {
            # Raw game state
            "player_x":       player_x,
            "player_y":       player_y,
            "room":           room,
            "room_name":      room_name,
            "lives":          lives,
            "key_collected":  key_bit,
            "items_raw":      items,
            "score":          score,
            # Normalised 0–1 (for feature engineering)
            "player_x_norm":  round(player_x / 159.0,  4),
            "player_y_norm":  round(player_y / 228.0,  4),
            "room_norm":      round(room      / 23.0,   4),
            "lives_norm":     round(lives     / 5.0,    4),
            # Derived flags (higher-level — what Tetra can reason about)
            "in_start_room":  room == 0,
            "has_key":        key_bit,
            "low_lives":      lives <= 1,
            "_decoder":       "MontezumaDecoder",
            "_confidence":    {
                "player_x":  "high",
                "player_y":  "high",
                "room":      "high",
                "lives":     "high",
                "key":       "estimated",
                "score":     "estimated",
            },
        }


# ─────────────────────────────────────────────────────────────────────
# Breakout
# ─────────────────────────────────────────────────────────────────────

class BreakoutDecoder(BaseDecoder):
    """
    RAM addresses from ALE community (widely verified).
        72  paddle_x     (0-159)
        99  ball_x       (0-159)
       101  ball_y       (0-210)
        57  lives        (0-5)
       119  bricks_left  (roughly, not exact)
    """
    game_id = "ALE/Breakout-v5"
    confidence = "high"

    def decode(self, ram: np.ndarray) -> dict[str, Any]:
        ram = np.asarray(ram, dtype=np.uint8)
        paddle_x = int(ram[72])
        ball_x   = int(ram[99])
        ball_y   = int(ram[101])
        lives    = int(ram[57])

        return {
            "paddle_x":       paddle_x,
            "ball_x":         ball_x,
            "ball_y":         ball_y,
            "lives":          lives,
            "dx":             ball_x - paddle_x,        # relative: <0 ball left of paddle
            "paddle_x_norm":  round(paddle_x / 159.0, 4),
            "ball_x_norm":    round(ball_x   / 159.0, 4),
            "ball_y_norm":    round(ball_y   / 210.0, 4),
            "lives_norm":     round(lives    / 5.0,   4),
            "ball_above_paddle": ball_y < 180,
            "_decoder":       "BreakoutDecoder",
            "_confidence":    "high",
        }


# ─────────────────────────────────────────────────────────────────────
# Pong
# ─────────────────────────────────────────────────────────────────────

class PongDecoder(BaseDecoder):
    """
    RAM addresses from ALE community.
        51  player_y  (right paddle, 0-255)
        50  opponent_y (left paddle, 0-255)
        49  ball_x   (0-159)
        54  ball_y   (0-255)
        13  player_score
        14  opponent_score
    """
    game_id = "ALE/Pong-v5"
    confidence = "high"

    def decode(self, ram: np.ndarray) -> dict[str, Any]:
        ram = np.asarray(ram, dtype=np.uint8)
        player_y   = int(ram[51])
        opponent_y = int(ram[50])
        ball_x     = int(ram[49])
        ball_y     = int(ram[54])

        return {
            "player_y":        player_y,
            "opponent_y":      opponent_y,
            "ball_x":          ball_x,
            "ball_y":          ball_y,
            "player_dy":       player_y  - ball_y,   # positive = paddle above ball
            "opponent_dy":     opponent_y - ball_y,
            "player_y_norm":   round(player_y   / 255.0, 4),
            "opponent_y_norm": round(opponent_y / 255.0, 4),
            "ball_x_norm":     round(ball_x     / 159.0, 4),
            "ball_y_norm":     round(ball_y     / 255.0, 4),
            "ball_coming":     ball_x > 80,   # rough: ball on player's half
            "_decoder":        "PongDecoder",
            "_confidence":     "high",
        }


# ─────────────────────────────────────────────────────────────────────
# Space Invaders
# ─────────────────────────────────────────────────────────────────────

class SpaceInvadersDecoder(BaseDecoder):
    """
    RAM addresses from ALE community.
        28  player_x     (cannon, 0-159)
        73  lives        (0-3)
        17  score_hi
        18  score_lo
    """
    game_id = "ALE/SpaceInvaders-v5"
    confidence = "estimated"

    def decode(self, ram: np.ndarray) -> dict[str, Any]:
        ram = np.asarray(ram, dtype=np.uint8)
        player_x = int(ram[28])
        lives    = int(ram[73])
        score    = int(ram[17]) * 256 + int(ram[18])

        return {
            "player_x":      player_x,
            "lives":         lives,
            "score":         score,
            "player_x_norm": round(player_x / 159.0, 4),
            "lives_norm":    round(lives     / 3.0,   4),
            "low_lives":     lives <= 1,
            "_decoder":      "SpaceInvadersDecoder",
            "_confidence":   "estimated",
        }


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, BaseDecoder] = {
    d.game_id: d()
    for d in [
        MontezumaDecoder,
        BreakoutDecoder,
        PongDecoder,
        SpaceInvadersDecoder,
    ]
}


def get_decoder(game_id: str) -> BaseDecoder | None:
    """
    Return a RAM decoder for *game_id*, or None if not available.

    Example
    -------
    decoder = get_decoder("ALE/MontezumaRevenge-v5")
    if decoder:
        features = decoder.decode(ram_obs)
    """
    return _REGISTRY.get(game_id)


def list_decoders() -> list[str]:
    """Return all game IDs with registered decoders."""
    return sorted(_REGISTRY.keys())
