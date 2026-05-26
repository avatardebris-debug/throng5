"""Canonical Montezuma semantic game state from ALE RAM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, TypedDict

import numpy as np

from brain.games.montezuma.room_constants import X_KEY, Y_KEY, Y_FLOOR
from brain.games.montezuma.runner.constants import (
    RAM_ITEMS,
    RAM_LIVES,
    RAM_PLAYER_X,
    RAM_PLAYER_Y,
    RAM_ROOM,
    RAM_SCORE_1,
    RAM_SCORE_2,
    RAM_SKULL_X,
    RAM_SKULL_Y,
)


class ItemInfo(TypedDict):
    x: int
    y: int
    collected: bool


class EnemyInfo(TypedDict):
    x: int
    y: int


class GameStateDict(TypedDict):
    """Dict contract shared by runner, intentional controller, and skill library."""

    player_x: int
    player_y: int
    room: int
    lives: int
    score: int
    skull_x: int
    skull_y: int
    items_mask: int
    items: Dict[str, ItemInfo]
    enemies: Dict[str, EnemyInfo]
    has_key: bool


# Bit flags in RAM[65] and known screen positions for skill preconditions.
_ITEM_SPECS: Dict[str, Dict[str, Any]] = {
    "key": {"bit": 0x01, "x": X_KEY, "y": Y_KEY},
    "door": {"bit": 0x02, "x": 133, "y": Y_FLOOR},
}


@dataclass
class GameState:
    player_x: int
    player_y: int
    room: int
    lives: int
    score: int
    skull_x: int
    skull_y: int
    items_mask: int
    items: Dict[str, ItemInfo] = field(default_factory=dict)
    enemies: Dict[str, EnemyInfo] = field(default_factory=dict)
    has_key: bool = False

    def to_dict(self) -> GameStateDict:
        return {
            "player_x": self.player_x,
            "player_y": self.player_y,
            "room": self.room,
            "lives": self.lives,
            "score": self.score,
            "skull_x": self.skull_x,
            "skull_y": self.skull_y,
            "items_mask": self.items_mask,
            "items": dict(self.items),
            "enemies": dict(self.enemies),
            "has_key": self.has_key,
        }


def _build_items(items_mask: int) -> Dict[str, ItemInfo]:
    items: Dict[str, ItemInfo] = {}
    for name, spec in _ITEM_SPECS.items():
        bit = int(spec["bit"])
        items[name] = {
            "x": int(spec["x"]),
            "y": int(spec["y"]),
            "collected": bool(items_mask & bit),
        }
    return items


def game_state_from_ram(ram: np.ndarray) -> GameState:
    """Build canonical GameState from ALE RAM."""
    items_mask = int(ram[RAM_ITEMS])
    skull_x = int(ram[RAM_SKULL_X])
    skull_y = int(ram[RAM_SKULL_Y])
    return GameState(
        player_x=int(ram[RAM_PLAYER_X]),
        player_y=int(ram[RAM_PLAYER_Y]),
        room=int(ram[RAM_ROOM]),
        lives=int(ram[RAM_LIVES]),
        score=int(ram[RAM_SCORE_1]) * 100 + int(ram[RAM_SCORE_2]),
        skull_x=skull_x,
        skull_y=skull_y,
        items_mask=items_mask,
        items=_build_items(items_mask),
        enemies={"skull": {"x": skull_x, "y": skull_y}},
        has_key=bool(items_mask & 0x01),
    )


def get_game_state(ram: np.ndarray) -> GameStateDict:
    """Return game state dict (adapter entry point for runner modes)."""
    return game_state_from_ram(ram).to_dict()
