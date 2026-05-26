"""NES controller bitmask mapping for discrete actions."""

from __future__ import annotations

from typing import Dict

# LSB-first: A, B, Select, Start, Up, Down, Left, Right
ACTION_MAP: Dict[int, int] = {
    0: 0x00,
    1: 0x80,
    2: 0x80 | 0x01,
    3: 0x80 | 0x02,
    4: 0x80 | 0x01 | 0x02,
    5: 0x40,
    6: 0x40 | 0x01,
    7: 0x01,
    8: 0x02,
    9: 0x20,
    10: 0x10,
    11: 0x40 | 0x02,
}
N_ACTIONS = len(ACTION_MAP)


def action_to_bitmask(action: int) -> int:
    return ACTION_MAP.get(int(action), 0)
