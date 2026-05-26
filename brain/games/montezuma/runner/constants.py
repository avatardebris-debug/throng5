"""Montezuma constants and RAM layout."""

from __future__ import annotations

from pathlib import Path


# ── Constants ────────────────────────────────────────────────────────

GAME_ID = "ALE/MontezumaRevenge-v5"
N_ACTIONS = 18
RAM_SIZE = 128
RESULTS_DIR = Path("experiments/montezuma_brain")
RECORDINGS_DIR = RESULTS_DIR / "recordings"


def ensure_results_dirs() -> None:
    """Create experiment dirs on first use (not at import)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ── Montezuma RAM Addresses ─────────────────────────────────────────

RAM_PLAYER_X    = 42
RAM_PLAYER_Y    = 43
RAM_ROOM        = 3
RAM_LIVES       = 58
RAM_SCORE_1     = 19
RAM_SCORE_2     = 20
RAM_SKULL_X     = 47
RAM_SKULL_Y     = 46
RAM_ITEMS       = 65

# Atari action names for display
ACTION_NAMES = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]

# Keyboard mapping for human play (pygame key → atari action)
# Standard: arrow keys + space(fire), diagonals with shift
KEYBOARD_MAP = {
    "noop":       0,
    "fire":       1,
    "up":         2,
    "right":      3,
    "left":       4,
    "down":       5,
    "upright":    6,
    "upleft":     7,
    "downright":  8,
    "downleft":   9,
    "upfire":    10,
    "rightfire": 11,
    "leftfire":  12,
    "downfire":  13,
}

