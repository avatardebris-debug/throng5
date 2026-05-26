"""NES ROM and saved-state paths."""

from __future__ import annotations

import os

THRONG5_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DEFAULT_ROM = os.path.join(THRONG5_ROOT, "roms", "nes", "Mega Man 2 (USA)_ines1.nes")
DEFAULT_STATE = os.path.join(THRONG5_ROOT, "roms", "nes", "megaman2_stage_start.pkl")
