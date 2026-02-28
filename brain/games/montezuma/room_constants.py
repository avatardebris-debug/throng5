"""
room_constants.py — Calibrated room coordinate constants for Room 0.

Source: montezuma_room_map.md (human-playthrough 2026-02-25)
Used by: _is_lethal(), ShapedRewardEngine, FearMemory, MCTSPlanner.

All values are in raw RAM coordinates (RAM[42]=x, RAM[43]=y).
Y increases downward (0=top of screen). Jump height ≈ 20 y units.
"""

import numpy as np

# ── Platform heights (y) ─────────────────────────────────────────────────

Y_TOP_PLATFORM    = 235   # Agent spawn platform
Y_CENTER_PLATFORM = 220   # Mid conveyor belt platform (approx)
Y_KEY_PLATFORM    = 205   # Left / key platform midpoint  (195–215)
Y_FLOOR           = 148   # Main floor (skull traversal level)
JUMP_HEIGHT       = 20    # Approximate jump height in y units

# ── Platform x ranges ────────────────────────────────────────────────────

X_LEFT_PLATFORM_MIN  = 9     # Left key platform
X_LEFT_PLATFORM_MAX  = 33
X_CENTER_SAFE_MIN    = 67    # Center platform safe x range
X_CENTER_SAFE_MAX    = 88
X_RIGHT_PLATFORM_MIN = 127   # Right platform
X_RIGHT_PLATFORM_MAX = 145

# ── Rope ─────────────────────────────────────────────────────────────────

X_ROPE             = 109     # Rope x position (approx)
X_ROPE_MIN         = 105
X_ROPE_MAX         = 113
Y_ROPE_MIN         = 198
Y_ROPE_MAX         = 212

# ── Ladder ───────────────────────────────────────────────────────────────

X_LADDER           = 20      # Left ladder column (approx ±2)
X_LEFT_LADDER_MIN  = 18
X_LEFT_LADDER_MAX  = 23

# ── Key location ─────────────────────────────────────────────────────────

X_KEY_MIN = 13
X_KEY_MAX = 17
X_KEY     = 15
Y_KEY     = 200

# ── Doors ────────────────────────────────────────────────────────────────

X_LEFT_DOOR  = 7     # x < X_LEFT_DOOR → exit room left (key required)
X_RIGHT_DOOR = 150   # x > X_RIGHT_DOOR → exit room right (approx, TBD)

# ── Skull zone ───────────────────────────────────────────────────────────

X_SKULL_MIN = 60
X_SKULL_MAX = 111
Y_SKULL     = 148    # floor level, patrol is horizontal

# ── Conveyor belt ────────────────────────────────────────────────────────

X_CONVEYOR_MIN = 65   # approx — pushes agent LEFT
X_CONVEYOR_MAX = 105
Y_CONVEYOR     = 220  # center platform level

# ── Void / lethal zones ──────────────────────────────────────────────────
# The mid-platform void is the gap between left platform (x=9-33) and
# the rope (x=109). There is no floor or rope between x=34 and x=104.
# Right of the right platform (x>145) at platform height is also void.

Y_VOID_MIN = 192    # top of void zone (above key platform level)
Y_VOID_MAX = 220    # bottom of void zone (above floor)
X_VOID_MIN = 34     # right edge of left platform
X_VOID_MAX = 104    # left edge of rope grab zone

# ── ALE action IDs (18-action set for MontezumaRevenge-v5) ───────────────
# 0=NOOP  1=FIRE  2=UP    3=RIGHT  4=LEFT  5=DOWN
# 6=UPRIGHT  7=UPLEFT  8=DOWNRIGHT  9=DOWNLEFT
# 10=UPFIRE  11=RIGHTFIRE  12=LEFTFIRE  13=DOWNFIRE
# 14=UPRIGHTFIRE  15=UPLEFTFIRE  16=DOWNRIGHTFIRE  17=DOWNLEFTFIRE

_RIGHTWARD = {3, 6, 8, 11, 14, 16}
_LEFTWARD  = {4, 7, 9, 12, 15, 17}
_UPWARD    = {2, 6, 7, 10, 14, 15}
_DOWNWARD  = {5, 8, 9, 13, 16, 17}

# ── Fall detection ────────────────────────────────────────────────────────
# Jump height ≈ 20 y units (calibrated 2026-02-25).
# A drop of > FALL_THRESHOLD in one step means the agent is falling,
# not jumping — a jump would have a peak and then come back down gradually.
#
# Safe fall exception: approaching the rope from above while falling.
#   - Rope at x≈109, y=198-212
#   - If abs(x - X_ROPE) <= ROPE_CATCH_RADIUS the agent may be mid-jump to rope
#   - Everywhere else a fall of this magnitude = certain death

FALL_THRESHOLD    = 20    # y decrease (y gets smaller going down in our coords)
ROPE_CATCH_RADIUS = 20    # horizontal distance from rope where fall is acceptable


def is_falling(prev_y: int, y: int) -> bool:
    """
    Return True if the agent dropped more than FALL_THRESHOLD in one step.
    (In Montezuma RAM coordinates, y DECREASES going down on screen.)
    A legitimate jump arc never drops more than ~20px in a single frame.
    """
    return (prev_y - y) > FALL_THRESHOLD


def is_safe_fall(x: int, y: int, prev_y: int) -> bool:
    """
    Return True if this fall could plausibly land on a safe platform.

    Safe fall zones (bi-directional around rope at x=109):
      - Approaching rope from top/center: x=89-130 while falling → rope grab attempt
      - Jumping LEFT off rope to center/conveyor: x=67-109 while falling
      - Jumping RIGHT off rope to right platform: x=109-145 while falling

    Combined: any x in 67-145 while falling = potentially a legitimate
    platform-to-platform jump (conveyor ↔ rope ↔ right platform).

    Lethal fall cases:
      - x < 67 while falling: too far left, lands off ledge or in void
      - x > 145 while falling: off the right edge, no platform
      - x=89-130 falling straight down onto floor below rope: skull patrol at
        x=60-111, y=148 kills the agent → caught by is_lethal_zone on landing
    """
    if not is_falling(prev_y, y):
        return True   # not falling at all — always safe
    # Any x inside the conveyor-rope-right_platform band is a valid arc
    return X_CENTER_SAFE_MIN <= x <= X_RIGHT_PLATFORM_MAX   # 67 ≤ x ≤ 145



def is_lethal_zone(x: int, y: int) -> bool:
    """
    Return True if (x, y) is a known void / death zone.

    Void zone: between left platform and rope at mid-platform height.
    Right void: to the right of the right platform at the same height.
    Excludes: rope grab area (x=105-113), left platform (x=9-33),
              left ladder column (x=18-23).

    Calibrated from human playthrough 2026-02-25.
    """
    if not (Y_VOID_MIN < y < Y_VOID_MAX):
        return False  # Wrong height — safe (floor, top platform, etc.)

    # Main void between left platform and rope
    if X_VOID_MIN <= x <= X_VOID_MAX:
        # But the left ladder is always safe to be on
        if X_LEFT_LADDER_MIN <= x <= X_LEFT_LADDER_MAX:
            return False
        return True

    # Right void: past the right platform
    if x > X_RIGHT_PLATFORM_MAX:
        return True

    return False


def action_mask_for_position(x: int, y: int, n_actions: int = 18) -> np.ndarray:
    """
    Return a boolean action mask for the current real position.

    mask[a] = True  → action a is allowed
    mask[a] = False → action a is BANNED

    Rules (calibrated 2026-02-25):
      Rule 1: Already in the void zone → ban rightward + upward (can't escape)
      Rule 2: On left platform right edge (x=30-33, y=192-220) → ban rightward
              (stepping further right = entering the void)
      Rule 3: On right platform right edge (x=143-145, y=192-220) → ban rightward

    The top platform (y=235) is entirely safe — no bans applied there.
    """
    mask = np.ones(n_actions, dtype=bool)

    # Rule 1: already in void — can't go right or up
    if is_lethal_zone(x, y):
        for a in (_RIGHTWARD | _UPWARD):
            if a < n_actions:
                mask[a] = False

    # Rule 2: standing at right edge of left platform, about to step into void
    if Y_VOID_MIN < y < Y_VOID_MAX and 29 <= x <= 34:
        if not (X_LEFT_LADDER_MIN <= x <= X_LEFT_LADDER_MAX):
            for a in _RIGHTWARD:
                if a < n_actions:
                    mask[a] = False

    # Rule 3: right edge of right platform
    if Y_VOID_MIN < y < Y_VOID_MAX and 142 <= x <= 147:
        for a in _RIGHTWARD:
            if a < n_actions:
                mask[a] = False

    return mask


# ── Conveyor belt traversal: actions to boost ─────────────────────────────
# On the conveyor belt the agent must jump right continuously.
# MCTS prior is boosted for these actions BEFORE simulations run,
# so the search explores jump-right paths even when Q-values haven't learned them.

# Action indices to boost in each zone:
_BOOST_CONVEYOR   = {6, 14}   # UPRIGHT, UPRIGHTFIRE — jump right
_BOOST_KEY_LEVEL  = {3, 6}    # RIGHT, UPRIGHT — rightward at key platform height


def action_preference_for_position(
    x: int, y: int, n_actions: int = 18
) -> "np.ndarray":
    """
    Return an additive prior-boost vector for MCTS root expansion.

    This is ADDED to the normalized prior before Dirichlet noise, then
    re-normalized. Higher values strongly bias MCTS toward those actions.

    Zones:
      Conveyor belt (x=65-105, y=210-240):
        Boost UPRIGHT(6) and UPRIGHTFIRE(14) by +3.0
        Agent must jump-right to make progress against leftward push.

      Key-platform level (y=192-202, x=33-115, not in void):
        Boost RIGHT(3) and UPRIGHT(6) by +1.5
        Agent needs to walk/jump rightward toward rope at x=109.

    Returns zero vector everywhere else (no bias — MCTS gets pure Q-prior).
    """
    boost = np.zeros(n_actions, dtype=np.float32)

    # Conveyor belt zone: strong jump-right bias
    on_conveyor = (X_CONVEYOR_MIN <= x <= X_CONVEYOR_MAX
                   and Y_CONVEYOR - 20 <= y <= Y_TOP_PLATFORM + 5)
    if on_conveyor:
        for a in _BOOST_CONVEYOR:
            if a < n_actions:
                boost[a] = 3.0

    # Key-platform level: moderate rightward bias when NOT in void
    elif (192 <= y <= 202 and x > X_LEFT_PLATFORM_MAX
          and not is_lethal_zone(x, y)):
        for a in _BOOST_KEY_LEVEL:
            if a < n_actions:
                boost[a] = 1.5

    return boost


def platform_name(x: int, y: int) -> str:
    """Human-readable zone name for current (x, y). Used in the HUD."""
    if y >= 230:
        return "top_platform"
    if is_lethal_zone(x, y):
        return "VOID/LETHAL"
    if Y_ROPE_MIN <= y <= Y_ROPE_MAX and X_ROPE_MIN <= x <= X_ROPE_MAX:
        return "rope"
    if Y_VOID_MIN < y < Y_VOID_MAX and X_LEFT_PLATFORM_MIN <= x <= X_LEFT_PLATFORM_MAX:
        return "key_platform"
    if X_LEFT_LADDER_MIN <= x <= X_LEFT_LADDER_MAX:
        return "left_ladder"
    if Y_VOID_MIN < y < Y_VOID_MAX and X_RIGHT_PLATFORM_MIN <= x <= X_RIGHT_PLATFORM_MAX:
        return "right_platform"
    if y <= Y_FLOOR + 10:
        if X_SKULL_MIN <= x <= X_SKULL_MAX:
            return "floor(skull_zone)"
        return "floor"
    if Y_CONVEYOR - 15 <= y <= Y_CONVEYOR + 10:
        return "center_platform"
    return f"mid({x},{y})"
