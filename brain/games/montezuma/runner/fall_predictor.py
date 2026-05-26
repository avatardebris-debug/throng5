"""Predictive fall avoidance for Montezuma platform geometry."""

from __future__ import annotations


class FallPredictor:
    """
    PREDICTIVE fall avoidance — filters dangerous actions BEFORE taking them.

    Instead of detecting falls mid-air (too late to recover), this:
    1. Learns which (position, action) combos led to falls
    2. Knows Montezuma platform geometry (hardcoded boot knowledge)
    3. Before action selection, removes actions likely to cause falls

    Rope zone (x=67-145) is exempt — y-drops there are expected.
    """

    FALL_THRESHOLD = 20  # y-drop bigger than a jump = fall
    ROPE_X_MIN = 67
    ROPE_X_MAX = 145
    BIN_SIZE = 8  # Spatial binning for position lookup
    LEARN_THRESHOLD = 2  # Falls before banning an action at a position

    # Atari Montezuma action mapping
    # 0=NOOP, 1=FIRE, 2=UP, 3=RIGHT, 4=LEFT, 5=DOWN
    # 6=UPRIGHT, 7=UPLEFT, 8=DOWNRIGHT, 9=DOWNLEFT
    # 10=UPFIRE, 11=RIGHTFIRE, 12=LEFTFIRE, 13=DOWNFIRE
    # 14=UPRIGHTFIRE, 15=UPLEFTFIRE, 16=DOWNRIGHTFIRE, 17=DOWNLEFTFIRE
    RIGHTWARD_ACTIONS = {3, 6, 8, 11, 14, 16}   # Actions that move right
    LEFTWARD_ACTIONS = {4, 7, 9, 12, 15, 17}     # Actions that move left
    DOWNWARD_ACTIONS = {5, 8, 9, 13, 16, 17}      # Actions that move down

    def __init__(self):
        # Learned danger map: (x_bin, y_bin, room) → {action: fall_count}
        self._danger_map = {}

        # Position history for fall detection (to learn FROM)
        self._pos_history = []  # [(x, y, room, action)]
        self._prev_y = None
        self._prev_x = None
        self._prev_room = None
        self._prev_action = None

        # Hardcoded Montezuma room 0 platform edges
        # These are x-values where the platform ends
        # Going further in that direction = falling
        self._platform_edges = {
            # room: [(x_left_edge, x_right_edge, y_level, tolerance)]
            0: [
                # Top platform (spawn level): x ≈ 5-149, y ≈ 148
                (5, 149, 148, 4),
                # Middle-left platform: x ≈ 5-60, y ≈ 185
                (5, 60, 185, 4),
                # Middle-right platform: x ≈ 100-149, y ≈ 185
                (100, 149, 185, 4),
                # Bottom platform: x ≈ 5-149, y ≈ 235
                (5, 149, 235, 4),
            ],
            1: [
                (5, 149, 148, 4),
                (5, 60, 185, 4),
                (100, 149, 185, 4),
                (5, 149, 235, 4),
            ],
        }

        # Stats
        self._falls_learned = 0
        self._actions_blocked = 0
        self._edge_warnings = 0

    def on_episode_start(self):
        """Reset per-episode tracking (learned knowledge persists)."""
        self._pos_history = []
        self._prev_y = None
        self._prev_x = None
        self._prev_room = None
        self._prev_action = None

    def _bin(self, x, y):
        """Bin position for spatial lookup."""
        return (x // self.BIN_SIZE, y // self.BIN_SIZE)

    def observe(self, game_state: dict, action_taken: int):
        """
        Call AFTER stepping the environment. Detects if a fall happened
        and records the (position, action) that caused it for future avoidance.
        """
        px = game_state["player_x"]
        py = game_state["player_y"]
        room = game_state["room"]
        in_rope_zone = self.ROPE_X_MIN <= px <= self.ROPE_X_MAX

        if self._prev_y is not None and not in_rope_zone:
            y_delta = py - self._prev_y  # Positive = falling down

            if y_delta > self.FALL_THRESHOLD:
                # A fall happened! Record the PREVIOUS position + action as dangerous
                bx, by = self._bin(self._prev_x, self._prev_y)
                key = (bx, by, self._prev_room)

                if key not in self._danger_map:
                    self._danger_map[key] = {}

                prev_act = self._prev_action
                self._danger_map[key][prev_act] = self._danger_map[key].get(prev_act, 0) + 1
                self._falls_learned += 1

                # Also learn nearby positions (the approach was dangerous too)
                for hist_x, hist_y, hist_room, hist_act in self._pos_history[-5:]:
                    hkey = (hist_x // self.BIN_SIZE, hist_y // self.BIN_SIZE, hist_room)
                    if hkey not in self._danger_map:
                        self._danger_map[hkey] = {}
                    self._danger_map[hkey][hist_act] = self._danger_map[hkey].get(hist_act, 0) + 1

        # Update history
        self._pos_history.append((px, py, room, action_taken))
        if len(self._pos_history) > 20:
            self._pos_history.pop(0)

        self._prev_x = px
        self._prev_y = py
        self._prev_room = room
        self._prev_action = action_taken

    def filter_actions(self, game_state: dict, available_actions: list) -> list:
        """
        BEFORE action selection: remove actions predicted to cause falls.

        Uses:
        1. Learned danger map (prior falls at this position)
        2. Platform edge geometry (hardcoded Montezuma knowledge)
        3. Rope zone exemption

        Returns filtered action list (always at least 1 action).
        """
        px = game_state["player_x"]
        py = game_state["player_y"]
        room = game_state["room"]

        # Exempt rope zone
        if self.ROPE_X_MIN <= px <= self.ROPE_X_MAX:
            return available_actions

        dangerous_actions = set()

        # ── 1. Learned danger map ─────────────────────────────────
        bx, by = self._bin(px, py)
        key = (bx, by, room)
        if key in self._danger_map:
            for act, count in self._danger_map[key].items():
                if count >= self.LEARN_THRESHOLD:
                    dangerous_actions.add(act)

        # Also check adjacent bins (danger zone is fuzzy)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                adj_key = (bx + dx, by + dy, room)
                if adj_key in self._danger_map:
                    for act, count in self._danger_map[adj_key].items():
                        if count >= self.LEARN_THRESHOLD + 1:  # Higher bar for neighbors
                            dangerous_actions.add(act)

        # ── 2. Platform edge geometry ─────────────────────────────
        edges = self._platform_edges.get(room, [])
        for x_left, x_right, y_level, tol in edges:
            if abs(py - y_level) <= tol:
                # On this platform
                edge_margin = 12  # Pixels from edge to start being cautious

                # Near left edge → don't go further left
                if px <= x_left + edge_margin:
                    dangerous_actions.update(self.LEFTWARD_ACTIONS)
                    self._edge_warnings += 1

                # Near right edge → don't go further right
                if px >= x_right - edge_margin:
                    dangerous_actions.update(self.RIGHTWARD_ACTIONS)
                    self._edge_warnings += 1

        # ── 3. Filter ────────────────────────────────────────────
        safe = [a for a in available_actions if a not in dangerous_actions]

        if dangerous_actions:
            self._actions_blocked += len(dangerous_actions)

        # Always return at least one action
        if not safe:
            # Prefer UP/NOOP as safest fallbacks
            for fallback in [2, 0, 1]:  # UP, NOOP, FIRE
                if fallback in available_actions:
                    return [fallback]
            return available_actions[:1]

        return safe

    def report(self) -> dict:
        return {
            "falls_learned": self._falls_learned,
            "danger_zones": len(self._danger_map),
            "actions_blocked": self._actions_blocked,
            "edge_warnings": self._edge_warnings,
        }
