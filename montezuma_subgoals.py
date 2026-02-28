"""
montezuma_subgoals.py
=====================
RAM-based sub-goal detector and shaped reward engine for Montezuma's Revenge.

The game is deterministic from the same seed. The start room has a known
spatial layout encoded in RAM that we can use to define explicit sub-goals
and assign bonus shaped rewards for reaching milestones — without requiring
any game reward signal.

Architecture
------------
  SubgoalDetector.check(ram_bytes) → list[SubgoalEvent]
  ShapedRewardEngine.step(ram_bytes, game_reward, done) → (shaped_reward, events)
  NoveltyTracker.visit(ram_bytes) → novelty_bonus

RAM addresses (calibrated / high-confidence):
  RAM[3]   = room number  (1=start room where player spawns)
  RAM[42]  = player_x
  RAM[43]  = player_y
  RAM[58]  = lives (5=full)
  RAM[56]  = key item byte (0=no key, 255=key collected)
  RAM[65]  = key secondary byte (bit1 set when key held; 0x02)

Room 1 FULL layout — 3 ladders, skull, key, return trip, door:
  (verified: spawn x=77,y=235; rope zone x=112-133,y=148-180 from live runs)
  (verified: key at x=15,y=201 room=1 from human session RAM log 2026-02-22)
  (marked [est] = estimated, not yet confirmed from RAM log)

  OUTBOUND:
    Spawn              x~77,  y~235   top center-left
    Center ladder v    x~70,  y>230   descend to lower platform
    Lower platform ->  x>100, y~210-240  run right
    Rope swing ->      x>108, y<195   grab rope, cross gap
    Right platform     x>138, y~145-185  landed right side
    Right ladder v     x>130, y>210  [est] descend to skull level
    Skull zone cross<- x<50,  y>210  [est] cross under skull toward left
    Left ladder ^      x<30,  y<210  climbing up toward key [est]
    Key corner         x<25,  y in 185-215, room=1  CALIBRATED: key at x=15,y=201

  RETURN:
    Skull zone cross-> x>100, y>210  [est] after key, cross skull again
    Right ladder ^     x>130, y<185  [est] climb back up
    Rope swing <-      x<120, y<195  [est] cross back left
    Door threshold     y<175, x~40-115  [est] top of room, left or right door

  NOTE: [est] coordinates will auto-calibrate as the agent reaches them.
  Enable --verbose and watch actual x/y values at each new sub-goal fire.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Sub-goal definitions
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Subgoal:
    name: str
    description: str
    reward: float          # shaped bonus when first triggered per episode
    room: int | None       # None = any room
    calibrated: bool = True   # False = coordinate estimate, needs verification


# Full room 1 sub-goal sequence — 14 milestones covering both legs
# Rewards escalate sharply: return-leg milestones > outbound, room_advanced >> all
STAGE_1_SUBGOALS: list[Subgoal] = [
    # ── Outbound leg ──────────────────────────────────────────────────
    Subgoal("right_lower_ladder_top",
            "Reached top of right lower ladder (x<73, y<235)",
            reward=0.3, room=1),
    Subgoal("center_ladder_descended",
            "Descended center ladder to lower platform (x in 60-90, y>230)",
            reward=0.8, room=1),
    Subgoal("lower_platform_right",
            "Running right on lower platform (x>100, y in 205-240)",
            reward=1.2, room=1),
    Subgoal("rope_grabbed",
            "Grabbed rope/vine and swinging right (x>108, y<195)",
            reward=2.0, room=1),
    Subgoal("right_platform_reached",
            "Landed on right platform after rope (x>128, y in 145-185)",
            reward=3.0, room=1),
    Subgoal("right_ladder_entered",
            "Started descending right ladder from platform (x>130, y in 148-165)",
            reward=1.5, room=1),
    Subgoal("lower_floor_reached",
            "Reached the lowest floor level (y<=150, any x) — CALIBRATED y=148",
            reward=2.0, room=1, calibrated=True),
    Subgoal("right_ladder_descended",
            "Descended right ladder to bottom (x>130, y<=152)",
            reward=4.5, room=1, calibrated=True),
    Subgoal("skull_zone_crossed",
            "Crossed skull zone leftward (x<=37, y<=155, bottom floor)",
            reward=6.0, room=1, calibrated=True),
    Subgoal("left_ladder_base",
            "Reached base of left ladder — CALIBRATED x=19-22, y=148",
            reward=6.3, room=1, calibrated=True),
    Subgoal("left_ladder_climbing",
            "Mid-ascent on left ladder — CALIBRATED x=19-22, y=150-181",
            reward=6.6, room=1, calibrated=True),
    Subgoal("key_side_reached",
            "Climbed left ladder to upper platform (x<=25, y>=185) — CALIBRATED x=21,y=195",
            reward=7.0, room=1, calibrated=True),
    Subgoal("key_corner_reached",
            "Near key on left wall (x<25, y in 185-215, room=1) CALIBRATED x=15,y=201",
            reward=8.0, room=1, calibrated=True),
    Subgoal("key_collected",
            "Key picked up — RAM[56]==0xFF AND RAM[65]&0x02 (dual-byte confirmed)",
            reward=5000.0, room=None),
    # ── Return leg (calibrated) ───────────────────────────────────────
    Subgoal("key_left_descent",
            "Descending left ladder after key (x=18-23, y=153-181) — CALIBRATED",
            reward=5006.0, room=1, calibrated=True),
    Subgoal("key_floor_returned",
            "Back on floor after key (y<=154, x<=37) — save new frontier — CALIBRATED",
            reward=5012.0, room=1, calibrated=True),
    # ── Return leg (estimated) ────────────────────────────────────────
    Subgoal("skull_zone_cleared",
            "Crossed skull zone rightward on return (x>37, y<=155, after key) [est]",
            reward=5020.0, room=1, calibrated=False),
    Subgoal("right_ladder_climbed",
            "Climbed back up right ladder (x>130, y<185, after key) [est]",
            reward=5030.0, room=1, calibrated=False),
    Subgoal("rope_returned",
            "Swung back left across rope (x<120, y<195, after right_ladder_climbed) [est]",
            reward=5040.0, room=1, calibrated=False),
    Subgoal("door_threshold",
            "Reached door area at top of room (y<175, x in 40-115, after key) [est]",
            reward=5050.0, room=1, calibrated=False),
    # ── Terminal ──────────────────────────────────────────────────────
    Subgoal("room_advanced",
            "Advanced to a higher room number (RAM[3] increased)",
            reward=10000.0, room=None),
]

# Index by name for quick lookup
_SUBGOAL_MAP = {sg.name: sg for sg in STAGE_1_SUBGOALS}


# ──────────────────────────────────────────────────────────────────────
# Sub-goal event
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SubgoalEvent:
    name: str
    reward: float
    room: int
    player_x: int
    player_y: int
    step: int


# ──────────────────────────────────────────────────────────────────────
# Sub-goal detector
# ──────────────────────────────────────────────────────────────────────

class SubgoalDetector:
    """
    Stateful detector — tracks which sub-goals have been awarded this episode
    so each bonus fires at most once per episode.
    """

    def __init__(self) -> None:
        self._awarded: set[str] = set()
        self._prev_room: int | None = None
        self._prev_key_byte: int | None = None
        self._prev_key2_byte: int | None = None

    # Sub-goals to unlock when the key is collected, so they can re-fire
    # on the return leg (the agent traverses these zones a second time).
    # Each can still only fire once per leg — not infinitely.
    _POST_KEY_RESET: frozenset[str] = frozenset({
        "rope_grabbed",           # agent grabs rope again going left
        "right_platform_reached", # agent swings back to right platform
        "lower_platform_right",   # agent re-crosses lower platform toward center
        "right_lower_ladder_top", # agent ascends right lower ladder toward door
    })

    def reset(self) -> None:
        """Call at the start of every episode."""
        self._awarded.clear()
        self._prev_room = None
        self._prev_key_byte = None
        self._prev_key2_byte = None

    def check(self, ram: bytes | list, step: int = 0) -> list[SubgoalEvent]:
        """
        Given a 128-byte RAM array, return list of newly triggered sub-goals
        (those not already awarded this episode).
        """
        x     = int(ram[42])
        y     = int(ram[43])
        room  = int(ram[3])
        k56   = int(ram[56])
        k65   = int(ram[65])
        has_key = "key_collected" in self._awarded

        events: list[SubgoalEvent] = []

        def _award(name: str) -> None:
            if name not in self._awarded:
                sg = _SUBGOAL_MAP[name]
                self._awarded.add(name)
                events.append(SubgoalEvent(
                    name=name,
                    reward=sg.reward,
                    room=room,
                    player_x=x,
                    player_y=y,
                    step=step,
                ))
                # When key is picked up, unlock outbound spatial milestones
                # so the return-leg traversal earns gradient signal too.
                if name == "key_collected":
                    self._awarded -= self._POST_KEY_RESET

        # ── Room 1: Outbound leg ─────────────────────────────────────
        if room == 1:

            # 1. Right lower ladder top — moved left from spawn (x=77) toward ladder (x≈70)
            if x < 73 and y < 235:
                _award("right_lower_ladder_top")

            # 2. Center ladder descended — dropped from spawn (y=235) to mid-floor (y≈192)
            if 60 <= x <= 90 and y <= 200:
                _award("center_ladder_descended")

            # 3. Lower platform run right — x>100, on mid-floor (y≈192, skull level)
            if x > 100 and 185 <= y <= 200:
                _award("lower_platform_right")

            # 4. Rope grabbed — entered rope/vine zone going right
            if x > 108 and y < 195:
                _award("rope_grabbed")

            # 5. Right platform reached — crossed rope, landed on right side
            #    Calibrated: agent reaches x≈128-140 at y=145-185 (was x>138, too tight)
            if x > 128 and 145 <= y <= 185:
                _award("right_platform_reached")

            # 5b. Right ladder entered — first step DOWN from platform (y decreases toward 148)
            if "right_platform_reached" in self._awarded:
                if x > 130 and 148 <= y <= 165:
                    _award("right_ladder_entered")

            # 5c. Lower floor reached — y<=150 anywhere (floor = y=148, nothing lower exists)
            if "right_platform_reached" in self._awarded:
                if y <= 150:
                    _award("lower_floor_reached")

            # 6. Right ladder descended — at the bottom (x>130, y<=152)
            #    Gate: must have entered the ladder
            if "right_ladder_entered" in self._awarded:
                if x > 130 and y <= 152:
                    _award("right_ladder_descended")

            # 7. Skull zone crossed — on bottom floor, left of skull (x<=37, y<=155)
            #    No prerequisite gate needed — position is specific enough.
            #    Works correctly on checkpoint loads (detector starts empty each ep).
            if x <= 37 and y <= 155:
                _award("skull_zone_crossed")

            # 7a. Left ladder base — at floor level at base of left ladder
            #     CALIBRATED: x=19-22, y=148
            if "skull_zone_crossed" in self._awarded:
                if 18 <= x <= 23 and y <= 152:
                    _award("left_ladder_base")

            # 7b. Left ladder climbing — mid-ascent on left ladder
            #     CALIBRATED: x=19-22, y=150-181
            if "left_ladder_base" in self._awarded:
                if 18 <= x <= 23 and 153 <= y <= 181:
                    _award("left_ladder_climbing")

            # 7c. Key side reached — climbed left ladder to upper platform (x<=25, y>=182)
            #     Calibrated: upper left platform is x=21, y=195. Key is at x=15, y=201.
            if "skull_zone_crossed" in self._awarded:
                if x <= 25 and y >= 182:
                    _award("key_side_reached")

            # 8. Key corner reached — at the key on left wall. CALIBRATED: x=15,y=201
            if "skull_zone_crossed" in self._awarded:
                if x < 25 and 185 <= y <= 215:
                    _award("key_corner_reached")

            # ── Return leg (all gate on key_collected) ────────────────

            # 9a. Descending left ladder after key (x=18-23, y=153-181)
            if has_key:
                if 18 <= x <= 23 and 153 <= y <= 181:
                    _award("key_left_descent")

            # 9b. Returned to floor after key (y<=154, x<=37)
            #     Triggers a new frontier save — agent can now load here for skull crossing
            if has_key and "key_left_descent" in self._awarded:
                if x <= 37 and y <= 154:
                    _award("key_floor_returned")

            # 9c. Skull zone cleared on return (x>37, y<=155, after key)
            if has_key and "skull_zone_cleared" not in self._awarded:
                if x > 37 and y <= 155:
                    _award("skull_zone_cleared")

            # 10. Right ladder climbed on return [est]
            if "skull_zone_cleared" in self._awarded:
                if x > 130 and y < 185:
                    _award("right_ladder_climbed")

            # 11. Rope returned — swung back left after climbing right ladder [est]
            if "right_ladder_climbed" in self._awarded:
                if x < 120 and y < 195:
                    _award("rope_returned")

            # 12. Door threshold — top of room, approaching left or right door [est]
            if has_key and y < 175 and 40 <= x <= 115:
                _award("door_threshold")

        # ── Key collected (RAM signal, any room) ──────────────────────
        # Dual-byte check: RAM[56]==0xFF AND bit1 of RAM[65] set
        if self._prev_key_byte is not None:
            now_key = (k56 == 255) and bool(k65 & 0x02)
            was_key = (self._prev_key_byte == 255) and bool((self._prev_key2_byte or 0) & 0x02)
            if now_key and not was_key:
                _award("key_collected")

        # ── Room advanced (any room) ──────────────────────────────────
        if self._prev_room is not None and room > self._prev_room:
            _award("room_advanced")

        # Update state
        self._prev_room      = room
        self._prev_key_byte  = k56
        self._prev_key2_byte = k65

        return events



# ──────────────────────────────────────────────────────────────────────
# Novelty tracker (count-based intrinsic reward)
# ──────────────────────────────────────────────────────────────────────

class NoveltyTracker:
    """
    Count-based novelty bonus: reward for visiting (room, x_bin, y_bin) cells
    that haven't been seen before. The bonus decays as a cell is revisited.

    bonus = novelty_scale / sqrt(visit_count)
    """

    def __init__(self, bin_size: int = 8, novelty_scale: float = 0.3) -> None:
        self._bin    = bin_size
        self._scale  = novelty_scale
        self._counts: dict[tuple, int] = {}

    def reset_episode(self) -> None:
        """Optional: reset per-episode visit counts (not needed for global novelty)."""
        pass

    def visit(self, ram: bytes | list) -> float:
        """
        Register a RAM state visit; return the novelty bonus.
        Uses (room, x//bin, y//bin) as the cell key.
        """
        room  = int(ram[3])
        x_bin = int(ram[42]) // self._bin
        y_bin = int(ram[43]) // self._bin
        key   = (room, x_bin, y_bin)

        count = self._counts.get(key, 0) + 1
        self._counts[key] = count
        return self._scale / (count ** 0.5)

    @property
    def n_unique_cells(self) -> int:
        return len(self._counts)

    @property
    def n_total_visits(self) -> int:
        return sum(self._counts.values())

    def summary(self) -> dict:
        return {
            "unique_cells":  self.n_unique_cells,
            "total_visits":  self.n_total_visits,
            "most_visited":  max(self._counts.values(), default=0),
        }


# ──────────────────────────────────────────────────────────────────────
# Shaped reward engine (combines game reward + subgoals + novelty)
# ──────────────────────────────────────────────────────────────────────

class ShapedRewardEngine:
    """
    Drop-in reward shaper for Montezuma's Revenge.

    Usage in training loop:
        shaper = ShapedRewardEngine()
        shaper.reset()                       # start of each episode

        # inside step loop:
        shaped_r, events = shaper.step(ram_obs, game_reward, done)
        # use shaped_r for replay buffer / training instead of game_reward
    """

    def __init__(
        self,
        subgoal_scale:  float = 1.0,
        novelty_scale:  float = 0.3,
        novelty_bin:    int   = 5,   # 5px grid cells for finer exploration credit
        death_penalty:  float = -5.0,
        ladder_scale:   float = 0.05,
        stall_penalty_300: float = -0.01,
        stall_penalty_500: float = -0.05,
        progress_bonus:    float = 0.5,   # bonus on new max_x (forward personal-best)
        regress_penalty:   float = -0.3,  # penalty when x < max_x * regress_threshold
        regress_threshold: float = 0.90,  # fraction of max_x that triggers regress hate
        pass_through_game_reward: bool = True,
    ) -> None:
        self._sub_scale    = subgoal_scale
        self._pass         = pass_through_game_reward
        self._death_pen    = death_penalty
        self._ladder_scale = ladder_scale
        self._stall_300    = stall_penalty_300
        self._stall_500    = stall_penalty_500
        self._prog_bonus   = progress_bonus
        self._regress_pen  = regress_penalty
        self._regress_thr  = regress_threshold
        self._detector     = SubgoalDetector()
        self._novelty      = NoveltyTracker(bin_size=novelty_bin,
                                            novelty_scale=novelty_scale)
        self._prev_lives: int | None = None
        self._step         = 0
        self._steps_since_death = 0
        self._min_x        = 999  # best leftward advance (toward key platform)
        self._max_x        = 0    # best rightward advance (toward rope / door)
        # Ladder persistence state
        self._prev_y:       int | None = None
        self._prev_x:       int | None = None   # for fall / momentum detection
        self._ladder_best_y: dict[str, int] = {}
        # Rightward momentum: track best x toward rope (stage 1)
        self._best_rope_x:  int = 77            # starts at spawn x

    def reset(self) -> None:
        self._detector.reset()
        self._novelty.reset_episode()
        self._prev_lives      = None
        self._prev_y          = None
        self._ladder_best_y   = {}
        self._step            = 0
        self._steps_since_death = 0
        self._min_x           = 999  # reset pre-key leftward tracker
        self._max_x           = 0    # reset post-key rightward tracker

    def step(
        self,
        ram: bytes | list,
        game_reward: float,
        done: bool,
    ) -> tuple[float, list[SubgoalEvent]]:
        """
        Returns (shaped_reward, events_fired_this_step).
        shaped_reward is what should be stored in the replay buffer.
        """
        x = int(ram[42])
        y = int(ram[43])

        # 1. Novelty bonus
        novelty  = self._novelty.visit(ram)

        # 2. Sub-goal events
        events   = self._detector.check(ram, step=self._step)
        subgoal_bonus = sum(e.reward for e in events) * self._sub_scale

        # 3. Ladder bonus — direction-aware per ladder
        #
        #   Right ladder  (x=128-150): agent descends TO floor → reward y DECREASING
        #   Center ladder (x=58-82):   agent descends TO floor → reward y DECREASING
        #   Left ladder   (x=0-40):    agent ASCENDS to key    → reward y INCREASING
        #
        #   Also: small per-step bonus (+0.02) for staying in any ladder zone —
        #   encourages the agent not to jump off mid-climb.
        ladder_bonus = 0.0
        if self._ladder_scale > 0:
            if 58 <= x <= 82:
                lid, ascend = "center", False
            elif 128 <= x <= 150:
                lid, ascend = "right", False
            elif x <= 40 and y >= 140:    # left ladder zone, only when on/near it
                # Direction depends on whether key has been collected:
                #   Pre-key:  ascend toward key (reward y increasing)
                #   Post-key: descend back to floor (reward y decreasing)
                has_key_flag = "key_collected" in self._detector._awarded
                lid, ascend = "left", not has_key_flag
            else:
                lid, ascend = None, False

            if lid is not None:
                # Per-step stay-on-ladder bonus
                ladder_bonus += 0.02

                if self._prev_y is not None:
                    if ascend:
                        # Left ladder pre-key: reward new highest y (moving up toward key)
                        prev_best = self._ladder_best_y.get(lid, 0)   # sentinel = 0
                        if y > prev_best:
                            ladder_bonus += self._ladder_scale * (y - prev_best)
                            self._ladder_best_y[lid] = y
                    else:
                        # Right/center/left post-key: reward new lowest y (moving down)
                        prev_best = self._ladder_best_y.get(lid, 999)  # sentinel = high
                        if y < prev_best:
                            ladder_bonus += self._ladder_scale * (prev_best - y)
                            self._ladder_best_y[lid] = y

        # 3b. Ledge / void danger penalty (calibrated zone)
        #     The void zone is between the left platform (x≤33) and rope (x≈109).
        #     Any position at y=192-220 and x=34-104 (not on ladder) is mid-air falling.
        from throng4.basal_ganglia.room_constants import is_lethal_zone, Y_VOID_MIN, Y_VOID_MAX
        ledge_penalty = 0.0
        if is_lethal_zone(x, y):
            ledge_penalty = -1.0   # strong per-step discouragement while falling

        # 3c. Fall-height detection
        #     If y increases by >15 in one step AND agent is not on ladder AND
        #     not in the void zone (which already penalises) → small cliff-jump penalty.
        #     This teaches "trying new ledges leads to falling" before CNN can see geometry.
        fall_penalty = 0.0
        if self._prev_y is not None and not (9 <= x <= 40):
            dy_fall = y - self._prev_y
            if dy_fall > 15 and not is_lethal_zone(x, y):  # sudden unexpected drop
                fall_penalty = -0.3 * (dy_fall / 20.0)   # proportional to fall size

        # 4. Death penalty + per-life stall penalty
        lives = int(ram[58])
        death_penalty = 0.0
        if self._prev_lives is not None and lives < self._prev_lives:
            death_penalty = self._death_pen
            self._steps_since_death = 0   # reset per-life counter on death
        else:
            self._steps_since_death += 1

        # 5. Stall penalty — discourages loitering
        stall_penalty = 0.0
        if self._steps_since_death > 500:
            stall_penalty = self._stall_500
        elif self._steps_since_death > 300:
            stall_penalty = self._stall_300

        # 6. Stage-aware directional momentum policy (replaces simple min_x bias)
        #
        #   Stage 1 — No rope grabbed yet, no key:
        #     RIGHTWARD bias on top/center platform toward rope at x≈109.
        #     Jump bonus if on conveyor belt zone (x=65-105, y≈215-230).
        #
        #   Stage 2 — Rope grabbed, descending, no key yet:
        #     LEFT bias toward key platform (x<33).
        #
        #   Stage 3 — Key held:
        #     RIGHTWARD bias toward exit door (x>120).
        #
        from throng4.basal_ganglia.room_constants import (
            X_ROPE_MIN, Y_VOID_MIN, Y_TOP_PLATFORM,
            X_CONVEYOR_MIN, X_CONVEYOR_MAX, Y_CONVEYOR,
        )
        key_held       = bool(int(ram[56]) == 0xFF and int(ram[65]) & 0x02)
        rope_grabbed   = "rope_grabbed" in self._detector._awarded
        progress_r     = 0.0
        jump_r         = 0.0

        if key_held:
            # Stage 3: rightward toward exit
            if x > self._max_x:
                progress_r  = self._prog_bonus
                self._max_x = x
            elif self._max_x > 10 and x < self._max_x * self._regress_thr:
                progress_r  = self._regress_pen

        elif rope_grabbed:
            # Stage 2: leftward toward key
            if x < self._min_x:
                progress_r  = self._prog_bonus
                self._min_x = x

        else:
            # Stage 1: rightward toward rope (x≈109), no key grabbed yet.
            #
            # PRIMARY zone   y=192-202 (key platform level, walking toward rope):
            #   Full rightward momentum — agent at this height MUST go right to reach rope.
            # SECONDARY zone y>=225 (top platform):
            #   Softer bias — direct jump to rope is occasionally possible from top.
            on_key_level = (192 <= y <= 202)
            on_top       = (y >= 225)

            if on_key_level or on_top:
                strength = self._prog_bonus * (0.8 if on_key_level else 0.4)
                reg_str  = self._regress_pen * (0.5 if on_key_level else 0.25)
                if x > self._best_rope_x:
                    progress_r        = strength
                    self._best_rope_x = x
                elif x < self._best_rope_x - 15:
                    progress_r        = reg_str

            # Jump encouragement on conveyor belt zone (x=65-105, y≈215-240).
            # Agent must jump rightward to fight the leftward push.
            # Reward any upward-registered movement (y decrease) while on conveyor.
            on_conveyor = (X_CONVEYOR_MIN <= x <= X_CONVEYOR_MAX
                           and Y_CONVEYOR - 20 <= y <= Y_TOP_PLATFORM + 5)
            if on_conveyor and self._prev_y is not None:
                dy_up = self._prev_y - y   # positive = moving upward (jumped)
                if dy_up > 3:
                    jump_r = 0.15

        # 7. Movement bonus — small reward for any positional change this step.
        #    Encourages constant motion without being punishing for natural pauses.
        #    Stationary = 0 (no penalty), any movement = +0.03.
        #    This is separate from the stall penalty which only fires after 300+ idle steps.
        movement_bonus = 0.0
        if self._prev_x is not None and self._prev_y is not None:
            if abs(x - self._prev_x) + abs(y - self._prev_y) > 0:
                movement_bonus = 0.03

        # 8. Game reward (key pickup, door opening, score)
        game_r = game_reward if self._pass else 0.0

        shaped = (game_r + subgoal_bonus + novelty + death_penalty
                  + ladder_bonus + ledge_penalty + fall_penalty
                  + stall_penalty + progress_r + jump_r + movement_bonus)
        self._prev_lives = lives
        self._prev_y     = y
        self._prev_x     = x
        self._step      += 1

        return shaped, events

    @property
    def novelty_tracker(self) -> NoveltyTracker:
        return self._novelty

    @property
    def awarded_subgoals(self) -> set[str]:
        return set(self._detector._awarded)
