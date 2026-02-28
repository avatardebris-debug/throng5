"""
montezuma_frontier.py
=====================
FrontierManager — automated checkpoint curriculum for Montezuma's Revenge.

Instead of manually pressing S to save checkpoints, this watches the subgoal
detector for *first-time* subgoal fires and automatically advances the training
frontier to that position.

Death-trap protection: if the agent dies within TRAP_STEPS steps for
TRAP_COUNT consecutive episodes after a frontier advance, the frontier
is automatically rolled back to the previous safe position.

Usage::

    fm = FrontierManager(save_dir, start_state_path=initial_path)
    # inside episode loop, after subgoal events:
    for ev in events:
        fm.on_subgoal(ev.name, x, y, step, env, render_env)
    # at episode end:
    fm.record_episode_steps(steps_taken)
    # pass current frontier into next episode:
    path = fm.begin_episode()
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Ordered subgoal names — used to decide which frontier is "deeper"
_SUBGOAL_ORDER: list[str] = [
    "right_lower_ladder_top",
    "center_ladder_descended",
    "lower_platform_right",
    "rope_grabbed",
    "right_platform_reached",
    "right_ladder_entered",
    "lower_floor_reached",
    "right_ladder_descended",
    "skull_zone_crossed",
    "left_ladder_base",        # x=19-22, y=148 — at base of left ladder
    "left_ladder_climbing",    # x=19-22, y=150-181 — mid-ascent
    "key_side_reached",        # x<=20, y<=155 — all the way to left wall
    "key_corner_reached",
    "key_collected",
    "key_left_descent",        # after key: descending left ladder (x=18-23, y=153-181)
    "key_floor_returned",      # after key: back on floor (y=148-154) — new frontier save
    "skull_zone_cleared",
    "right_ladder_climbed",
    "rope_returned",
    "door_threshold",
    "room_advanced",
]

_SUBGOAL_RANK: dict[str, int] = {name: i for i, name in enumerate(_SUBGOAL_ORDER)}


@dataclass
class FrontierEvent:
    subgoal:  str
    step:     int
    x:        int
    y:        int
    episode:  int
    saved_as: str
    path:     "Path | None" = field(default=None, repr=False)


class FrontierManager:
    """
    Tracks first-time subgoal fires and auto-advances the training frontier.

    The "frontier" is the ALE checkpoint saved at the step a new subgoal
    first fired this run. Subsequent episodes load from this frontier so
    the agent always starts from its furthest achieved position.

    Death-trap protection: consecutive quick deaths trigger automatic rollback.
    """

    TRAP_STEPS = 50   # deaths within this many steps = potential death trap
    TRAP_COUNT = 1    # even ONE trapped episode triggers rollback

    def __init__(
        self,
        save_dir: "Path | str",
        start_state_path: "Path | str | None" = None,
        freeze: bool = False,
        min_rank_to_advance: int = 3,
        commit_delay: int = 60,   # steps agent must survive before frontier commits
    ) -> None:
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._freeze       = freeze
        self._min_rank     = min_rank_to_advance
        self._commit_delay = commit_delay

        # Frontier stack — index 0 = original, last = current
        _initial = Path(start_state_path) if start_state_path else None
        self._stack: list["Path | None"] = [_initial]

        # Subgoals seen this session
        self._seen: set[str] = set()

        # History of frontier advancements
        self.advancements: list[FrontierEvent] = []

        # Episode counter
        self._episode = 0

        # Death-trap tracking
        self._trap_deaths = 0
        self._last_advance_ep = 0

        # Deferred frontier save (pending commit)
        self._pending: "dict | None" = None

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self) -> "Path | None":
        """Call at start of each episode. Returns checkpoint path to load."""
        self._episode += 1
        self.cancel_pending()   # dying ended the episode — cancel any pending save
        return self.frontier_path

    def record_episode_steps(self, steps: int) -> bool:
        """
        Call at end of each episode with min_steps_per_life from run_episode.
        Returns True if a death-trap rollback was triggered.
        """
        if not self.advancements:
            return False
        if steps < self.TRAP_STEPS:
            self._trap_deaths += 1
            if self._trap_deaths >= self.TRAP_COUNT:
                fp = self.frontier_path
                print(
                    f"\n  [frontier] ⚠  DEATH TRAP: min survival={steps} steps "
                    f"< {self.TRAP_STEPS} — rolling back from "
                    f"{fp.name if fp else 'spawn'}"
                )
                self.rollback()
                return True
        else:
            self._trap_deaths = 0   # survived well → reset counter
        return False

    # ------------------------------------------------------------------
    # Subgoal hook
    # ------------------------------------------------------------------

    def on_subgoal(
        self,
        name: str,
        x: int,
        y: int,
        step: int,
        env: Any,
        render_env: Any = None,
    ) -> bool:
        """
        Called when a subgoal event fires. Saves the frontier immediately.
        Returns True if a new frontier was saved.
        """
        if self._freeze:
            return False
        if name in self._seen:
            return False
        self._seen.add(name)

        rank = _SUBGOAL_RANK.get(name, -1)
        if rank < self._min_rank:
            return False
        if rank <= self._frontier_rank():
            return False

        # Immediate commit — no survival delay
        return self._commit_now(name=name, rank=rank, step=step,
                                x=x, y=y, env=env, render_env=render_env)

    def _commit_now(self, name: str, rank: int, step: int,
                    x: int, y: int, env: Any, render_env: Any) -> bool:
        """Actually write a frontier file right now, no survival guard."""
        try:
            ts    = int(time.time())
            fname = (f"frontier_{name}_ep{self._episode:04d}"
                     f"_s{step:06d}_{ts}.bin")
            fpath = self._save_dir / fname
            with open(fpath, "wb") as f:
                pickle.dump({
                    "subgoal": name, "rank": rank,
                    "step":    step, "episode": self._episode,
                    "x": x, "y": y,
                    "ram": env.unwrapped.ale.cloneState(),
                    "rgb": (render_env.unwrapped.ale.cloneState()
                            if render_env is not None
                            else env.unwrapped.ale.cloneState()),
                }, f)
            prev = self.frontier_path.name if self.frontier_path else "spawn"
            self._stack.append(fpath)
            self._last_advance_ep = self._episode
            self._trap_deaths = 0
            ev = FrontierEvent(
                subgoal=name, step=step, x=x, y=y,
                episode=self._episode, saved_as=fname, path=fpath,
            )
            self.advancements.append(ev)
            print(
                f"\n  [frontier] ✅ SAVED: {name} (x={x},y={y}) step={step}\n"
                f"             {prev} → {fname}"
            )
            return True
        except Exception as exc:
            print(f"\n  [frontier] WARNING: commit failed for {name}: {exc}")
            return False

    def try_commit(self, step: int, env: Any, render_env: Any = None) -> bool:
        """Legacy shim — immediately commits any pending save (no delay needed)."""
        if self._pending is None:
            return False
        p = self._pending
        self._pending = None
        return self._commit_now(
            name=p["name"], rank=p["rank"], step=p["step"],
            x=p["x"], y=p["y"],
            env=env or p.get("env"), render_env=render_env or p.get("render_env"),
        )

    def cancel_pending(self) -> None:
        """Cancel pending frontier if agent died before commitment."""
        if self._pending is not None:
            name = self._pending["name"]
            self._pending = None
            self._seen.discard(name)     # unlock for next attempt
            print(f"  [frontier] ✗ CANCELLED pending '{name}' (died before commit)")

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback(self) -> bool:
        """Revert to the previous frontier. Returns True if rollback happened."""
        if len(self._stack) <= 1:
            print("  [frontier] Already at starting point, cannot roll back.")
            return False
        discarded = self._stack.pop()
        fp = self.frontier_path
        print(
            f"  [frontier] ⏪  ROLLED BACK: "
            f"{discarded.name if discarded else 'spawn'} "
            f"→ {fp.name if fp else 'spawn'}"
        )
        self._trap_deaths = 0
        if self.advancements:
            rolled_subgoal = self.advancements[-1].subgoal
            self.advancements.pop()
            # Allow this subgoal to fire again (agent gets another attempt)
            self._seen.discard(rolled_subgoal)
            print(f"  [frontier] Unlocked subgoal '{rolled_subgoal}' for retry")
        return True

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    @property
    def frontier_path(self) -> "Path | None":
        return self._stack[-1] if self._stack else None

    @property
    def n_advancements(self) -> int:
        return len(self.advancements)

    def _frontier_rank(self) -> int:
        if not self.advancements:
            return -1
        return _SUBGOAL_RANK.get(self.advancements[-1].subgoal, -1)

    def summary(self) -> str:
        lines = ["  [frontier] Frontier advancements this run:"]
        if not self.advancements:
            return "  [frontier] No advancement this run."
        for adv in self.advancements:
            lines.append(
                f"    ep={adv.episode:3d}  step={adv.step:6d}  "
                f"{adv.subgoal:<28}  x={adv.x:3d}  y={adv.y:3d}"
            )
        lines.append(f"  [frontier] Final frontier: {self.frontier_path}")
        return "\n".join(lines)
