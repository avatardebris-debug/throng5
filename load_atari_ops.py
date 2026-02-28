"""
load_atari_ops.py
=================
Utility: read experiments/atari_active_ops.json and translate hypothesis
enactions into concrete training parameters.

Usage (in benchmark_human.py or eval_atari_agent.py):

    from load_atari_ops import AtariActiveOps
    ops = AtariActiveOps(game_id="ALE/MontezumaRevenge-v5")

    # Override AgentConfig before constructing PortableNNAgent
    cfg.imitation_phase_steps = ops.imitation_phase_steps(cfg.imitation_phase_steps)

    # Override priority when pushing to replay buffer
    priority = ops.priority_multiplier(flags)  # flags: dict with near_death etc.
    buf.push(feat, reward, next_feats, done, priority_scale=priority)

No hard dependency on experiments.db — reads a plain JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT        = Path(__file__).resolve().parent
_ACTIVE_PATH = _ROOT / "experiments" / "atari_active_ops.json"


class AtariActiveOps:
    """
    Loads atari_active_ops.json and exposes helpers to apply the ops
    during training / eval.
    """

    def __init__(
        self,
        game_id: str,
        ops_path: str | Path = _ACTIVE_PATH,
        verbose: bool = True,
    ) -> None:
        self.game_id   = game_id
        self._all_ops: list[dict] = []

        path = Path(ops_path)
        if path.exists():
            try:
                self._all_ops = json.loads(path.read_text(encoding="utf-8"))
                if verbose:
                    n = len(self._game_ops())
                    if n:
                        print(f"[atari_ops] Loaded {n} active op(s) for {game_id}")
            except Exception as exc:
                if verbose:
                    print(f"[atari_ops] Could not read {path}: {exc}")
        else:
            if verbose:
                print(f"[atari_ops] No active ops file found at {path}")

    def _game_ops(self) -> list[dict]:
        """Filter ops for this specific game (or ops that apply to ALL games)."""
        return [
            op for op in self._all_ops
            if not op.get("game") or op["game"] == self.game_id
        ]

    # ── enaction helpers ──────────────────────────────────────────────

    def imitation_phase_steps(self, default: int) -> int:
        """
        Return the override imitation_phase_steps if a phase_extend op exists,
        otherwise return default.
        """
        for op in self._game_ops():
            if op.get("enaction", {}).get("type") == "phase_extend":
                override = op["enaction"].get("phase_steps")
                if isinstance(override, int) and override > 0:
                    return override
        return default

    def priority_multiplier(self, flags: dict[str, Any]) -> float:
        """
        Given transition flags (near_death, high_entropy, disagree, etc.),
        return the maximum priority multiplier from any matching priority_boost op.

        flags example: {"near_death": True, "disagree": True, "high_entropy": False}
        """
        multiplier = 1.0
        for op in self._game_ops():
            enact = op.get("enaction", {})
            if enact.get("type") != "priority_boost":
                continue
            condition = enact.get("condition", "")
            if self._condition_matches(condition, flags):
                mult = float(enact.get("multiplier", 1.0))
                multiplier = max(multiplier, mult)
        return multiplier

    @staticmethod
    def _condition_matches(condition: str, flags: dict[str, Any]) -> bool:
        """
        Check whether a priority_boost condition string matches the given flags.
        Supported conditions: near_death, room_boundary, high_entropy,
                              disagree, high_conf_disagree
        """
        if not condition:
            return True
        c = condition.lower().replace("-", "_")
        return bool(flags.get(c, False))

    def imitation_alpha(self, action_name: str, default: float) -> float:
        """
        Return the imitation alpha for a specific action (or default if no op matches).
        """
        for op in self._game_ops():
            enact = op.get("enaction", {})
            if enact.get("type") != "imitation_weight":
                continue
            target = enact.get("action", "ALL")
            if target == "ALL" or target == action_name:
                alpha = float(enact.get("alpha", default))
                return alpha
        return default

    def summary(self) -> list[dict]:
        """Return the active ops for this game (for logging)."""
        return self._game_ops()
