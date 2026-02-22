"""
throng4/storage/atari_event.py
==============================
Canonical per-step event schema for human play analysis.

Both paths feed the same format:
  Option 2 — live: emitted during human play using agent Q-values → softmax
  Option 1 — post-hoc: emitted by imitation-trained agent replaying states

Event (one per step):
  game, episode, step, human_action, agent_topk, entropy,
  disagree, reward, done, flags{near_death, novel_state}

Aggregate (per episode/game summary):
  alignment_rate, high_conf_disagree_rate,
  disagree_near_terminal_rate, mean_entropy, calibration_proxy

Output: JSONL  (one JSON object per line, one file per session)
  experiments/atari_events/<game_slug>/<session_id>.jsonl
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_EVENTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "atari_events"

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _softmax(q: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    q = np.asarray(q, dtype=np.float64)
    q = q - q.max()
    e = np.exp(q)
    return e / e.sum()


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats."""
    p = np.clip(probs, 1e-9, 1.0)
    return float(-np.sum(p * np.log(p)))


def _topk(probs: np.ndarray, action_meanings: list[str], k: int = 3) -> list[dict]:
    """Return top-k {action, p} dicts sorted by probability descending."""
    idx = np.argsort(probs)[::-1][:k]
    return [{"action": action_meanings[i], "p": round(float(probs[i]), 4)}
            for i in idx]


# ─────────────────────────────────────────────────────────────────────
# Per-session JSONL logger
# ─────────────────────────────────────────────────────────────────────

class AtariEventLogger:
    """
    Writes canonical per-step events to
    experiments/atari_events/<game_slug>/<session_id>.jsonl

    Usage
    -----
    logger = AtariEventLogger(game_id, action_meanings, session_id)
    logger.begin_episode(episode_idx)
    logger.log_step(step, human_action_idx, q_values, reward, done, near_death)
    ...
    logger.end_session()   # flushes and closes the file
    """

    def __init__(
        self,
        game_id: str,
        action_meanings: list[str],
        session_id: str,
        top_k: int = 3,
    ):
        self.game_id = game_id
        self.action_meanings = action_meanings
        self.session_id = session_id
        self.top_k = top_k
        self._episode: int = 0

        slug = game_id.replace("/", "_").replace("-", "_")
        out_dir = _EVENTS_DIR / slug
        out_dir.mkdir(parents=True, exist_ok=True)
        self._path = out_dir / f"{session_id}.jsonl"
        self._fh = self._path.open("w", encoding="utf-8")

    def begin_episode(self, episode: int) -> None:
        self._episode = episode

    def log_step(
        self,
        step: int,
        human_action_idx: int,
        q_values: np.ndarray | list[float] | None,
        reward: float,
        done: bool,
        near_death: bool = False,
        novel_state: bool = False,
    ) -> dict[str, Any]:
        """
        Compute probabilities from raw Q-values and write one event line.

        q_values: Q(s,a) for every action in order — will be softmaxed.
                  Pass None or uniform zeros if the agent is disabled.
        Returns the event dict (also written to JSONL).
        """
        n = len(self.action_meanings)

        if q_values is None or len(q_values) == 0:
            probs = np.ones(n, dtype=np.float64) / n   # uniform prior
        else:
            q = np.array(q_values, dtype=np.float64)
            # Pad / truncate to match action space
            if len(q) < n:
                q = np.pad(q, (0, n - len(q)))
            else:
                q = q[:n]
            probs = _softmax(q)

        agent_greedy = int(np.argmax(probs))
        disagree = bool(human_action_idx != agent_greedy)
        high_conf = bool(probs[agent_greedy] >= 0.5)

        event: dict[str, Any] = {
            "game":         self.game_id,
            "episode":      self._episode,
            "step":         step,
            "human_action": self.action_meanings[
                min(human_action_idx, n - 1)
            ],
            "agent_topk":   _topk(probs, self.action_meanings, self.top_k),
            "entropy":      round(_entropy(probs), 4),
            "disagree":     disagree,
            "high_conf_disagree": disagree and high_conf,
            "reward":       round(float(reward), 4),
            "done":         done,
            "flags": {
                "near_death":   near_death,
                "novel_state":  novel_state,
            },
        }
        self._fh.write(json.dumps(event) + "\n")
        return event

    def end_session(self) -> Path:
        self._fh.flush()
        self._fh.close()
        return self._path

    @property
    def path(self) -> Path:
        return self._path


# ─────────────────────────────────────────────────────────────────────
# Aggregator — loads JSONL files and computes summary stats
# ─────────────────────────────────────────────────────────────────────

def _load_events(game_id: str) -> list[dict]:
    """Load all events for a game from all session JSONL files."""
    slug = game_id.replace("/", "_").replace("-", "_")
    game_dir = _EVENTS_DIR / slug
    if not game_dir.exists():
        return []
    events = []
    for f in sorted(game_dir.glob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events


def aggregate_game(game_id: str) -> dict[str, Any]:
    """
    Compute per-game summary statistics from all logged events.

    Returns
    -------
    {
      "game": ...,
      "n_steps": ...,
      "n_episodes": ...,
      "alignment_rate":               P(human == agent_greedy)
      "high_conf_disagree_rate":      P(disagree AND agent confident ≥50%)
      "disagree_near_terminal_rate":  P(disagree | near_death OR done within 10 steps)
      "mean_entropy":                 average H(agent distribution)
      "calibration_proxy":            P(human chose agent top-1 | agent entropy <1.0)
      "reward_on_agree":              mean reward when agent agreed with human
      "reward_on_disagree":           mean reward when agent disagreed
      "top_human_actions":            most frequent human actions {action: count}
      "top_disagreement_actions":     human actions most often disagreed on
    }
    """
    events = _load_events(game_id)
    if not events:
        return {"game": game_id, "n_steps": 0, "error": "no events found"}

    n = len(events)
    n_agree      = sum(1 for e in events if not e["disagree"])
    n_hc_dis     = sum(1 for e in events if e.get("high_conf_disagree", False))
    entropies    = [e["entropy"] for e in events]
    rewards_agree   = [e["reward"] for e in events if not e["disagree"]]
    rewards_dis     = [e["reward"] for e in events if     e["disagree"]]

    # Near-terminal: near_death flag OR done=True within 10 steps of episode end
    # Simple proxy: near_death OR done
    near_term    = [e for e in events if e["flags"]["near_death"] or e["done"]]
    n_dis_near   = sum(1 for e in near_term if e["disagree"])

    # Calibration: when agent is confident (entropy < 1.0), how often does human agree?
    confident    = [e for e in events if e["entropy"] < 1.0]
    n_calib_agree = sum(1 for e in confident if not e["disagree"])

    # Top human actions
    from collections import Counter
    human_ctr = Counter(e["human_action"] for e in events)
    disagree_ctr = Counter(e["human_action"] for e in events if e["disagree"])

    n_episodes = len({(e["episode"],) for e in events})

    return {
        "game":                       game_id,
        "n_steps":                    n,
        "n_episodes":                 n_episodes,
        "alignment_rate":             round(n_agree / n, 4),
        "high_conf_disagree_rate":    round(n_hc_dis / n, 4),
        "disagree_near_terminal_rate":round(n_dis_near / max(len(near_term), 1), 4),
        "mean_entropy":               round(float(np.mean(entropies)), 4),
        "calibration_proxy":          round(n_calib_agree / max(len(confident), 1), 4),
        "reward_on_agree":            round(float(np.mean(rewards_agree))
                                           if rewards_agree else 0.0, 4),
        "reward_on_disagree":         round(float(np.mean(rewards_dis))
                                           if rewards_dis else 0.0, 4),
        "top_human_actions":          dict(human_ctr.most_common(8)),
        "top_disagreement_actions":   dict(disagree_ctr.most_common(5)),
    }


def aggregate_all(game_ids: list[str] | None = None) -> list[dict]:
    """
    Aggregate all games (or specified subset).
    Returns list of per-game summary dicts.
    """
    if game_ids is None:
        # Auto-detect from directory structure
        if not _EVENTS_DIR.exists():
            return []
        game_ids = [
            p.name.replace("_v5", "-v5").replace("_", "/", 1)
            for p in _EVENTS_DIR.iterdir() if p.is_dir()
        ]
    return [aggregate_game(g) for g in sorted(game_ids)]
