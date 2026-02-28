"""
human_recorder.py — Record RAM + actions during human play for analysis.

Captures frame-by-frame snapshots of:
  - Full RAM state (128 bytes for ALE, 2048 bytes for NES)
  - Action taken
  - Reward received
  - Done flag

After recording, the data can be analyzed by RAMSemanticMapper to
auto-discover which bytes correspond to game objects, positions,
inventory, etc.

Usage:
    recorder = HumanRecorder("montezuma_session_1")
    recorder.start(env)

    # During human play:
    recorder.record(ram, action, reward, done)

    # After:
    recorder.save("experiments/recordings/")
    analysis = recorder.analyze()
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class HumanRecorder:
    """
    Records human gameplay for offline analysis.

    Stores frame-by-frame RAM snapshots, actions, and rewards.
    Can be analyzed to discover game objects, subgoals, and
    successful action sequences.
    """

    def __init__(self, session_name: str = ""):
        self.session_name = session_name or f"session_{int(time.time())}"
        self._frames: List[Dict[str, Any]] = []
        self._recording: bool = False
        self._start_time: float = 0.0
        self._prev_ram: Optional[np.ndarray] = None
        self._episode: int = 0
        self._step: int = 0

    def start(self, env=None) -> None:
        """Start recording. Optionally capture initial RAM."""
        self._recording = True
        self._start_time = time.time()
        self._step = 0

        if env is not None and hasattr(env, 'get_ram'):
            self._prev_ram = env.get_ram().copy()

    def stop(self) -> None:
        """Stop recording."""
        self._recording = False

    def record(
        self,
        ram: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        info: Optional[Dict] = None,
    ) -> None:
        """Record a single frame."""
        if not self._recording:
            return

        ram = np.asarray(ram, dtype=np.uint8)

        # Compute RAM diff from previous frame
        diff_bytes = []
        if self._prev_ram is not None:
            changed = np.where(ram != self._prev_ram)[0]
            for idx in changed:
                diff_bytes.append({
                    "addr": int(idx),
                    "old": int(self._prev_ram[idx]),
                    "new": int(ram[idx]),
                })

        frame = {
            "step": self._step,
            "episode": self._episode,
            "time": round(time.time() - self._start_time, 4),
            "action": action,
            "reward": reward,
            "done": done,
            "ram_snapshot": ram.tolist(),
            "ram_diff": diff_bytes,
            "n_changes": len(diff_bytes),
        }

        if info:
            frame["info"] = {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}

        self._frames.append(frame)
        self._prev_ram = ram.copy()
        self._step += 1

        if done:
            self._episode += 1

    def save(self, directory: str = "experiments/recordings") -> str:
        """Save recording to JSONL file. Returns filepath."""
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, f"{self.session_name}.jsonl")

        with open(filepath, "w") as f:
            for frame in self._frames:
                # Convert RAM snapshot to compact hex for storage
                ram_hex = "".join(f"{b:02x}" for b in frame["ram_snapshot"])
                compact = {**frame, "ram_hex": ram_hex}
                del compact["ram_snapshot"]  # Save space
                f.write(json.dumps(compact) + "\n")

        return filepath

    @staticmethod
    def load(filepath: str) -> List[Dict[str, Any]]:
        """Load recording from JSONL file."""
        frames = []
        with open(filepath) as f:
            for line in f:
                frame = json.loads(line.strip())
                # Restore RAM from hex
                if "ram_hex" in frame:
                    ram_bytes = bytes.fromhex(frame["ram_hex"])
                    frame["ram_snapshot"] = list(ram_bytes)
                    del frame["ram_hex"]
                frames.append(frame)
        return frames

    def analyze(self) -> Dict[str, Any]:
        """
        Quick analysis of the recording.

        Returns:
            Summary with subgoal candidates, hotspot bytes, reward events.
        """
        if not self._frames:
            return {"empty": True}

        # Find reward events
        reward_events = [
            {"step": f["step"], "reward": f["reward"], "diff": f["ram_diff"]}
            for f in self._frames if f["reward"] != 0
        ]

        # Find death events
        death_events = [
            {"step": f["step"], "episode": f["episode"], "diff": f["ram_diff"]}
            for f in self._frames if f["done"] and f["reward"] <= 0
        ]

        # Find most-changed RAM bytes
        change_counts: Dict[int, int] = {}
        for f in self._frames:
            for d in f["ram_diff"]:
                addr = d["addr"]
                change_counts[addr] = change_counts.get(addr, 0) + 1

        # Classify bytes by change frequency
        total_frames = len(self._frames)
        position_candidates = []  # Change almost every frame
        state_candidates = []     # Change rarely
        counter_candidates = []   # Change steadily

        for addr, count in sorted(change_counts.items(), key=lambda x: -x[1]):
            freq = count / total_frames
            if freq > 0.5:
                position_candidates.append({"addr": addr, "freq": round(freq, 3)})
            elif freq < 0.05:
                state_candidates.append({"addr": addr, "changes": count})
            else:
                counter_candidates.append({"addr": addr, "freq": round(freq, 3)})

        # Identify subgoal transitions: RAM changes at reward moments
        subgoal_bytes = set()
        for event in reward_events:
            for d in event["diff"]:
                subgoal_bytes.add(d["addr"])

        return {
            "total_frames": total_frames,
            "episodes": self._episode,
            "reward_events": len(reward_events),
            "death_events": len(death_events),
            "position_candidates": position_candidates[:10],
            "state_candidates": state_candidates[:20],
            "counter_candidates": counter_candidates[:10],
            "subgoal_bytes": sorted(subgoal_bytes),
            "reward_details": reward_events[:20],
        }

    def get_subgoal_sequences(self) -> List[Dict[str, Any]]:
        """
        Extract action sequences between reward events.

        Each sequence represents a "subgoal" — the actions taken
        between two reward moments.
        """
        sequences = []
        current_actions = []
        current_start = 0

        for f in self._frames:
            current_actions.append(f["action"])

            if f["reward"] > 0:
                sequences.append({
                    "start_step": current_start,
                    "end_step": f["step"],
                    "actions": current_actions.copy(),
                    "length": len(current_actions),
                    "reward": f["reward"],
                    "ram_changes": f["ram_diff"],
                })
                current_actions = []
                current_start = f["step"] + 1

            elif f["done"]:
                current_actions = []
                current_start = f["step"] + 1

        return sequences

    def report(self) -> Dict[str, Any]:
        return {
            "session": self.session_name,
            "recording": self._recording,
            "frames": len(self._frames),
            "episodes": self._episode,
            "steps": self._step,
        }
