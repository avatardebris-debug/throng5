"""Montezuma runner mode: ground."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from brain.environments.human_recorder import HumanRecorder
from brain.planning.ram_semantic_mapper import RAMSemanticMapper
from brain.games.montezuma.runner.constants import (
    RAM_SIZE, RESULTS_DIR, RECORDINGS_DIR, ensure_results_dirs,
    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_ROOM, RAM_LIVES,
    RAM_SKULL_X, RAM_SKULL_Y, RAM_ITEMS,
)

def mode_ground(args):
    """
    Analyze a human recording to discover RAM semantics.

    Reads a .jsonl recording (from 'human' mode), feeds every frame
    through RAMSemanticMapper, and outputs:
    - Which RAM bytes are positions, flags, counters
    - Which bytes correlate with rewards / deaths
    - Entity groups (co-changing position bytes)
    - Subgoal sequences (action chains between reward events)
    """
    print("=" * 60)
    print("MODE: GROUNDING — Analyze Recording")
    print("=" * 60)

    recording_path = args.recording
    if not recording_path:
        ensure_results_dirs()
        # Find most recent recording
        recordings = sorted(RECORDINGS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not recordings:
            print("No recordings found. Run 'human' mode first.")
            return
        recording_path = str(recordings[-1])
        print(f"Using most recent recording: {recording_path}")

    print(f"Loading recording from: {recording_path}")

    # Load recording
    frames = []
    with open(recording_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not frames:
        print("No frames found in recording.")
        return

    print(f"  Loaded {len(frames)} frames")

    # Feed through RAM mapper
    mapper = RAMSemanticMapper(ram_size=RAM_SIZE)

    for i, frame in enumerate(frames):
        # HumanRecorder saves as 'ram_hex' (hex-encoded 128 bytes)
        ram_hex = frame.get("ram_hex", "") or frame.get("ram", "")
        action = frame.get("action", 0)
        reward = frame.get("reward", 0.0)
        done = frame.get("done", False)

        # Decode hex RAM
        if isinstance(ram_hex, str) and len(ram_hex) >= 2:
            ram = np.array([int(ram_hex[j:j+2], 16) for j in range(0, len(ram_hex), 2)],
                           dtype=np.uint8)
        elif isinstance(ram_hex, list):
            ram = np.array(ram_hex, dtype=np.uint8)
        else:
            continue

        mapper.observe(ram, action=action, reward=reward, done=done)

    # Get results
    registry = mapper.get_registry()
    subgoals = mapper.get_subgoal_bytes()
    entities = mapper.get_entity_groups()
    report = mapper.report()

    print()
    print("RAM SEMANTIC MAP")
    print("=" * 60)
    print(f"  Active bytes: {report['active_bytes']}")
    print()

    for category, entries in registry.items():
        print(f"  {category.upper()} ({len(entries)} bytes):")
        for entry in entries[:10]:
            addr = entry["addr"]
            print(f"    0x{addr:02X} (byte {addr}): "
                  f"changes={entry.get('change_count', '?')}")
        if len(entries) > 10:
            print(f"    ... and {len(entries) - 10} more")
        print()

    if subgoals:
        print(f"  SUBGOAL BYTES ({len(subgoals)}):")
        for sg in subgoals:
            print(f"    0x{sg['addr']:02X}: changed at reward "
                  f"{sg.get('reward_changes', '?')} times")
        print()

    if entities:
        print(f"  ENTITY GROUPS ({len(entities)}):")
        for i, group in enumerate(entities):
            addrs = [f"0x{a:02X}" for a in group["bytes"]]
            print(f"    Entity {i}: bytes {', '.join(addrs)} "
                  f"(type: {group.get('type', '?')})")
        print()

    # Cross-reference with known addresses
    print("KNOWN ADDRESS VERIFICATION:")
    known = {
        RAM_PLAYER_X: "player_x",
        RAM_PLAYER_Y: "player_y",
        RAM_ROOM: "room",
        RAM_LIVES: "lives",
        RAM_SKULL_X: "skull_x",
        RAM_SKULL_Y: "skull_y",
        RAM_ITEMS: "items",
    }
    for addr, name in known.items():
        discovered = "NOT found"
        for cat, entries in registry.items():
            if any(e["addr"] == addr for e in entries):
                discovered = f"found as {cat}"
                break
        print(f"  0x{addr:02X} ({name:10s}): {discovered}")

    # Save grounding data
    ensure_results_dirs()
    grounding_path = RESULTS_DIR / "grounding.json"
    grounding_data = {
        "source": recording_path,
        "frames_analyzed": len(frames),
        "registry": {k: v for k, v in registry.items()},
        "subgoal_bytes": subgoals,
        "entity_groups": entities,
        "report": report,
    }
    with open(grounding_path, "w") as f:
        json.dump(grounding_data, f, indent=2, default=str)
    print(f"\nGrounding data saved to {grounding_path}")

    # Also extract subgoal sequences for the rehearsal loop
    recorder = HumanRecorder("analysis")
    recorder._frames = frames  # Inject frames
    sequences = recorder.get_subgoal_sequences()
    if sequences:
        print(f"\nSubgoal sequences found: {len(sequences)}")
        for i, seq in enumerate(sequences[:5]):
            print(f"  Seq {i}: {len(seq.get('actions', []))} actions "
                  f"at frame {seq.get('start_frame', '?')}")
    else:
        print("\nNo subgoal sequences found (no reward events in recording)")
