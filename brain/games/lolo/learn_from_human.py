"""
learn_from_human.py — Imitation learning from FCEUX .fm2 movie recordings.

Records human expert gameplay and trains the DQN to imitate it.

Workflow:
  1. Play Lolo in FCEUX, record a .fm2 movie (File → Movie → Record)
  2. Copy the .fm2 file to throng5/recordings/
  3. Run: python3 brain/games/lolo/learn_from_human.py recordings/my_play.fm2
  4. Script replays the movie in stable-retro, reads RAM, extracts
     (compressed_state, action) pairs, and trains the DQN.

FCEUX .fm2 format:
  Header lines start with |
  Each frame line: |command|buttons_p1|buttons_p2||
  Buttons order: R L D U T S B A
  Example: |0|...U....|........|| = UP pressed
"""

from __future__ import annotations

import argparse
import gzip
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

try:
    import retro
except ImportError:
    import stable_retro as retro

# ── FM2 Parsing ──────────────────────────────────────────────────────

# FCEUX button order in .fm2: R L D U T(start) S(select) B A
FM2_BUTTONS = ['R', 'L', 'D', 'U', 'T', 'S', 'B', 'A']

# Map FM2 buttons to our stable-retro button array
# stable-retro NES: [B, _, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
# Index:             0  1    2       3     4    5     6     7     8


def parse_fm2(path: str) -> List[np.ndarray]:
    """
    Parse an FCEUX .fm2 movie file into frame-by-frame button arrays.

    Returns list of numpy arrays, one per frame, in stable-retro button format.
    """
    frames = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('|'):
                continue  # Skip header/comments

            parts = line.split('|')
            # Format: |cmd|p1_buttons|p2_buttons||
            # parts = ['', 'cmd', 'p1_buttons', 'p2_buttons', '', ...]
            if len(parts) < 3:
                continue

            p1 = parts[2] if len(parts) > 2 else '........'

            # Parse button states
            buttons = np.zeros(9, dtype=np.int8)

            if len(p1) >= 8:
                # R L D U T S B A → map to stable-retro format
                if p1[0] != '.':  # R (Right)
                    buttons[7] = 1
                if p1[1] != '.':  # L (Left)
                    buttons[6] = 1
                if p1[2] != '.':  # D (Down)
                    buttons[5] = 1
                if p1[3] != '.':  # U (Up)
                    buttons[4] = 1
                if p1[4] != '.':  # T (Start)
                    buttons[3] = 1
                if p1[5] != '.':  # S (Select)
                    buttons[2] = 1
                if p1[6] != '.':  # B
                    buttons[0] = 1
                if p1[7] != '.':  # A
                    buttons[8] = 1

            frames.append(buttons)

    return frames


def buttons_to_action(buttons: np.ndarray) -> int:
    """
    Convert stable-retro button array to our 6-action space.

    Returns:
        0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=SHOOT(A), 5=WAIT
    """
    if buttons[4]:  return 0  # UP
    if buttons[5]:  return 1  # DOWN
    if buttons[6]:  return 2  # LEFT
    if buttons[7]:  return 3  # RIGHT
    if buttons[8]:  return 4  # A = SHOOT
    return 5  # No button = WAIT


# ── Replay & Extract ────────────────────────────────────────────────

def replay_and_extract(fm2_path: str, game: str = "AdventuresOfLolo-Nes",
                       skip_frames: int = 0, sample_every: int = 4,
                       ) -> List[Tuple[np.ndarray, int]]:
    """
    Replay an FM2 movie in stable-retro and extract (state, action) pairs.

    Args:
        fm2_path: Path to .fm2 file
        game: stable-retro game name
        skip_frames: Skip this many frames at the start (menu/intro)
        sample_every: Only sample every Nth frame (reduces redundancy)

    Returns:
        List of (compressed_84dim_state, action_int) pairs
    """
    from brain.games.lolo.lolo_rom_env import LoloROMEnv, RAM_MAP
    from brain.games.lolo.lolo_compressed_state import LoloCompressedState

    encoder = LoloCompressedState()

    print(f"  Parsing FM2: {fm2_path}")
    frames = parse_fm2(fm2_path)
    print(f"  Total frames: {len(frames)}")

    # Try to load from save state, fall back to State.NONE
    try:
        env = retro.make(
            game=game,
            state="Start",
            inttype=retro.data.Integrations.STABLE,
            use_restricted_actions=retro.Actions.ALL,
        )
    except Exception:
        env = retro.make(
            game=game,
            state=retro.State.NONE,
            inttype=retro.data.Integrations.STABLE,
            use_restricted_actions=retro.Actions.ALL,
        )

    env.reset()

    pairs = []
    prev_action = 5  # WAIT
    prev_pos = (-1, -1)

    for i, buttons in enumerate(frames):
        # Step the emulator with exact button inputs
        obs, reward, terminated, truncated, info = env.step(buttons)

        if i < skip_frames:
            continue

        if i % sample_every != 0:
            continue

        # Read RAM
        ram = env.get_ram()

        # Get position and game state
        px = int(ram[RAM_MAP["player_x_grid"]])
        py = int(ram[RAM_MAP["player_y_grid"]])
        hearts = int(ram[RAM_MAP["hearts_collected"]])
        shots = int(ram[RAM_MAP["magic_shots"]])
        lives = int(ram[RAM_MAP["lives"]])

        # Skip if still in menu (position 0,0 with 0 lives)
        if lives == 0 and hearts == 0 and px == 0 and py == 0:
            continue

        # Get the action for this frame
        action = buttons_to_action(buttons)

        # Build RAM state dict for encoder
        ram_state = {
            "grid": np.zeros((13, 11), dtype=np.int32),  # Simplified
            "player_row": py,
            "player_col": px,
            "hearts_collected": hearts,
            "hearts_total": max(2, hearts),
            "chest_open": bool(ram[RAM_MAP["chest_open"]]),
            "has_jewel": False,
            "magic_shots": shots,
            "step_count": i,
            "enemies": [],
        }

        try:
            compressed = encoder.encode_from_ram(ram_state)
            pairs.append((compressed.copy(), action))
        except Exception:
            pass  # Skip frames where encoding fails

        # Progress
        if len(pairs) % 100 == 0 and len(pairs) > 0:
            print(f"    Frame {i}: pos=({py},{px}) hearts={hearts} "
                  f"shots={shots} action={action} | {len(pairs)} pairs")

    env.close()

    print(f"  Extracted {len(pairs)} (state, action) pairs")
    return pairs


# ── DQN Training ────────────────────────────────────────────────────

def train_dqn_from_demos(pairs: List[Tuple[np.ndarray, int]],
                          weights_path: str = "brain/games/lolo/dqn_weights.pt",
                          epochs: int = 50, batch_size: int = 64,
                          lr: float = 0.001):
    """Train DQN from human demonstration (state, action) pairs."""
    import torch
    import torch.nn.functional as F
    from brain.games.lolo.lolo_dqn_learner import LoloDQNLearner

    if not pairs:
        print("  No pairs to train on!")
        return

    states = np.array([p[0] for p in pairs], dtype=np.float32)
    actions = np.array([p[1] for p in pairs], dtype=np.int64)

    print(f"\n  Training DQN on {len(pairs)} demo pairs")
    print(f"    Action distribution: {np.bincount(actions, minlength=6)}")

    # Load existing DQN or create new
    dqn = LoloDQNLearner(n_actions=6)
    if os.path.exists(weights_path):
        dqn.load(weights_path)
        print(f"    Loaded existing weights from {weights_path}")

    device = dqn.device
    s_tensor = torch.tensor(states, device=device)
    a_tensor = torch.tensor(actions, device=device)

    optimizer = torch.optim.Adam(dqn.q_net.parameters(), lr=lr)

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(len(s_tensor))
        s_shuffled = s_tensor[perm]
        a_shuffled = a_tensor[perm]

        total_loss = 0.0
        batches = 0

        for start in range(0, len(s_shuffled), batch_size):
            end = min(start + batch_size, len(s_shuffled))
            s_batch = s_shuffled[start:end]
            a_batch = a_shuffled[start:end]

            # Forward pass
            q_values = dqn.q_net(s_batch)

            # Cross-entropy loss (treat as classification)
            loss = F.cross_entropy(q_values, a_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.q_net.parameters(), 10.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(batches, 1)

        # Accuracy check
        with torch.no_grad():
            pred = dqn.q_net(s_tensor).argmax(dim=1)
            acc = (pred == a_tensor).float().mean().item()

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} acc={acc:.1%}")

    # Save
    dqn.save(weights_path)
    print(f"\n  ✅ DQN saved to {weights_path}")
    print(f"     Final accuracy: {acc:.1%}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Learn from FCEUX .fm2 human gameplay recordings"
    )
    parser.add_argument("fm2", help="Path to .fm2 movie file")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Skip N frames at start (menu/intro)")
    parser.add_argument("--sample-every", type=int, default=4,
                        help="Sample every Nth frame (default: 4)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="DQN training epochs (default: 50)")
    parser.add_argument("--weights", default="brain/games/lolo/dqn_weights.pt",
                        help="DQN weights path")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract pairs, don't train")
    parser.add_argument("--save-pairs", default="",
                        help="Save extracted pairs to .npz file")
    args = parser.parse_args()

    print("=" * 60)
    print("  LOLO — Learning from Human Demonstrations")
    print("=" * 60)

    # Extract (state, action) pairs
    pairs = replay_and_extract(
        args.fm2,
        skip_frames=args.skip_frames,
        sample_every=args.sample_every,
    )

    if args.save_pairs:
        states = np.array([p[0] for p in pairs], dtype=np.float32)
        actions = np.array([p[1] for p in pairs], dtype=np.int64)
        np.savez(args.save_pairs, states=states, actions=actions)
        print(f"  Saved pairs to {args.save_pairs}")

    if not args.extract_only:
        train_dqn_from_demos(pairs, weights_path=args.weights,
                              epochs=args.epochs)


if __name__ == "__main__":
    main()
