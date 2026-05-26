"""Montezuma runner modes."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from brain.environments.human_recorder import HumanRecorder
from brain.games.montezuma.runner.constants import (
    N_ACTIONS, RESULTS_DIR, RECORDINGS_DIR, ensure_results_dirs,
    ACTION_NAMES,
)
from brain.games.montezuma.runner.setup import get_game_state, make_env

def mode_human(args):
    """
    Human play mode.

    Renders the game window. You play with keyboard (via pygame).
    Every frame, RAM + action + reward are recorded to JSONL.
    After you quit, the recording is analyzed to discover RAM semantics.

    Controls:
      Arrow keys = move (UP/DOWN/LEFT/RIGHT)
      Space      = FIRE (jump)
      Q          = quit and save recording
    """
    print("=" * 60)
    print("MODE: HUMAN PLAY")
    print("=" * 60)
    print()
    print("Controls:")
    print("  Arrow keys = move")
    print("  Space      = jump/fire")
    print("  Q          = quit and save")
    print()

    # Try importing pygame for keyboard input
    try:
        import pygame
        has_pygame = True
    except ImportError:
        has_pygame = False
        print("WARNING: pygame not installed. Using random actions.")
        print("Install with: pip install pygame")
        print()

    env = make_env(render=True, frameskip=args.frameskip)
    recorder = HumanRecorder("human_play")

    session_name = f"human_{int(time.time())}"
    total_frames = 0
    total_reward = 0.0
    best_room = 0
    episode = 0

    if has_pygame:
        pygame.init()
        # We need a tiny window to capture keyboard events
        # (the ALE render window doesn't capture keys)
        key_surface = pygame.display.set_mode((200, 100))
        pygame.display.set_caption("Montezuma Keys - Press Q to quit")
        font = pygame.font.SysFont("monospace", 14)

    try:
        for ep in range(args.episodes):
            episode = ep
            obs, info = env.reset()
            ram = env.unwrapped.ale.getRAM()
            recorder.start(None)

            ep_reward = 0.0
            ep_steps = 0
            done = False

            while not done and ep_steps < args.max_steps:
                # Get keyboard action
                action = 0  # NOOP
                quit_requested = False

                if has_pygame:
                    pygame.event.pump()
                    keys = pygame.key.get_pressed()

                    if keys[pygame.K_q]:
                        quit_requested = True
                    elif keys[pygame.K_SPACE] and keys[pygame.K_UP]:
                        action = 10   # UPFIRE
                    elif keys[pygame.K_SPACE] and keys[pygame.K_RIGHT]:
                        action = 11   # RIGHTFIRE
                    elif keys[pygame.K_SPACE] and keys[pygame.K_LEFT]:
                        action = 12   # LEFTFIRE
                    elif keys[pygame.K_SPACE] and keys[pygame.K_DOWN]:
                        action = 13   # DOWNFIRE
                    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                        action = 6    # UPRIGHT
                    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                        action = 7    # UPLEFT
                    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                        action = 8    # DOWNRIGHT
                    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                        action = 9    # DOWNLEFT
                    elif keys[pygame.K_SPACE]:
                        action = 1    # FIRE
                    elif keys[pygame.K_UP]:
                        action = 2    # UP
                    elif keys[pygame.K_RIGHT]:
                        action = 3    # RIGHT
                    elif keys[pygame.K_LEFT]:
                        action = 4    # LEFT
                    elif keys[pygame.K_DOWN]:
                        action = 5    # DOWN

                    # Update key window
                    game_state = get_game_state(ram)
                    key_surface.fill((0, 0, 0))
                    lines = [
                        f"Rm:{game_state['room']} "
                        f"({game_state['player_x']},{game_state['player_y']})",
                        f"R={ep_reward:.0f} "
                        f"Act={ACTION_NAMES[action]:10s}",
                        f"Lives={game_state['lives']} "
                        f"Items={game_state['items_mask']:08b}",
                    ]
                    for i, line in enumerate(lines):
                        surf = font.render(line, True, (0, 255, 0))
                        key_surface.blit(surf, (5, 5 + i * 20))
                    pygame.display.flip()

                    if quit_requested:
                        print("\nQuit requested. Saving recording...")
                        break
                else:
                    # No pygame — use simple random actions for testing
                    action = np.random.randint(N_ACTIONS)

                # Record
                ram = env.unwrapped.ale.getRAM()
                recorder.record(ram, action, ep_reward, done)

                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_steps += 1
                total_frames += 1
                ram = env.unwrapped.ale.getRAM()

                game_state = get_game_state(ram)
                best_room = max(best_room, game_state["room"])

                # Slow down to human speed
                time.sleep(0.016)  # ~60fps

            recorder.stop()
            total_reward += ep_reward
            print(f"  Episode {ep}: reward={ep_reward:.0f}, "
                  f"room={game_state['room']}, steps={ep_steps}")

            if quit_requested:
                break

    except KeyboardInterrupt:
        print("\nInterrupted. Saving recording...")

    # Save recording
    ensure_results_dirs()
    rec_path = recorder.save(str(RECORDINGS_DIR))
    print(f"\nRecording saved to {rec_path}")
    print(f"  Total frames: {total_frames}")
    print(f"  Total reward: {total_reward:.0f}")
    print(f"  Best room: {best_room}")

    # Analyze
    analysis = recorder.analyze()
    print(f"\nRAM Analysis:")
    print(f"  Position candidates: {len(analysis.get('position_candidates', []))}")
    print(f"  State flags: {len(analysis.get('state_flag_candidates', []))}")
    print(f"  Reward events: {analysis.get('reward_events', 0)}")

    # Save analysis
    analysis_path = RECORDINGS_DIR / f"{session_name}_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"  Analysis saved to {analysis_path}")

    if has_pygame:
        pygame.quit()
    env.close()
