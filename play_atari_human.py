"""
play_atari_human.py
====================
Human-play shim for ALE/Gymnasium Atari games with HumanPlayLogger integration.

Opens a live pygame window. You play with the keyboard while the
PortableNNAgent (or heuristic dummy) logs its own greedy action each step.
Disagreements (human ≠ agent) and near-death events are flagged and
written to experiments/replay_db.sqlite via HumanPlayLogger.

At episode end, compute_derived() runs the full backward pass so the
replay DB is immediately ready for PrioritizedReplayBuffer sampling.

Usage
-----
    # Play Breakout (default)
    python play_atari_human.py

    # Pong
    python play_atari_human.py --game ALE/Pong-v5

    # Play 3 episodes on level 7 then exit
    python play_atari_human.py --episodes 3

    # Disable agent voting (pure human session, no disagreement signal)
    python play_atari_human.py --no-agent

Controls — Breakout / Pong
--------------------------
    ←  /  →     LEFT / RIGHT
    SPACE        FIRE (launch ball)
    ESC  /  Q    Quit
    P            Pause / Resume
    R            Reset current episode

Controls are configurable per-game via --game flag.

Key bindings that are missing map to NOOP (action 0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# ── project imports ───────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from throng4.storage.human_play_logger import HumanPlayLogger
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig


# ─────────────────────────────────────────────────────────────────────
# Per-game key maps  (pygame key constants → ALE action index)
# ─────────────────────────────────────────────────────────────────────
#
# Key constants: pygame.K_LEFT, pygame.K_RIGHT, etc.
# We import pygame lazily inside main() so the module can be imported
# without a display (e.g. for testing).
#
# The dict below maps GAME-ID → { pygame_key_int: action_int }
# NOOP (action 0) is the fallback for any unmapped key.

def _build_key_map(game_id: str):
    """Return (keys_to_action dict, action_meanings list)."""
    import pygame

    if "Breakout" in game_id or "Pong" in game_id or "Tennis" in game_id:
        return {
            frozenset([pygame.K_RIGHT]): 2,   # RIGHT
            frozenset([pygame.K_LEFT]):  3,   # LEFT
            frozenset([pygame.K_SPACE]): 1,   # FIRE
        }
    if "SpaceInvaders" in game_id:
        return {
            frozenset([pygame.K_RIGHT]): 2,
            frozenset([pygame.K_LEFT]):  3,
            frozenset([pygame.K_SPACE]): 1,
        }
    if "MsPacman" in game_id or "Pacman" in game_id:
        return {
            frozenset([pygame.K_UP]):    2,
            frozenset([pygame.K_DOWN]):  5,
            frozenset([pygame.K_LEFT]):  3,
            frozenset([pygame.K_RIGHT]): 4,
        }
    # Generic fallback: arrow keys → RIGHT(2) LEFT(3) UP(4) DOWN(5) FIRE=SPACE(1)
    return {
        frozenset([pygame.K_RIGHT]): 2,
        frozenset([pygame.K_LEFT]):  3,
        frozenset([pygame.K_UP]):    4,
        frozenset([pygame.K_DOWN]):  5,
        frozenset([pygame.K_SPACE]): 1,
    }


# ─────────────────────────────────────────────────────────────────────
# Agent wrapper — picks agent's greedy action from RAM obs
# ─────────────────────────────────────────────────────────────────────

class _AgentVoter:
    """
    Wraps PortableNNAgent with Atari adapter logic so we can get a
    greedy action vote without stepping the env.
    """

    def __init__(self, n_features: int, n_actions: int, weights_path: Optional[str] = None):
        cfg = AgentConfig(epsilon=0.0, epsilon_min=0.0)   # greedy only
        self.agent = PortableNNAgent(n_features + n_actions, config=cfg)
        self.n_actions = n_actions
        self.n_features = n_features
        if weights_path and Path(weights_path).exists():
            self.agent.load_weights(weights_path)
            print(f"[agent] Loaded weights from {weights_path}")
        else:
            print("[agent] No weights found — agent votes randomly (untrained)")

    def vote(self, ram_obs: np.ndarray) -> int:
        """Return agent's greedy action for current RAM observation."""
        state = (np.array(ram_obs, dtype=np.float32) / 255.0)
        best_action, best_val = 0, -float("inf")
        for a in range(self.n_actions):
            ah = np.zeros(self.n_actions, dtype=np.float32)
            ah[a] = 1.0
            feat = np.concatenate([state, ah])
            val = self.agent.forward(feat)
            if val > best_val:
                best_val = val
                best_action = a
        return best_action


# ─────────────────────────────────────────────────────────────────────
# HUD rendering helpers
# ─────────────────────────────────────────────────────────────────────

def _draw_hud(
    screen,
    font,
    small_font,
    step: int,
    episode: int,
    total_reward: float,
    lives: int,
    agent_action: int,
    human_action: int,
    action_meanings: list[str],
    paused: bool,
    near_death: bool,
    hud_w: int,
):
    import pygame
    hud_surf = pygame.Surface((hud_w, screen.get_height()), pygame.SRCALPHA)
    hud_surf.fill((10, 10, 20, 220))

    def txt(s, x, y, color=(220, 220, 220), f=None):
        rendered = (f or font).render(s, True, color)
        hud_surf.blit(rendered, (x, y))

    y = 12
    dy = 22

    txt("THRONG4 PLAY", 8, y, (100, 200, 255))
    y += dy + 4
    txt(f"Ep   {episode}", 8, y);           y += dy
    txt(f"Step {step}",    8, y);           y += dy
    txt(f"Rew  {total_reward:+.1f}", 8, y); y += dy
    txt(f"Lives {lives}",  8, y,
        (255, 80, 80) if lives <= 1 else (220, 220, 220))
    y += dy + 6

    # Agent vs human
    am = action_meanings
    ag_str = am[agent_action] if agent_action < len(am) else str(agent_action)
    hu_str = am[human_action] if human_action < len(am) else str(human_action)
    agree  = agent_action == human_action
    color  = (100, 220, 100) if agree else (255, 180, 50)

    txt("Agent:",  8, y, (160, 160, 160), small_font); y += 16
    txt(ag_str,    8, y, (150, 200, 255), small_font); y += 18
    txt("You:",    8, y, (160, 160, 160), small_font); y += 16
    txt(hu_str,    8, y, color,           small_font); y += 18

    if not agree:
        txt("DISAGREE ★", 8, y, (255, 180, 50), small_font)
    y += 18

    if near_death:
        txt("⚠ NEAR DEATH", 8, y, (255, 60, 60), small_font)
        y += 18

    if paused:
        txt("⏸ PAUSED", 8, y, (255, 220, 50))

    # Controls reminder at bottom
    controls = [
        "←/→  move",
        "SPC  fire",
        "P    pause",
        "R    reset",
        "Q    quit",
    ]
    cy = screen.get_height() - len(controls) * 16 - 8
    for c in controls:
        txt(c, 4, cy, (100, 100, 130), small_font)
        cy += 16

    screen.blit(hud_surf, (screen.get_width() - hud_w, 0))


# ─────────────────────────────────────────────────────────────────────
# Main play loop
# ─────────────────────────────────────────────────────────────────────

def play(
    game_id: str = "ALE/Breakout-v5",
    n_episodes: int = 999,
    fps: int = 30,
    scale: int = 4,
    use_agent: bool = True,
    weights_path: Optional[str] = None,
    near_death_threshold: float = -1.0,
    seed: Optional[int] = None,
    action_hold_frames: int = 4,
):
    """
    action_hold_frames: directional keys (LEFT/RIGHT) only fire every N frames
    while held.  Frames in between send NOOP so the paddle drifts more slowly.
    FIRE (SPACE) always fires immediately every frame — it's a tap action.
    Try --action-hold 2 for faster, 6 for very slow.
    """
    import pygame

    # ── env setup ────────────────────────────────────────────────────
    # rgb_array for rendering; ram obs for agent (separate env)
    env_rgb = gym.make(game_id, obs_type="rgb",  render_mode="rgb_array")
    env_ram = gym.make(game_id, obs_type="ram",  render_mode=None)

    n_actions = env_rgb.action_space.n
    action_meanings = env_rgb.unwrapped.get_action_meanings()

    voter: Optional[_AgentVoter] = None
    if use_agent:
        voter = _AgentVoter(
            n_features=128,
            n_actions=n_actions,
            weights_path=weights_path,
        )

    # ── pygame window ────────────────────────────────────────────────
    pygame.init()
    HUD_W = 140
    GFX_W, GFX_H = 160 * scale, 210 * scale
    WIN_W = GFX_W + HUD_W
    screen = pygame.display.set_mode((WIN_W, GFX_H))
    pygame.display.set_caption(f"throng4 human play — {game_id}")
    clock = pygame.time.Clock()
    font       = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)

    key_map = _build_key_map(game_id)

    # ── logger ───────────────────────────────────────────────────────
    logger = HumanPlayLogger()
    session_id = logger.open_session(
        env_name=game_id,
        source="human_play",
        config={"fps": fps, "scale": scale, "use_agent": use_agent},
    )
    print(f"[logger] session_id = {session_id}")
    print(f"[logger] DB → {logger.db_path}")

    eps_played = 0
    running = True

    while running and eps_played < n_episodes:
        # ── episode reset ─────────────────────────────────────────
        ep_seed = (seed + eps_played) if seed is not None else None
        rgb_obs, _  = env_rgb.reset(seed=ep_seed)
        ram_obs, _  = env_ram.reset(seed=ep_seed)

        episode_id = logger.open_episode(session_id, eps_played, seed=ep_seed)

        step_idx     = 0
        total_reward = 0.0
        paused       = False
        episode_done = False
        near_death   = False

        pressed_keys: set[int] = set()
        human_action = 0   # NOOP until first keypress
        _hold_counter = 0  # frame counter for action throttle

        print(f"\n[ep {eps_played}] Starting — press SPACE to fire")

        while not episode_done and running:
            clock.tick(fps)
            agent_action = voter.vote(ram_obs) if voter else 0

            # ── event handling ────────────────────────────────────
            reset_ep = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False; break
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False; break
                    if event.key == pygame.K_p:
                        paused = not paused
                    if event.key == pygame.K_r:
                        reset_ep = True; break
                    pressed_keys.add(event.key)
                if event.type == pygame.KEYUP:
                    pressed_keys.discard(event.key)

            if reset_ep:
                # abort this episode (don't log, don't compute derived)
                print(f"[ep {eps_played}] Reset by user")
                break

            if paused:
                # Still render but don't step
                _render(screen, rgb_obs, scale, GFX_W, GFX_H)
                _draw_hud(screen, font, small_font,
                          step_idx, eps_played, total_reward,
                          _lives(ram_obs, game_id),
                          agent_action, human_action, action_meanings,
                          paused=True, near_death=near_death, hud_w=HUD_W)
                pygame.display.flip()
                continue

            # Resolve human action from held keys, with throttle
            # FIRE (action 1 / SPACE) is always instant.
            # Directional actions only fire every action_hold_frames frames.
            raw_action = 0   # NOOP
            fs = frozenset(pressed_keys)
            for combo, act in key_map.items():
                if combo.issubset(fs):
                    raw_action = act
                    break

            if raw_action == 1:          # FIRE — always immediate
                human_action = raw_action
                _hold_counter = 0
            elif raw_action != 0:        # directional — throttle
                _hold_counter += 1
                if _hold_counter >= action_hold_frames:
                    human_action = raw_action
                    _hold_counter = 0
                else:
                    human_action = 0     # NOOP this frame
            else:
                human_action = 0
                _hold_counter = 0

            # Execute human action in both envs (keep them in sync)
            rgb_obs, reward, term, trunc, info = env_rgb.step(human_action)
            ram_obs, _,      _,    _,     _    = env_ram.step(human_action)

            done      = term or trunc
            lives_now = _lives(ram_obs, game_id)
            near_death = (reward < near_death_threshold)
            total_reward += reward

            # Abstract feature vector from RAM for logging
            state_vec = list(np.array(ram_obs, dtype=np.float32) / 255.0)

            logger.log_step(
                session_id=session_id,
                episode_id=episode_id,
                step_idx=step_idx,
                state_vec=state_vec,
                executed_action=human_action,
                action_source="human",
                action_space_n=n_actions,
                reward=float(reward),
                done=done,
                human_action=human_action,
                agent_action=agent_action,
                lives=lives_now,
                score=float(info.get("score", 0)),
            )

            step_idx += 1

            # ── render ────────────────────────────────────────────
            _render(screen, rgb_obs, scale, GFX_W, GFX_H)
            _draw_hud(screen, font, small_font,
                      step_idx, eps_played, total_reward,
                      lives_now,
                      agent_action, human_action, action_meanings,
                      paused=False, near_death=near_death, hud_w=HUD_W)
            pygame.display.flip()

            if done:
                episode_done = True

        # ── episode close + derived ───────────────────────────────
        if step_idx > 0:
            logger.close_episode(
                episode_id,
                total_reward=total_reward,
                total_steps=step_idx,
                final_score=total_reward,
                terminated=True,
            )
            n_updated = logger.compute_derived(episode_id, near_death_threshold=near_death_threshold)
            print(f"[ep {eps_played}] Done — reward={total_reward:.1f}  steps={step_idx}  "
                  f"derived={n_updated} rows")

        eps_played += 1

    # ── cleanup ───────────────────────────────────────────────────────
    logger.close()
    env_rgb.close()
    env_ram.close()
    pygame.quit()
    print(f"\n[done] {eps_played} episode(s) logged to {logger.db_path}")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _render(screen, rgb_obs, scale, gfx_w, gfx_h):
    """Blit scaled RGB observation to the left portion of the window."""
    import pygame
    surf = pygame.surfarray.make_surface(
        np.transpose(rgb_obs, (1, 0, 2))   # H×W×C → W×H×C
    )
    surf = pygame.transform.scale(surf, (gfx_w, gfx_h))
    screen.blit(surf, (0, 0))


def _lives(ram_obs, game_id: str) -> int:
    """Extract lives from RAM — game-specific."""
    if "Breakout" in game_id:
        return int(ram_obs[57])
    if "Pong" in game_id:
        return 0   # Pong has no lives concept
    return 0


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Human play recorder for Atari ALE games",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--game",      default="ALE/Breakout-v5",
                   help="Gymnasium game ID")
    p.add_argument("--episodes",  type=int, default=999,
                   help="Max episodes before auto-quit")
    p.add_argument("--fps",       type=int, default=30,
                   help="Target frame rate")
    p.add_argument("--scale",     type=int, default=4,
                   help="Window scale factor (4=640x840, 5=800x1050, 6=960x1260)")
    p.add_argument("--action-hold", type=int, default=4, dest="action_hold",
                   help=("Frames a direction key must be held before firing again. "
                         "Higher = slower/less sensitive paddle. 1=every frame (original), "
                         "4=default, 8=very slow"))
    p.add_argument("--seed",      type=int, default=None,
                   help="RNG seed")
    p.add_argument("--weights",   default=None,
                   help="Path to agent .npz weights file")
    p.add_argument("--no-agent",  action="store_true",
                   help="Disable agent voting (no disagreement signal)")
    p.add_argument("--near-death-threshold", type=float, default=-1.0,
                   help="Reward threshold for near_death_flag")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    play(
        game_id=args.game,
        n_episodes=args.episodes,
        fps=args.fps,
        scale=args.scale,
        use_agent=not args.no_agent,
        weights_path=args.weights,
        near_death_threshold=args.near_death_threshold,
        seed=args.seed,
        action_hold_frames=args.action_hold,
    )
