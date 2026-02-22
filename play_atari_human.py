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
from throng4.storage.atari_event import AtariEventLogger, update_brief
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig


# ─────────────────────────────────────────────────────────────────────
# DAS (Delayed Auto Shift) tracker
# ─────────────────────────────────────────────────────────────────────
# Implements the timing system used in every modern Tetris game:
#   Frame 0 (first press)  → fire immediately
#   Frames 1 .. das_delay-1 → silent (charge)
#   Frame das_delay+         → fire, then every das_repeat frames
#
# Usage: one _DASTracker instance per action category.
#   tracker.update(active_action_this_frame) → action to actually send

class _DASTracker:
    def __init__(self, das_delay: int, das_repeat: int):
        self.das_delay  = max(1, das_delay)
        self.das_repeat = max(1, das_repeat)
        self._action: int = 0
        self._held:   int = 0   # frames the current action has been held

    def update(self, raw: int) -> int:
        """
        raw: action selected this frame (0 = NOOP).
        Returns: action to actually send to env (0 = NOOP this frame).
        """
        if raw == 0:
            self._action = 0
            self._held   = 0
            return 0

        if raw != self._action:
            # New action — fire immediately and start charge
            self._action = raw
            self._held   = 1
            return raw

        # Same action still held
        self._held += 1
        net = self._held - self.das_delay
        if net <= 0:
            return 0   # still charging
        if net % self.das_repeat == 0:
            return raw  # auto-repeat fires
        return 0

    def reset(self) -> None:
        self._action = 0
        self._held   = 0


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

# ─────────────────────────────────────────────────────────────────────
# Universal key map — auto-built from ALE action meanings
# ─────────────────────────────────────────────────────────────────────
#
# ALE action names are composed tokens:
#   Directions : UP  DOWN  LEFT  RIGHT
#   Diagonal   : UPRIGHT  UPLEFT  DOWNRIGHT  DOWNLEFT
#   With fire  : FIRE  UPFIRE  RIGHTFIRE ...  UPRIGHTFIRE ...
#
# Key layout:
#   UP → K_UP  |  DOWN → K_DOWN  |  LEFT → K_LEFT
#   RIGHT → K_RIGHT  |  FIRE → K_SPACE
#
# Combos are sorted longest-first so when multiple keys are held,
# the most specific combo (e.g. UPRIGHTFIRE) wins over subsets.

def _build_key_map_from_meanings(action_meanings: list):
    """
    Auto-build {frozenset_of_keys: action_idx} from ALE action name list.
    """
    import pygame

    _TOKEN_KEYS = {
        "UP":    {pygame.K_UP},
        "DOWN":  {pygame.K_DOWN},
        "LEFT":  {pygame.K_LEFT},
        "RIGHT": {pygame.K_RIGHT},
        "FIRE":  {pygame.K_SPACE},
        # Diagonal tokens decompose into two keys
        "UPRIGHT":   {pygame.K_UP,   pygame.K_RIGHT},
        "UPLEFT":    {pygame.K_UP,   pygame.K_LEFT},
        "DOWNRIGHT": {pygame.K_DOWN, pygame.K_RIGHT},
        "DOWNLEFT":  {pygame.K_DOWN, pygame.K_LEFT},
    }
    # Longest tokens first so greedy parse is correct
    _PARSE_ORDER = sorted(_TOKEN_KEYS, key=len, reverse=True)

    entries: list = []
    for idx, name in enumerate(action_meanings):
        if name == "NOOP":
            continue
        keys: set = set()
        rest = name
        for token in _PARSE_ORDER:
            if token in rest:
                keys |= _TOKEN_KEYS[token]
                rest = rest.replace(token, "", 1)
        if keys:
            entries.append((frozenset(keys), idx))

    # Longest combo first — UPRIGHTFIRE wins over UPRIGHT when all keys held
    entries.sort(key=lambda t: -len(t[0]))
    return dict(entries)


def _build_key_map(game_id: str, env=None):
    """
    Build key map for the given game.
    If env is supplied its action meanings drive the map (always preferred).
    """
    if env is not None:
        return _build_key_map_from_meanings(env.unwrapped.get_action_meanings())
    # Fallback — only hit if called without env (shouldn't happen in play())
    import pygame
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
        q = self.q_values_all(ram_obs)
        return int(np.argmax(q))

    def q_values_all(self, ram_obs: np.ndarray) -> np.ndarray:
        """Return Q(s,a) for every action — used by AtariEventLogger."""
        state = (np.array(ram_obs, dtype=np.float32) / 255.0)
        q = np.empty(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            ah = np.zeros(self.n_actions, dtype=np.float32)
            ah[a] = 1.0
            feat = np.concatenate([state, ah])
            q[a] = self.agent.forward(feat)
        return q


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
    scale_x: int = 5,
    scale_y: int = 4,
    use_agent: bool = True,
    weights_path: Optional[str] = None,
    near_death_threshold: float = -1.0,
    seed: Optional[int] = None,
    action_hold_frames: int = 4,   # kept for backwards compat
    das_delay:   int = 10,
    das_repeat:  int = 2,
    fire_delay:  int = 2,   # 2 frames ≈ 67ms — prevents tap double-fire, allows rapid hold
    fire_repeat: int = 1,   # every frame once charging done — original rapid-fire feel
    drop_repeat: int = 1,
):
    """
    scale_x / scale_y: independent horizontal and vertical scale factors.
    Native ALE frame is 160x210 (portrait).  Default 5x / 4y gives 800x840
    (roughly square), which is more comfortable.

    DAS controls (Delayed Auto Shift — standard Tetris timing system)
    -----------------------------------------------------------------
    das_delay   : frames before directional auto-repeat kicks in
                  (default 10 ≈ 333ms at 30fps  — feels like a real D-pad charge)
    das_repeat  : frames between auto-repeat moves after DAS kicks in
                  (default 2  ≈  67ms — rapid slide once DAS charged)
    fire_delay  : frames before auto-repeat rotation kicks in
                  (default 8  ≈ 267ms — prevents double-rotate, allows rapid spin)
    fire_repeat : frames between auto-repeat rotations
                  (default 4  ≈ 133ms — comfortable multi-rotate speed)
    drop_repeat : frames between soft-drop repeats (1 = every frame, very fast)
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
    HUD_W = 160
    # Native ALE frame: 160(w) x 210(h)
    GFX_W = 160 * scale_x
    GFX_H = 210 * scale_y
    WIN_W = GFX_W + HUD_W
    screen = pygame.display.set_mode((WIN_W, GFX_H))
    pygame.display.set_caption(f"throng4 human play — {game_id}  [{GFX_W}x{GFX_H}]")
    clock = pygame.time.Clock()
    font       = pygame.font.SysFont("monospace", 14, bold=True)
    small_font = pygame.font.SysFont("monospace", 12)

    key_map = _build_key_map(game_id, env=env_rgb)
    # Actions that contain FIRE
    fire_actions = {i for i, m in enumerate(action_meanings) if "FIRE" in m}
    # Pure-drop action (DOWN with no FIRE/direction combo) — gets rapid repeat
    drop_actions = {i for i, m in enumerate(action_meanings)
                    if m == "DOWN"}

    # ── loggers ───────────────────────────────────────────────────────
    logger = HumanPlayLogger()
    session_id = logger.open_session(
        env_name=game_id,
        source="human_play",
        config={"fps": fps, "scale_x": scale_x, "scale_y": scale_y,
                    "use_agent": use_agent},
    )
    event_logger = AtariEventLogger(game_id, action_meanings, session_id)
    print(f"[logger] session_id = {session_id}")
    print(f"[logger] DB  → {logger.db_path}")
    print(f"[logger] EVT → {event_logger.path}")

    eps_played = 0
    running = True

    while running and eps_played < n_episodes:
        # ── episode reset ─────────────────────────────────────────
        ep_seed = (seed + eps_played) if seed is not None else None
        rgb_obs, _  = env_rgb.reset(seed=ep_seed)
        ram_obs, _  = env_ram.reset(seed=ep_seed)

        episode_id = logger.open_episode(session_id, eps_played, seed=ep_seed)
        event_logger.begin_episode(eps_played)

        step_idx     = 0
        total_reward = 0.0
        paused       = False
        episode_done = False
        near_death   = False

        pressed_keys: set[int] = set()
        just_pressed: set[int] = set()  # KEYDOWN this frame
        human_action = 0

        # One DAS tracker per timing category, reset each episode
        dir_das  = _DASTracker(das_delay,  das_repeat)   # LEFT / RIGHT / UP
        fire_das = _DASTracker(fire_delay, fire_repeat)  # FIRE / ROTATE combos
        drop_das = _DASTracker(1,          drop_repeat)  # DOWN soft-drop

        # Save state slots: in-memory AND on disk
        # Disk path: experiments/save_states/<game_slug>/<session_id>_slot<N>.bin
        _save_states_dir = (_ROOT / "experiments" / "save_states" /
                            game_id.replace("/", "_").replace("-", "_"))
        _save_states_dir.mkdir(parents=True, exist_ok=True)
        _saved_state_rgb = None
        _saved_state_ram = None
        _save_slot = 0   # increments each F5 press so history is kept

        # RAM reward log — records full RAM when reward fires (for byte calibration)
        _reward_ram_log = (_save_states_dir.parent / "reward_ram_log" /
                           game_id.replace("/", "_").replace("-", "_"))
        _reward_ram_log.mkdir(parents=True, exist_ok=True)
        _reward_ram_fh = open(
            _reward_ram_log / f"{session_id}_rewards.jsonl", "w", encoding="utf-8"
        )
        _prev_ram = None   # RAM from previous step (for before/after diff)

        print(f"\n[ep {eps_played}] Starting — press SPACE to fire")

        while not episode_done and running:
            clock.tick(fps)
            agent_action = voter.vote(ram_obs) if voter else 0

            # ── event handling ────────────────────────────────────
            reset_ep = False
            just_pressed.clear()   # reset tap-set each frame
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
                    # ── Save / Load state (disk-persistent) ────────
                    if event.key == pygame.K_F5:
                        _saved_state_rgb = env_rgb.unwrapped.ale.cloneState()
                        _saved_state_ram = env_ram.unwrapped.ale.cloneState()
                        # Write to disk so state survives quit
                        slot_path = _save_states_dir / f"{session_id}_slot{_save_slot:03d}_step{step_idx}.bin"
                        import pickle
                        with open(slot_path, 'wb') as sf:
                            pickle.dump({
                                'rgb': _saved_state_rgb,
                                'ram': _saved_state_ram,
                                'step': step_idx,
                                'reward': total_reward,
                                'episode': eps_played,
                            }, sf)
                        _save_slot += 1
                        print(f"  [F5] State saved to disk -> {slot_path.name}")
                        continue
                    if event.key == pygame.K_F9:
                        # Try in-memory first, then disk (most recent slot file)
                        if _saved_state_rgb is None:
                            slot_files = sorted(_save_states_dir.glob(f"*_slot*.bin"))
                            if slot_files:
                                import pickle
                                with open(slot_files[-1], 'rb') as sf:
                                    saved = pickle.load(sf)
                                _saved_state_rgb = saved['rgb']
                                _saved_state_ram = saved['ram']
                                print(f"  [F9] Loaded from disk: {slot_files[-1].name}")
                        if _saved_state_rgb is not None:
                            env_rgb.unwrapped.ale.restoreState(_saved_state_rgb)
                            env_ram.unwrapped.ale.restoreState(_saved_state_ram)
                            rgb_obs = env_rgb.unwrapped.ale.getScreenRGB()
                            ram_obs = np.array(env_ram.unwrapped.ale.getRAM(),
                                               dtype=np.uint8)
                            print(f"  [F9] State restored (step {step_idx}  reward {total_reward:.0f})")
                        else:
                            print("  [F9] No save state found (in memory or disk)")
                        continue
                    # ──────────────────────────────────────────────
                    pressed_keys.add(event.key)
                    just_pressed.add(event.key)  # mark as tapped this frame
                if event.type == pygame.KEYUP:
                    pressed_keys.discard(event.key)

            if reset_ep:
                # abort this episode (don't log, don't compute derived)
                print(f"[ep {eps_played}] Reset by user")
                break

            if paused:
                # Still render but don't step
                _render(screen, rgb_obs, scale_x, scale_y, GFX_W, GFX_H)
                _draw_hud(screen, font, small_font,
                          step_idx, eps_played, total_reward,
                          _lives(ram_obs, game_id),
                          agent_action, human_action, action_meanings,
                          paused=True, near_death=near_death, hud_w=HUD_W)
                pygame.display.flip()
                continue

            # ── Resolve human action via DAS ──────────────────────────
            #
            # Priority order (highest first):
            #   1. FIRE-containing combos  → fire_das (debounce, then auto-repeat)
            #   2. Pure DOWN soft-drop     → drop_das (fast, every drop_repeat frames)
            #   3. Other directionals      → dir_das  (DAS charge then rapid slide)
            #
            # Special: if UP arrow pressed on a fire-less game (e.g. Tetris has no
            # UP action), treat UP as a hard-drop impulse: flood 20 soft-downs.

            # -- Find the highest-priority active combo --
            raw_fire = 0
            raw_dir  = 0
            fs = frozenset(pressed_keys)

            for combo, act in key_map.items():
                if act in fire_actions:
                    if combo.issubset(fs):
                        raw_fire = act
                        break
            for combo, act in key_map.items():
                if act not in fire_actions:
                    if combo.issubset(fs):
                        raw_dir = act
                        break

            # If fire combo active, suppress dir tracker (and vice-versa)
            if raw_fire:
                dir_das.reset()
                drop_das.reset()
                human_action = fire_das.update(raw_fire)
            elif raw_dir in drop_actions:
                fire_das.reset()
                dir_das.reset()
                human_action = drop_das.update(raw_dir)
            elif raw_dir:
                fire_das.reset()
                drop_das.reset()
                human_action = dir_das.update(raw_dir)
            else:
                fire_das.reset()
                dir_das.reset()
                drop_das.reset()
                human_action = 0

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

            # ── Canonical event log (for Tetra) ───────────────────
            q_vals = voter.q_values_all(ram_obs) if voter else None
            event_logger.log_step(
                step=step_idx,
                human_action_idx=human_action,
                q_values=q_vals,
                reward=float(reward),
                done=done,
                near_death=near_death,
            )

            # ── RAM reward snapshot (key byte calibration) ────────
            if reward != 0 and _prev_ram is not None:
                import json as _json
                snapshot = {
                    "step": step_idx, "reward": float(reward),
                    "action": int(human_action),
                    "ram_before": list(int(x) for x in _prev_ram),
                    "ram_after":  list(int(x) for x in ram_obs),
                    "changed_bytes": [
                        {"idx": i, "before": int(_prev_ram[i]), "after": int(ram_obs[i])}
                        for i in range(128) if _prev_ram[i] != ram_obs[i]
                    ],
                }
                _reward_ram_fh.write(_json.dumps(snapshot) + "\n")
                _reward_ram_fh.flush()
                print(f"  [RAM] reward={reward:+.0f} at step {step_idx}  "
                      f"{len(snapshot['changed_bytes'])} bytes changed")
            _prev_ram = ram_obs.copy()

            step_idx += 1

            # ── render ────────────────────────────────────────────
            _render(screen, rgb_obs, scale_x, scale_y, GFX_W, GFX_H)
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

    # -- cleanup -----------------------------------------------------------
    logger.close()
    _reward_ram_fh.close()
    evt_path = event_logger.end_session()
    brief_path = update_brief(game_id)
    env_rgb.close()
    env_ram.close()
    pygame.quit()
    print(f"\n[done] {eps_played} episode(s) logged to {logger.db_path}")
    print(f"[done] events  -> {evt_path}")
    print(f"[done] brief   -> {brief_path}  (updated for Tetra)")
    print(f"[done] RAM log -> {_reward_ram_fh.name}  (reward byte calibration)")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _render(screen, rgb_obs, scale_x, scale_y, gfx_w, gfx_h):
    """
    Blit the ALE RGB frame to the game area using smoothscale.
    smoothscale uses bilinear interpolation so small sprites
    (projectiles, bullets) don't alias to invisible pixels.
    """
    import pygame
    surf = pygame.surfarray.make_surface(
        np.transpose(rgb_obs, (1, 0, 2))   # H x W x C -> W x H x C
    )
    surf = pygame.transform.smoothscale(surf, (gfx_w, gfx_h))
    screen.blit(surf, (0, 0))


def _lives(ram_obs, game_id: str) -> int:
    """Extract lives from RAM — game-specific."""
    if "Breakout" in game_id:
        return int(ram_obs[57])
    if "SpaceInvaders" in game_id:
        return int(ram_obs[73])   # SI stores lives at byte 73
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
    p.add_argument("--game",      default="ALE/Breakout-v5", help="Gymnasium game ID")
    p.add_argument("--episodes",  type=int, default=999,    help="Max episodes")
    p.add_argument("--fps",       type=int, default=30,     help="Target frame rate")
    p.add_argument("--scale-x",   type=int, default=5,   dest="scale_x",
                   help="Horizontal scale (native=160px, 5=800px)")
    p.add_argument("--scale-y",   type=int, default=4,   dest="scale_y",
                   help="Vertical scale (native=210px, 4=840px)")
    # DAS timing
    p.add_argument("--das-delay",   type=int, default=10, dest="das_delay",
                   help="Directional: frames before auto-repeat kicks in (10≈333ms)")
    p.add_argument("--das-repeat",  type=int, default=2,  dest="das_repeat",
                   help="Directional: frames between auto-repeat moves (2≈67ms)")
    p.add_argument("--fire-delay",  type=int, default=2,  dest="fire_delay",
                   help="Rotate/fire: frames before auto-repeat (2≈67ms, default=rapid; Tetris=8)")
    p.add_argument("--fire-repeat", type=int, default=1,  dest="fire_repeat",
                   help="Rotate/fire: frames between auto-repeat (1=every frame; Tetris=4)")
    p.add_argument("--drop-repeat", type=int, default=1,  dest="drop_repeat",
                   help="Soft-drop: frames between repeats (1=every frame)")
    # Misc
    p.add_argument("--seed",      type=int,   default=None)
    p.add_argument("--weights",   default=None, help="Agent .npz weights path")
    p.add_argument("--no-agent",  action="store_true", help="Disable agent voting")
    p.add_argument("--near-death-threshold", type=float, default=-1.0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    play(
        game_id=args.game,
        n_episodes=args.episodes,
        fps=args.fps,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        use_agent=not args.no_agent,
        weights_path=args.weights,
        near_death_threshold=args.near_death_threshold,
        seed=args.seed,
        das_delay=args.das_delay,
        das_repeat=args.das_repeat,
        fire_delay=args.fire_delay,
        fire_repeat=args.fire_repeat,
        drop_repeat=args.drop_repeat,
    )
