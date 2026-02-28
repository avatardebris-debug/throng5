"""
montezuma_explorer.py
=====================
Shaped-reward training loop for Montezuma's Revenge.

Uses:
  - ShapedRewardEngine (subgoal bonuses + count-based novelty)
  - Trajectory survivor (best-room trajectories replayed preferentially)
  - Determinism exploit: after finding a trajectory that reaches room N,
    immediately oversample those transitions ("amygdala switch")

Replaces or wraps the standard benchmark_human.py loop for Montezuma.

Usage
-----
    # Run training with shaped rewards:
    python montezuma_explorer.py --episodes 500

    # Start from BC-pretrained weights:
    python montezuma_explorer.py --weights benchmark_results/bc_ALE_MontezumaRevenge_v5.npz

    # Continue from previous explorer run:
    python montezuma_explorer.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.learning.prioritized_replay import PrioritizedReplayBuffer
from throng4.storage.ram_decoders import get_decoder
from montezuma_subgoals import ShapedRewardEngine, STAGE_1_SUBGOALS
from montezuma_frontier import FrontierManager
try:
    from throng4.metastack_pipeline import MetaStackPipeline
    _METASTACK_AVAILABLE = True
except ImportError:
    MetaStackPipeline = None     # type: ignore
    _METASTACK_AVAILABLE = False
try:
    from montezuma_bandit import PolicyBandit
    _BANDIT_AVAILABLE = True
except ImportError:
    PolicyBandit = None          # type: ignore
    _BANDIT_AVAILABLE = False

# throng4 basal-ganglia components (imported lazily so they don't break
# existing runs if the path is missing)
try:
    from throng4.basal_ganglia.dreamer_engine import DreamerEngine, Hypothesis
    from throng4.basal_ganglia.hypothesis_profiler import (
        DreamerTeacher, OptionsLibrary, HypothesisProfile,
    )
    from throng4.basal_ganglia.compressed_state import (
        CompressedStateEncoder, EncodingMode,
    )
    _THRONG4_AVAILABLE = True
except ImportError as _e:
    _THRONG4_AVAILABLE = False
    print(f"[warn] throng4 basal_ganglia not importable ({_e}); --use-dreamer disabled")

try:
    from throng4.meta_policy.meta_adapter import MetaAdapter as _MetaAdapter
    _METAADAPTER_AVAILABLE = True
except ImportError:
    _MetaAdapter = None
    _METAADAPTER_AVAILABLE = False

_GAME_ID     = "ALE/MontezumaRevenge-v5"
_RESULTS_DIR = _ROOT / "benchmark_results"
_EXP_DIR     = _ROOT / "experiments"
_TRAJ_DIR    = _EXP_DIR / "montezuma_trajectories"
_RESULTS_DIR.mkdir(exist_ok=True)
_TRAJ_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Trajectory survivor - stores best-room transitions for replay boost
# ----------------------------------------------------------------------

class TrajectorySurvivor:
    """
    Keep transitions from episodes that reached the highest room so far.
    When a new best-room trajectory arrives, the amygdala switches:
      - All prior survivor transitions are dropped
      - New survivor transitions are seeded into the replay buffer at 10x priority

    Determinism means the same actions from the same seed reproduce the
    same trajectory - so we can cheaply re-inject the winner.
    """

    def __init__(self, priority_scale: float = 10.0) -> None:
        self._best_room    = 0
        self._best_ep_data: list[dict] = []   # [{feat, reward, next_feats, done}]
        self._priority     = priority_scale
        self.switch_count  = 0

    def record_trajectory(
        self,
        transitions: list[dict],
        max_room_reached: int,
    ) -> bool:
        """
        If max_room_reached > current best, switch to this trajectory.
        Returns True if this was a new best (amygdala switched).
        """
        if max_room_reached > self._best_room:
            print(f"  [amygdala] NEW BEST: room {self._best_room} "
                  f"-> room {max_room_reached}  "
                  f"({len(transitions)} transitions saved)")
            self._best_room    = max_room_reached
            self._best_ep_data = transitions
            self.switch_count += 1
            return True
        return False

    def inject_into_buffer(self, buf: PrioritizedReplayBuffer) -> int:
        """
        Seed the replay buffer with survivor transitions at boosted priority.
        Called after amygdala switch and at the start of each training run.
        """
        if not self._best_ep_data:
            return 0
        for t in self._best_ep_data:
            # Scale reward by priority so these get sampled more often
            buf.push(
                t["feat"],
                t["reward"] * self._priority,
                t["next_feats"],
                t["done"],
            )
        return len(self._best_ep_data)

    def save(self, path: Path) -> None:
        """Persist to disk so amygdala state survives restarts."""
        meta = {
            "best_room":    self._best_room,
            "switch_count": self.switch_count,
            "n_transitions": len(self._best_ep_data),
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        # Store the actual transitions as numpy arrays
        if self._best_ep_data:
            feats      = np.array([t["feat"]    for t in self._best_ep_data])
            rewards    = np.array([t["reward"]  for t in self._best_ep_data])
            dones      = np.array([t["done"]    for t in self._best_ep_data])
            np.savez(path, feats=feats, rewards=rewards, dones=dones)

    def load(self, path: Path, n_actions: int) -> bool:
        """Load from disk. Returns True if successful."""
        meta_path = path.with_suffix(".meta.json")
        npz_path  = path
        if not meta_path.exists() or not npz_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            data = np.load(str(npz_path), allow_pickle=True)
            self._best_room    = meta["best_room"]
            self.switch_count  = meta["switch_count"]
            feats   = data["feats"]
            rewards = data["rewards"]
            dones   = data["dones"]
            self._best_ep_data = [
                {"feat": feats[i], "reward": float(rewards[i]),
                 "next_feats": [], "done": bool(dones[i])}
                for i in range(len(feats))
            ]
            print(f"  [amygdala] Loaded survivor: best_room={self._best_room} "
                  f"({len(self._best_ep_data)} transitions)")
            return True
        except Exception as exc:
            print(f"  [amygdala] Could not load survivor: {exc}")
            return False

    @property
    def best_room(self) -> int:
        return self._best_room


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------

def _q_values(agent: PortableNNAgent, ram: np.ndarray, n_actions: int) -> np.ndarray:
    state = ram.astype(np.float32) / 255.0
    q = np.empty(n_actions, dtype=np.float64)
    for a in range(n_actions):
        ah = np.zeros(n_actions, dtype=np.float32)
        ah[a] = 1.0
        q[a] = agent.forward(np.concatenate([state, ah]))
    return q


def _select_action_meta(pipeline, ram: np.ndarray, n_actions: int) -> int:
    """Policy B adapter: map RAM bytes to flat state for MetaStackPipeline."""
    state = ram.astype(np.float32) / 255.0
    return pipeline.select_action(state, explore=True)


def _train_meta(pipeline, ram: np.ndarray, action: int, reward: float,
                next_ram: np.ndarray, done: bool) -> None:
    """Policy B adapter: update MetaStackPipeline from a transition."""
    s  = ram.astype(np.float32) / 255.0
    s2 = next_ram.astype(np.float32) / 255.0
    pipeline.update(s, action, reward, s2, done)


def _train_batch(agent: PortableNNAgent, batch: list) -> None:
    agent._training = True
    try:
        for x, r, next_x_list, done in batch:
            if done or not next_x_list:
                target = r
            else:
                max_q  = max(agent.forward_target(nx) for nx in next_x_list)
                max_q  = np.clip(max_q, -500.0, 500.0)
                target = np.clip(r + agent.config.gamma * max_q, -500.0, 500.0)
            x_noisy = agent._apply_ext_noise(x)
            pred    = agent.forward(x_noisy)
            error   = np.clip(pred - float(target), -5, 5)
            agent.W3 -= agent.config.learning_rate * error * agent._last_h2
            agent.b3 -= agent.config.learning_rate * error
            dh2 = error * agent.W3[0] * (agent._last_h2 > 0)
            agent.W2 -= agent.config.learning_rate * np.outer(dh2, agent._last_h1)
            agent.b2 -= agent.config.learning_rate * dh2
            dh1 = (agent.W2.T @ dh2) * (agent._last_h1 > 0)
            agent.W1 -= agent.config.learning_rate * np.outer(dh1, agent._last_x)
            agent.b1 -= agent.config.learning_rate * dh1
    finally:
        agent._training = False
    agent.total_updates += 1


# ----------------------------------------------------------------------
# Main explorer episode
# ----------------------------------------------------------------------

def run_episode(
    env,
    agent:    PortableNNAgent,
    buf:      PrioritizedReplayBuffer,
    shaper:   ShapedRewardEngine,
    survivor: TrajectorySurvivor,
    n_actions: int,
    max_steps: int = 27_000,
    seed: int = 0,
    verbose: bool = False,
    # -- throng4 ganglia (optional) ----------------------------
    dreamer:        "DreamerEngine | None"  = None,
    dreamer_teacher: "DreamerTeacher | None" = None,
    options_lib:    "OptionsLibrary | None" = None,
    state_encoder:  "CompressedStateEncoder | None" = None,
    dream_interval: int = 10,
    advisory_rate:  float = 0.25,
    _default_hypotheses: list = None,
    # -- diagnostic tools -------------------------------------
    start_state_path: "Path | None" = None,
    render_env = None,
    frontier_mgr: "FrontierManager | None" = None,
    # -- A/B bandit -------------------------------------------
    policy_label: str = "A",          # "A" = PortableNN, "B" = MetaStack
    meta_pipeline = None,              # MetaStackPipeline instance (Policy B)
) -> dict[str, Any]:
    ram_obs, _ = env.reset(seed=seed)

    # Always reset the render env at episode start
    if render_env is not None:
        render_env.reset(seed=seed)

    # Load checkpoint save state if provided — always trust the saved state
    if start_state_path and start_state_path.exists():
        import pickle
        try:
            with open(start_state_path, "rb") as sf:
                saved = pickle.load(sf)
            env.unwrapped.ale.restoreState(saved["ram"])
            ram_obs = np.array(env.unwrapped.ale.getRAM(), dtype=np.uint8)
            _sx, _sy = int(ram_obs[42]), int(ram_obs[43])
            print(f"  [start] loaded {start_state_path.name}  →  x={_sx} y={_sy}")
            if render_env is not None:
                render_env.unwrapped.ale.restoreState(saved["rgb"])
        except Exception as exc:
            print(f"  [start-state] Load failed: {exc} — starting from spawn")

    shaper.reset()
    ram = np.array(ram_obs)

    total_game_reward  = 0.0
    total_shaped_reward = 0.0
    steps = 0
    done  = False
    max_room_seen = int(ram[3])
    all_rooms: set[int] = {max_room_seen}
    subgoals_hit: list[str] = []
    ep_transitions: list[dict] = []
    use_dreamer       = dreamer is not None
    active_option_id  = None
    options_discovered_ep = 0
    state_enc         = np.zeros(1, dtype=np.float32)
    _prev_lives       = int(ram[58])
    _render_save_slot = 0
    # Track steps survived per life (for death-trap detection)
    _steps_since_reload  = 0   # steps since last checkpoint restore
    _min_steps_per_life  = 9999  # minimum seen across all reloads this episode

    # -- Fear Memory (backward-discounting fear response) ---------------
    from throng4.basal_ganglia.fear_memory import FearMemory
    _fear = FearMemory(base_fear=1.0, decay=0.80, window=60, spatial_r=6)

    # -- WorldModel calibration tracker ---------------------------------
    _calibrator = None
    if use_dreamer and dreamer is not None:
        try:
            from throng4.basal_ganglia.world_model_calibrator import WorldModelCalibrator
            _calib_log = _EXP_DIR / "wm_calibration.jsonl"
            _calibrator = WorldModelCalibrator(
                world_model   = dreamer.world_model,
                state_encoder = state_encoder,
                window        = 200,
                log_path      = _calib_log,
                log_interval  = 50,
            )
            _calibrator.reset_episode()
        except Exception as _ce:
            _calibrator = None
            if verbose:
                print(f"[calib] WorldModelCalibrator unavailable: {_ce}")

    # -- Dyna-Q system (synthetic rollouts + confirmation matching) ------
    _dyna = None
    if use_dreamer and dreamer is not None:
        try:
            from throng4.basal_ganglia.dyna_system import DynaSystem
            _dyna = DynaSystem(
                world_model   = dreamer.world_model,
                state_encoder = state_encoder,
                n_actions     = n_actions,
            )
        except Exception as _de:
            _dyna = None
            if verbose:
                print(f"[dyna] DynaSystem unavailable: {_de}")

    while not done and steps < max_steps:
        # -- State encoding for dreamer ---------------------------------
        x_now = int(ram[42])
        y_now = int(ram[43])
        if use_dreamer:
            state_enc = state_encoder.encode(ram.astype(np.float32) / 255.0).data

            # Adaptive MCTS Dirichlet noise based on fear level at current position
            if dreamer is not None and dreamer.mcts is not None:
                dreamer.mcts.dir_alpha = _fear.mcts_dirichlet_alpha(
                    x_now, y_now, base_alpha=0.3
                )

            # Keep MCTS fall-detector closure up to date with real prev_y
            if "_prev_y_cell" in dir():
                _prev_y_cell[0] = y_now

            # Real-coordinate action mask + action preference boost.
            # Both use actual RAM x/y — bypasses encoded-state coordinate issues.
            from throng4.basal_ganglia.room_constants import (
                action_mask_for_position, action_preference_for_position, platform_name,
            )
            _action_mask  = action_mask_for_position(x_now, y_now, n_actions)
            _action_boost = action_preference_for_position(x_now, y_now, n_actions)
            _zone_name    = platform_name(x_now, y_now)

            # Store boost on MCTS so it can apply it during root expansion.
            # The MCTSPlanner will add this to the prior before Dirichlet noise.
            if dreamer is not None and dreamer.mcts is not None:
                dreamer.mcts.action_boost = _action_boost   # set each step

            # Check if a BehavioralOption wants to activate / stay active
            active_opts = dreamer_teacher.options.get_active_options(state_enc, threshold=0.35)
            if active_opts and active_option_id is None:
                active_option_id = active_opts[0].option_id

        # -- Action selection -------------------------------------------
        mcts_action = None
        _action_mask = _action_mask if use_dreamer else None
        if use_dreamer and dreamer_teacher is not None:
            adv = dreamer_teacher.get_best_action(state_enc,
                                                   action_mask=_action_mask)
            if adv is not None:
                # If DreamerTeacher gives us a recommendation, and it's from MCTS
                # (which has hypothesis_id=-1), we trust it absolutely.
                best_hyp_id = adv[2] if len(adv) > 2 else None
                if best_hyp_id == -1:
                    mcts_action = int(adv[0])
                    # Final safety: block banned actions even if MCTS recommends them
                    if _action_mask is not None and not _action_mask[mcts_action]:
                        mcts_action = None  # MCTS override refused; fall through to Q
                else:
                    # Legacy advisory from linear rollout (epsilon-greedy applies)
                    advisory_action = int(adv[0])
                    if agent.rng.rand() < advisory_rate:
                        mcts_action = advisory_action

        if mcts_action is not None:
            # MCTS acts directly; no Q-network or random noise
            action = mcts_action
        elif policy_label == "B" and meta_pipeline is not None:
            # Policy B: MetaStackPipeline selects action from raw RAM
            action = _select_action_meta(meta_pipeline, ram, n_actions)
        elif agent.rng.rand() < agent.epsilon:
            action = int(agent.rng.randint(n_actions))
        else:
            q = _q_values(agent, ram, n_actions)
            action = int(np.argmax(q))

        # Record position + action into fear ring buffer
        _fear.record(steps, x_now, y_now, action)

        ram_next_obs, game_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ram_next = np.array(ram_next_obs)

        # Render env: step in sync + terminal keypress check
        if render_env is not None:
            render_env.step(action)
            # ALE uses SDL (not pygame) so we check the terminal for keypresses.
            # Press 'S' in the terminal while training to save at any point.
            try:
                import msvcrt
                if msvcrt.kbhit():
                    _key = msvcrt.getch()
                    if _key in (b's', b'S'):
                        import pickle
                        _slot_dir = _EXP_DIR / "save_states" / "ALE_MontezumaRevenge_v5"
                        _slot_dir.mkdir(parents=True, exist_ok=True)
                        _slot_name = f"agent_render_slot{_render_save_slot:03d}_step{steps}.bin"
                        with open(_slot_dir / _slot_name, "wb") as _sf:
                            pickle.dump({
                                "rgb":    render_env.unwrapped.ale.cloneState(),
                                "ram":    env.unwrapped.ale.cloneState(),
                                "step":   steps,
                                "reward": total_game_reward,
                            }, _sf)
                        print(f"\n  [S] Agent state saved -> {_slot_name}")
                        _render_save_slot += 1
            except Exception:
                pass  # msvcrt not available (non-Windows)

        # Life-loss detection: reload checkpoint instead of respawning at start
        lives_now = int(ram_next[58])
        if lives_now < _prev_lives:
            # Backward-discount fear from this death event
            _fear.on_death(steps)
        if start_state_path and lives_now < _prev_lives and not done:
            # Log skull-zone deaths for RAM calibration (find skull x/y byte)
            _death_y = int(ram_next[43])
            if _death_y <= 165:
                try:
                    import json, time as _t
                    _skull_log = _EXP_DIR / "save_states" / "skull_death_ram.jsonl"
                    _skull_log.parent.mkdir(parents=True, exist_ok=True)
                    with open(_skull_log, "a") as _sl:
                        _sl.write(json.dumps({
                            "t": _t.time(), "step": steps,
                            "x": int(ram_next[42]), "y": _death_y,
                            "ram": ram_next.tolist(),
                        }) + "\n")
                except Exception:
                    pass
            # Reload the checkpoint
            try:
                import pickle
                with open(start_state_path, "rb") as _sf:
                    _saved = pickle.load(_sf)
                env.unwrapped.ale.restoreState(_saved["ram"])
                ram_next = np.array(env.unwrapped.ale.getRAM(), dtype=np.uint8)
                if render_env is not None:
                    render_env.unwrapped.ale.restoreState(_saved["rgb"])
                lives_now = int(ram_next[58])  # reset to checkpoint lives
                # Record survival time for death-trap detection
                _min_steps_per_life = min(_min_steps_per_life, _steps_since_reload)
                _steps_since_reload = 0
                # Death cancelled any pending frontier commit
                if frontier_mgr is not None:
                    frontier_mgr.cancel_pending()
                print(f"\n  [checkpoint] Life lost — reloaded checkpoint (step {steps})")
            except Exception as _exc:
                print(f"\n  [checkpoint] Reload failed: {_exc}")
        _prev_lives = lives_now

        # Room tracking
        room_now = int(ram_next[3])
        all_rooms.add(room_now)
        if room_now > max_room_seen:
            max_room_seen = room_now

        # Shaped reward
        shaped_r, events = shaper.step(ram_next, float(game_reward), done)
        for ev in events:
            subgoals_hit.append(ev.name)
            if verbose:
                print(f"    * {ev.name} (+{ev.reward:.1f})  "
                      f"x={ev.player_x} y={ev.player_y} room={ev.room}")
            # Feed subgoal outcomes back to options library
            if use_dreamer and active_option_id is not None:
                dreamer_teacher.options.record_outcome(active_option_id, ev.reward)
            # FrontierManager: auto-save new frontier on first-time subgoal
            if frontier_mgr is not None:
                frontier_mgr.on_subgoal(
                    ev.name, x=ev.player_x, y=ev.player_y,
                    step=steps, env=env, render_env=render_env,
                )

        # Live diagnostic print (render mode): show position + current target
        if render_env is not None and steps % 10 == 0:
            x_hud, y_hud = int(ram_next[42]), int(ram_next[43])
            from throng4.basal_ganglia.room_constants import platform_name, action_mask_for_position
            _zone_hud = platform_name(x_hud, y_hud)
            _mask_hud = action_mask_for_position(x_hud, y_hud, n_actions)
            _banned   = [a for a in range(n_actions) if not _mask_hud[a]]

            _det = getattr(shaper, "detector", None)
            _awarded = _det._awarded if _det else set()
            _pre_rope = {"right_lower_ladder_top", "center_ladder_descended",
                         "lower_platform_right", "rope_grabbed"}
            _effective_skip = _awarded | (_pre_rope if start_state_path else set())
            _tgt = next(
                (sg.name for sg in STAGE_1_SUBGOALS if sg.name not in _effective_skip),
                "all_done",
            )
            _fear_hud  = _fear.query(x_hud, y_hud)
            _fear_str  = f" fear={_fear_hud:.2f}" if _fear_hud > 0.05 else ""
            _ban_str   = f" BAN={_banned}" if _banned else ""
            print(f"\r  x={x_hud:3d} y={y_hud:3d} zone={_zone_hud:<18}"
                  f"{_fear_str}{_ban_str}  tgt={_tgt:<22} step={steps}",
                  end="", flush=True)

        # Feed death back to options library
        if use_dreamer and done and float(game_reward) < 0 and active_option_id is not None:
            dreamer_teacher.options.record_outcome(active_option_id, -5.0)

        total_game_reward   += float(game_reward)
        total_shaped_reward += shaped_r

        # Build transition features
        state  = ram.astype(np.float32) / 255.0
        ah_cur = np.zeros(n_actions, dtype=np.float32)
        ah_cur[action] = 1.0
        feat = np.concatenate([state, ah_cur])

        state_next = ram_next.astype(np.float32) / 255.0
        next_feats = []
        for a in range(n_actions):
            ah = np.zeros(n_actions, dtype=np.float32)
            ah[a] = 1.0
            nf = np.concatenate([state_next, ah])
            if np.isfinite(nf).all():
                next_feats.append(nf)

        buf.push(feat, shaped_r, next_feats, done)

        # Record for survivor consideration
        ep_transitions.append({
            "feat": feat, "reward": shaped_r,
            "next_feats": next_feats, "done": done,
        })

        # -- Dreamer update (world model learns from every step) --------
        if use_dreamer:
            next_enc = state_encoder.encode(ram_next.astype(np.float32) / 255.0).data
            dreamer.learn(state_enc, action, next_enc, shaped_r)

            # WorldModel calibration: measure predicted vs real transition
            if _calibrator is not None:
                try:
                    _calibrator.record(ram, action, ram_next, float(game_reward))
                    if verbose and steps % 200 == 0:
                        print(f"  {_calibrator.summary_line()}")
                except Exception:
                    pass

            # Dyna-Q: confirm synthetic predictions against real transition
            # and generate new synthetic transitions from current state
            if _dyna is not None:
                try:
                    confirmed = _dyna.record_real(ram, action, ram_next,
                                                  float(game_reward))
                    # Push confirmed synthetic entries to main replay buffer
                    for csamp in confirmed:
                        s_feat = np.concatenate([
                            csamp.state_enc,
                            np.eye(n_actions, dtype=np.float32)[csamp.action],
                        ])
                        # Build next feats for confirmed synthetic
                        nf_list = []
                        for a in range(n_actions):
                            nf = np.concatenate([csamp.pred_next,
                                                 np.eye(n_actions, dtype=np.float32)[a]])
                            nf_list.append(nf)
                        buf.push(s_feat, csamp.pred_reward * csamp.weight,
                                 nf_list, False)

                    # Synthetic rollout from current state
                    _dyna_n = getattr(run_episode, "_dyna_ratio",
                                      (_calibrator.recommended_dyna_ratio()
                                       if _calibrator else 0))
                    if _dyna_n > 0 and dreamer.is_calibrated:
                        synth = _dyna.rollout(
                            state_enc, x_now, y_now, n_steps=_dyna_n,
                            action_mask=_action_mask,
                            action_boost=_action_boost,
                        )
                        for ss in synth:
                            ss_feat = np.concatenate([
                                ss.state_enc,
                                np.eye(n_actions, dtype=np.float32)[ss.action],
                            ])
                            nf_list = [
                                np.concatenate([ss.pred_next,
                                                np.eye(n_actions, dtype=np.float32)[a]])
                                for a in range(n_actions)
                            ]
                            buf.push(ss_feat, ss.pred_reward * ss.weight,
                                     nf_list, False)

                    if verbose and steps % 200 == 0:
                        ds = _dyna.stats
                        print(f"  [dyna] buf={ds['buf_size']}  "
                              f"conf_ratio={ds['confirmed_ratio']:.2f}  "
                              f"conf_rate={ds['confirmation_rate']:.2f}  "
                              f"synth_n={_dyna_n}")
                except Exception:
                    pass

            # Run dream cycle every N steps once calibrated
            if steps % dream_interval == 0 and dreamer.is_calibrated:
                hypotheses  = _default_hypotheses or dreamer.create_default_hypotheses(n_actions)
                dream_res   = dreamer.dream(state_enc, hypotheses, n_steps=8)
                signals     = dreamer_teacher.process_dream_results(
                    dream_res, state_enc, steps,
                )
                if verbose and signals and steps % 100 == 0:
                    best_sig = signals[0]
                    print(f"    [dreamer] advisory=act{best_sig.action_recommended} "
                          f"conf={best_sig.confidence:.2f} "
                          f"reliance={dreamer_teacher.dreamer_reliance:.2f}")

            # Option discovery from dream results
            if dreamer.is_calibrated and steps % 50 == 0:
                for hyp_id, prof in dreamer_teacher.profiles.items():
                    opt = dreamer_teacher.options.discover_option(prof, steps)
                    if opt is not None:
                        options_discovered_ep += 1
                        if verbose:
                            print(f"    [options] discovered: {opt.summary()}")

            # Reset active option on termination
            if done:
                active_option_id = None
                dreamer_teacher.record_policy_action(
                    state_enc, action, advisory_action, shaped_r
                )

        # Train from buffer
        if steps % agent.config.train_freq == 0 and len(buf) >= agent.config.batch_size:
            batch = buf.sample(agent.config.batch_size)
            _train_batch(agent, batch)

        ram   = ram_next
        steps += 1
        _steps_since_reload += 1
        # Deferred frontier commit: saves checkpoint 60 steps after subgoal fires
        if frontier_mgr is not None:
            frontier_mgr.try_commit(steps, env=env, render_env=render_env)

    # Epsilon decay
    agent.epsilon = max(agent.config.epsilon_min,
                        agent.epsilon * agent.config.epsilon_decay)
    agent.episode_count += 1

    # Amygdala check - did this episode beat the room record?
    switched = survivor.record_trajectory(ep_transitions, max_room_seen)
    if switched:
        n_injected = survivor.inject_into_buffer(buf)
        if verbose:
            print(f"  [amygdala] Injected {n_injected} survivor transitions @ 10x priority")

    dreamer_stats = {}
    if use_dreamer:
        dreamer_stats = {
            "dreamer_calibrated":   dreamer.is_calibrated,
            "dreamer_reliance":     round(dreamer_teacher.dreamer_reliance, 4),
            "options_active":       len(dreamer_teacher.options.active_options),
            "options_promoted":     len(dreamer_teacher.options.promoted_options),
            "options_discovered_ep": options_discovered_ep,
        }

    # Determine current milestone target (first unawarded in STAGE_1_SUBGOALS)
    awarded = shaper.detector._awarded if hasattr(shaper, 'detector') else set()
    current_target = next(
        (sg.name for sg in STAGE_1_SUBGOALS if sg.name not in awarded), "all_done"
    )

    return {
        "game_reward":    float(total_game_reward),
        "shaped_reward":  float(total_shaped_reward),
        "steps":          steps,
        "min_steps_per_life": _min_steps_per_life if _min_steps_per_life < 9999 else steps,
        "max_room":       max_room_seen,
        "rooms_visited":  sorted(all_rooms),
        "subgoals_hit":   subgoals_hit,
        "current_target": current_target,
        "amygdala_switch": switched,
        "epsilon":        float(agent.epsilon),
        "novelty_cells":  shaper.novelty_tracker.n_unique_cells,
        **dreamer_stats,
    }


# ----------------------------------------------------------------------
# Explorer training loop
# ----------------------------------------------------------------------

def run_explorer(
    n_episodes:    int   = 500,
    weights_path:  str | None = None,
    seed:          int   = 42,
    resume:        bool  = False,
    save_freq:     int   = 25,
    max_steps:     int   = 27_000,
    verbose:       bool  = False,
    use_dreamer:   bool  = False,
    dream_interval: int  = 10,
    start_state:   str | None = None,
    render:        bool  = False,
    frameskip_n:   int   = 4,
    auto_frontier: bool  = False,
    freeze_frontier: bool = False,
    ab_test:       bool  = False,   # run A/B bandit tournament
    policy:        str   = "A",     # "A", "B", or "AB" (bandit)
) -> None:
    # Auto-reduce max_steps for checkpoint runs (no need for 27k steps from the rope)
    if start_state is not None and max_steps == 27_000:
        max_steps = 5_000
        print(f"  [checkpoint mode] max_steps auto-capped to {max_steps}")
    # MetaAdapter override: Montezuma is deterministic+sparse — use its params
    if _METAADAPTER_AVAILABLE and use_dreamer:
        _meta_cfg = {"stochastic": False, "reward_type": "sparse"}
        _meta_p = _MetaAdapter.params_for(_meta_cfg)
        dream_interval = _meta_p.dream_interval
        _advisory_rate = _meta_p.advisory_rate
        print(f"  [MetaAdapter] {_meta_p.label}: "
              f"dream_interval={dream_interval}  advisory_rate={_advisory_rate:.2f}")
    else:
        _advisory_rate = 0.25
    out_path      = _RESULTS_DIR / "montezuma_explorer.json"
    weights_out   = _RESULTS_DIR / "montezuma_explorer_weights.npz"
    survivor_path = _TRAJ_DIR / "amygdala_survivor.npz"

    # -- Resume --------------------------------------------------------
    prior_episodes: list[dict] = []
    if resume and out_path.exists():
        prior_episodes = json.loads(out_path.read_text(encoding="utf-8"))
        already_done   = len(prior_episodes)
        print(f"Resuming from ep {already_done}  "
              f"(best_room={max((e['max_room'] for e in prior_episodes), default=0)})")
        n_episodes = max(0, n_episodes - already_done)
        seed += already_done

    # -- Environment ---------------------------------------------------
    # frameskip=4: each agent step advances 4 ALE frames, ~4x speedup
    env       = gym.make(_GAME_ID, obs_type="ram", render_mode=None,
                         frameskip=frameskip_n)
    n_actions = env.action_space.n
    n_feats   = 128 + n_actions

    # -- Agent ---------------------------------------------------------
    cfg = AgentConfig(
        n_hidden=256, n_hidden2=128,
        epsilon=0.15,        # start with some exploration
        epsilon_decay=0.998, # decay slowly - exploration matters here
        epsilon_min=0.05,
        gamma=0.97,          # higher gamma - long-horizon shaped rewards
        learning_rate=0.003,
        replay_buffer_size=100_000,
        batch_size=64,
        train_freq=4,
        use_prioritized_replay=True,
        use_imitation_head=False,
        imitation_n_actions=n_actions,
    )
    agent = PortableNNAgent(n_feats, config=cfg, seed=seed)

    if weights_path and Path(weights_path).exists():
        agent.load_weights(weights_path)
        print(f"  [agent] Loaded weights: {weights_path}")
    elif weights_out.exists() and resume:
        agent.load_weights(str(weights_out))
        print(f"  [agent] Resumed weights: {weights_out}")

    # -- Policy B (MetaStackPipeline) + A/B Bandit ----------------------
    _use_meta = policy in ("B", "AB") or ab_test
    _agent_B = None
    _bandit  = None
    if _use_meta:
        if not _METASTACK_AVAILABLE:
            print("  [bandit] WARNING: MetaStackPipeline not available — falling back to A-only")
        else:
            _agent_B = MetaStackPipeline(
                n_inputs=128,          # raw RAM bytes (normalised to 0-1)
                n_outputs=n_actions,
                n_hidden=256,
                gamma=0.97,
                buffer_size=50_000,
                batch_size=64,
            )
            print(f"  [bandit] Policy B (MetaStackPipeline) ready  n_inputs=128  n_actions={n_actions}")

        if ab_test or policy == "AB":
            _bandit = PolicyBandit(alpha=0.2, temp=1.0)
            print(f"  [bandit] A/B tournament enabled  (alpha=0.2)")
        elif policy == "B" and _agent_B is not None:
            _bandit = PolicyBandit(alpha=0.2, force="B")
            print(f"  [bandit] Forced policy B")
    if policy == "A" and not ab_test:
        print(f"  [bandit] Policy A only (default PortableNN)")

    # -- Replay buffer -------------------------------------------------
    rng = np.random.RandomState(seed)
    buf = PrioritizedReplayBuffer(capacity=100_000, rng=rng)

    # -- Survivor / amygdala -------------------------------------------
    survivor = TrajectorySurvivor(priority_scale=10.0)
    if resume:
        survivor.load(survivor_path, n_actions)
        if survivor.best_room > 0:
            n_inj = survivor.inject_into_buffer(buf)
            print(f"  [amygdala] Pre-seeded {n_inj} survivor transitions")

    # -- Shaped reward engine ------------------------------------------
    shaper = ShapedRewardEngine(
        subgoal_scale=1.0,
        novelty_scale=0.3,
        novelty_bin=8,
        death_penalty=-5.0,
        pass_through_game_reward=True,
    )

    # -- Start-state checkpoint loading ----------------------------------
    _start_state_path: Path | None = None
    _save_states_dir = _EXP_DIR / "save_states" / "ALE_MontezumaRevenge_v5"
    if start_state:
        if start_state == "newest":
            from montezuma_frontier import _SUBGOAL_RANK
            import pickle as _pk

            def _read_frontier(p: Path) -> dict:
                try:
                    with open(p, "rb") as _f:
                        return _pk.load(_f)
                except Exception:
                    return {}

            def _frontier_rank_key(p: Path) -> int:
                return _SUBGOAL_RANK.get(_read_frontier(p).get("subgoal", ""), -1)

            def _is_valid_frontier(p: Path) -> bool:
                """Reject checkpoints saved mid-air (y > 220) — falling death traps."""
                d = _read_frontier(p)
                y = d.get("y", 0)
                return y <= 220

            frontier_files = sorted(
                _save_states_dir.glob("frontier_*.bin"),
                key=_frontier_rank_key, reverse=True,
            )
            # Pick deepest frontier that isn't mid-air
            valid = [f for f in frontier_files if _is_valid_frontier(f)]
            skipped = [f for f in frontier_files if not _is_valid_frontier(f)]
            if skipped:
                for sf in skipped:
                    d = _read_frontier(sf)
                    print(f"  [start-state] SKIPPING mid-air save: {sf.name}  y={d.get('y','?')}")
            if valid:
                _start_state_path = valid[0]
            else:
                # No valid frontiers — fall back to slot saves
                slot_files = sorted(
                    _save_states_dir.glob("*_slot*.bin"),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                _start_state_path = slot_files[0] if slot_files else None
        else:
            _start_state_path = Path(start_state)
        if _start_state_path and _start_state_path.exists():
            print(f"  [start-state] {_start_state_path.name}")
        else:
            print(f"  [start-state] WARNING: file not found — starting from spawn")
            _start_state_path = None

    # -- Render env (for watching) ----------------------------------------
    _render_env = None
    if render:
        import ale_py as _ale_py
        _render_env = gym.make(_GAME_ID, obs_type="ram", render_mode="human")
        gym.register_envs(_ale_py)
        print("  [render] Game window enabled")

    # -- throng4 basal-ganglia components (opt-in) ---------------------
    dreamer = dreamer_teacher = options_lib = state_encoder = None
    _default_hypotheses = None
    if use_dreamer:
        if not _THRONG4_AVAILABLE:
            print("[warn] --use-dreamer requested but throng4 not available; disabling")
            use_dreamer = False
        else:
            print("  [throng4] Initialising DreamerEngine + OptionsLibrary...")
            state_encoder   = CompressedStateEncoder(
                mode=EncodingMode.QUANTIZED, n_quantize_levels=4
            )
            # Determine encoded state size from a dummy encode
            _dummy = state_encoder.encode(
                np.zeros(128, dtype=np.float32)
            ).data
            enc_size = int(_dummy.size)

            # Lethal state detector for MCTS pruning.
            # Checks both static void zones AND dynamically detected falls.
            # _prev_y_cell is a mutable container updated each real step so
            # the MCTS closure always sees the current real prev_y (approximate
            # for lookahead nodes, but catches the most dangerous first step).
            from throng4.basal_ganglia.room_constants import (
                is_lethal_zone, is_falling, is_safe_fall,
            )
            _prev_y_cell = [235]   # mutable — updated each real step below

            def _is_lethal(c_state: np.ndarray) -> bool:
                try:
                    px = int(round(c_state[42] * 255.0))
                    py = int(round(c_state[43] * 255.0))
                    # Static void zone (calibrated)
                    if is_lethal_zone(px, py):
                        return True
                    # Fall detection: if predicted y is much lower than prev real y
                    # AND the fall doesn't head toward a safe platform → lethal
                    prev_y = _prev_y_cell[0]
                    if is_falling(prev_y, py) and not is_safe_fall(px, py, prev_y):
                        return True
                    return False
                except (IndexError, ValueError):
                    return False

            # Prior function for MCTS
            # Converts a compressed state back into 18 feature vectors,
            # then asks the PortableNNAgent for a Q-value softmax distribution.
            def _mcts_prior(c_state: np.ndarray) -> np.ndarray:
                try:
                    # state_encoder.decode expects (128,) raw array
                    # The env adapter needs the full bytearray to make features
                    raw_state = state_encoder.decode(
                        CompressedState(data=c_state, mode=state_encoder.mode)
                    )
                    ram_bytes = bytearray(np.clip(raw_state * 255.0, 0, 255).astype(np.uint8))
                    
                    # Generate feature vector for all 18 actions
                    from throng35.core.features import get_feature_vector
                    features_list = [
                        get_feature_vector(ram_bytes, a, n_actions)
                        for a in range(n_actions)
                    ]
                    
                    # Get softmax prior from agent (temperature=1.0)
                    return agent.get_mcts_prior(features_list, temperature=1.0)
                except Exception as e:
                    # Fallback to uniform if anything fails during decode/feature gen
                    return np.ones(n_actions, dtype=np.float32) / n_actions

            dreamer = DreamerEngine(
                n_hypotheses=3,
                network_size="micro",
                state_size=enc_size,
                n_actions=n_actions,
                dream_interval=dream_interval,
                use_mcts=True,         # UCT tree search once world model calibrates
                mcts_simulations=50,   # ≈10ms/search on this machine
                mcts_lethal_fn=_is_lethal, # hard-prune jumping off mid-platform
                mcts_prior_fn=_mcts_prior, # bias search toward learned Q-values
            )
            _default_hypotheses = dreamer.create_default_hypotheses(n_actions)
            dreamer_teacher = DreamerTeacher(
                n_actions=n_actions, state_dim=enc_size,
            )
            # The DreamerTeacher manages its own OptionsLibrary internally
            # (dreamer_teacher.options) - no separate options_lib needed
            print(f"  [throng4] enc_size={enc_size}  dream_interval={dream_interval}")

    # -- Training loop -------------------------------------------------
    all_episodes: list[dict] = list(prior_episodes)
    t0 = time.time()
    best_room_ever = max((e["max_room"] for e in prior_episodes), default=0)

    print(f"\n{'='*65}")
    print(f"  Montezuma Explorer - {n_episodes} episodes")
    print(f"  Shaped rewards: subgoals + novelty + death_penalty")
    print(f"  Amygdala survivor: priority_scale=10x on new-best-room trajectories")
    if use_dreamer:
        print(f"  [throng4] DreamerEngine + OptionsLibrary ACTIVE  "
              f"(dream_interval={dream_interval})")
    print(f"{'='*65}\n")
    print(f"  Sub-goals defined ({len(STAGE_1_SUBGOALS)}):")
    for sg in STAGE_1_SUBGOALS:
        print(f"    {sg.name:<25} reward={sg.reward:+.1f}")
    print()

    # -- FrontierManager (auto-advancing checkpoint curriculum) ----------
    _frontier_mgr: FrontierManager | None = None
    if auto_frontier or freeze_frontier:
        _fm_dir = _EXP_DIR / "save_states" / "ALE_MontezumaRevenge_v5"
        _frontier_mgr = FrontierManager(
            save_dir=_fm_dir,
            start_state_path=_start_state_path,
            freeze=freeze_frontier,
            min_rank_to_advance=3,
            commit_delay=0,   # immediate — y>220 post-load guard is the safety net
        )
        print(f"  [frontier] {'FROZEN' if freeze_frontier else 'AUTO'} — "
              f"starting from {'spawn' if _start_state_path is None else _start_state_path.name}")

    for ep in range(n_episodes):
        ep_seed = seed + ep
        # Get (potentially updated) frontier path for this episode
        _ep_start_path = (
            _frontier_mgr.begin_episode() if _frontier_mgr is not None
            else _start_state_path
        )
        # -- A/B bandit: select policy for this episode --
        _ep_policy = "A"
        if _bandit is not None:
            _ep_policy = _bandit.select()
        _ep_meta = _agent_B if _ep_policy == "B" else None

        result  = run_episode(
            env, agent, buf, shaper, survivor, n_actions,
            max_steps=max_steps, seed=ep_seed, verbose=verbose,
            # throng4 ganglia
            dreamer=dreamer,
            dreamer_teacher=dreamer_teacher,
            options_lib=options_lib,
            state_encoder=state_encoder,
            dream_interval=dream_interval,
            advisory_rate=_advisory_rate,
            _default_hypotheses=_default_hypotheses,
            # start-state + render + frontier
            start_state_path=_ep_start_path,
            render_env=_render_env,
            frontier_mgr=_frontier_mgr,
            # A/B bandit
            policy_label=_ep_policy,
            meta_pipeline=_ep_meta,
        )
        result["episode"] = len(all_episodes)
        result["policy"]  = _ep_policy
        all_episodes.append(result)

        # Bandit: record shaped reward for this policy
        if _bandit is not None:
            _bandit.record(_ep_policy, result["shaped_reward"])

        # Death-trap detection: if frontier is a spawn-trap, auto-rollback
        if _frontier_mgr is not None:
            _frontier_mgr.record_episode_steps(result["min_steps_per_life"])

        # Track best room
        if result["max_room"] > best_room_ever:
            best_room_ever = result["max_room"]

        # Progress log
        if (ep + 1) % 10 == 0 or result["amygdala_switch"] or ep == 0:
            recent = all_episodes[-20:]
            avg_shaped = np.mean([e["shaped_reward"] for e in recent])
            avg_game   = np.mean([e["game_reward"]   for e in recent])
            elapsed    = time.time() - t0
            eta        = elapsed / (ep + 1) * (n_episodes - ep - 1)
            switch_flag = " * AMYGDALA SWITCH!" if result["amygdala_switch"] else ""
            dreamer_tag = ""
            if use_dreamer:
                rel   = result.get("dreamer_reliance", 0.0)
                n_opt = result.get("options_active", 0)
                n_pro = result.get("options_promoted", 0)
                dreamer_tag = (f"  dreamer_rel={rel:.2f}  "
                               f"opts={n_opt}(promoted={n_pro})")
            print(
                f"  ep {len(all_episodes):4d}  "
                f"game={avg_game:7.2f}  shaped={avg_shaped:7.2f}  "
                f"room={result['max_room']}(best={best_room_ever})  "
                f"cells={result['novelty_cells']:5d}  "
                f"eps={agent.epsilon:.3f}  "
                f"eta={eta/60:.0f}m"
                f"{switch_flag}"
                f"{dreamer_tag}"
            )
            if result["subgoals_hit"]:
                print(f"    subgoals: {result['subgoals_hit']}")
            if _bandit is not None:
                print(f"    {_bandit.status_line()}")

        # Periodic save
        if (ep + 1) % save_freq == 0:
            out_path.write_text(json.dumps(all_episodes, indent=2), encoding="utf-8")
            agent.save_weights(str(weights_out))
            survivor.save(survivor_path)

    env.close()

    # Final save
    out_path.write_text(json.dumps(all_episodes, indent=2), encoding="utf-8")
    agent.save_weights(str(weights_out))
    survivor.save(survivor_path)

    # Summary
    rooms_reached = [e["max_room"] for e in all_episodes]
    subgoal_eps   = [e for e in all_episodes if e["subgoals_hit"]]
    print(f"\n{'='*65}")
    print(f"  Explorer complete. best_room={max(rooms_reached)}  "
          f"amygdala_switches={survivor.switch_count}")
    print(f"  Episodes with subgoals: {len(subgoal_eps)}/{len(all_episodes)}")
    print(f"  Novelty cells explored: {shaper.novelty_tracker.n_unique_cells}")
    if _frontier_mgr is not None:
        print(_frontier_mgr.summary())
    if _bandit is not None:
        print(_bandit.summary())
    if use_dreamer and dreamer_teacher is not None:
        print(f"  Options (active={len(dreamer_teacher.options.active_options)}  "
              f"promoted={len(dreamer_teacher.options.promoted_options)}):")
        for opt in dreamer_teacher.options.promoted_options:
            print(f"    {opt.summary()}")
        print(f"  Dreamer reliance: {dreamer_teacher.dreamer_reliance:.3f}  "
              f"(goal: <0.2)")
    print(f"  Weights: {weights_out}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Montezuma shaped-reward explorer with amygdala trajectory selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes",  type=int, default=500)
    p.add_argument("--weights",   default=None,
                   help="Starting weights (.npz). Defaults to BC weights if found.")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--resume",    action="store_true")
    p.add_argument("--max-steps", type=int, default=27_000,
                   help="Max steps per episode before forced reset")
    p.add_argument("--save-freq", type=int, default=25,
                   help="Save checkpoint every N episodes")
    p.add_argument("--verbose",   action="store_true",
                   help="Print sub-goal events as they fire")
    p.add_argument("--use-dreamer", action="store_true",
                   help="Activate throng4 DreamerEngine + OptionsLibrary (Phase 1 wiring)")
    p.add_argument("--dream-interval", type=int, default=10,
                   help="Run a dream cycle every N steps (default 10)")
    p.add_argument("--start-state", default=None, metavar="PATH|newest",
                   help="Load a .bin save state at the start of every episode. "
                        "Use 'newest' to auto-pick the most recently saved file.")
    p.add_argument("--render", action="store_true",
                   help="Open a game window so you can watch the agent play")
    p.add_argument("--frameskip", type=int, default=4, metavar="N",
                   help="ALE frames per agent step (default 4 = ~4x speedup). "
                        "Use 1 to disable.")
    p.add_argument("--auto-frontier", action="store_true",
                   help="Auto-advance checkpoint when a new subgoal fires for the first time")
    p.add_argument("--freeze-frontier", action="store_true",
                   help="Lock frontier: reload checkpoint on life loss but never advance it")
    p.add_argument("--rollback-frontier", action="store_true",
                   help="Immediately rollback to the previous frontier .bin file, then exit")
    p.add_argument("--ab-test", action="store_true",
                   help="Run A/B bandit tournament: Policy A (PortableNN) vs Policy B (MetaStack)")
    p.add_argument("--policy", default="A", choices=["A", "B", "AB"],
                   help="Which policy to use: A=PortableNN (default), B=MetaStack, AB=bandit")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    # --rollback-frontier: delete the newest frontier .bin so the previous one becomes newest
    if args.rollback_frontier:
        _ss_dir = _EXP_DIR / "save_states" / "ALE_MontezumaRevenge_v5"
        frontier_bins = sorted(_ss_dir.glob("frontier_*.bin"),
                               key=lambda p: p.stat().st_mtime)
        if len(frontier_bins) < 1:
            print("No frontier .bin files found to rollback.")
        else:
            newest = frontier_bins[-1]
            prev   = frontier_bins[-2] if len(frontier_bins) >= 2 else None
            newest.unlink()
            print(f"  [rollback] Removed: {newest.name}")
            print(f"  [rollback] Previous frontier: {prev.name if prev else 'spawn (none)'}")
        raise SystemExit(0)

    # Default to BC weights if no explicit flag
    bc_weights = _RESULTS_DIR / "bc_ALE_MontezumaRevenge_v5.npz"
    weights = args.weights
    if weights is None and bc_weights.exists():
        weights = str(bc_weights)
        print(f"Auto-selected BC weights: {bc_weights}")

    run_explorer(
        n_episodes     = args.episodes,
        weights_path   = weights,
        seed           = args.seed,
        resume         = args.resume,
        save_freq      = args.save_freq,
        max_steps      = args.max_steps,
        verbose        = args.verbose,
        use_dreamer    = args.use_dreamer,
        dream_interval = args.dream_interval,
        start_state    = args.start_state,
        render         = args.render,
        frameskip_n    = args.frameskip,
        auto_frontier  = args.auto_frontier,
        freeze_frontier = args.freeze_frontier,
        ab_test        = args.ab_test,
        policy         = args.policy,
    )
