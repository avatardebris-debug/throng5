"""
round_robin_runner.py
=====================
Phase 1: Cross-game round-robin training.

Cycles through a game roster, running N episodes per game per round before
rotating. Each game's weights, episode log, and dreamer state are persisted
between rounds so learning is continuous.

Usage:
    python round_robin_runner.py --rounds 5 --dry-run
    python round_robin_runner.py --rounds 20 --episodes-per-round 50

Output per game:
    benchmark_results/<slug>_rr_episodes.json  -- episode history
    benchmark_results/<slug>_rr_weights.npz    -- agent weights
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── Game roster ────────────────────────────────────────────────────────
# environment_type metadata is read by MetaAdapter for parameter selection
GAME_ROSTER: list[dict] = [
    {
        "id":               "ALE/MontezumaRevenge-v5",
        "obs_type":         "ram",
        "n_features":       128,
        "stochastic":       False,
        "reward_type":      "sparse",
        "episodes_per_round": 50,
        "max_steps":        8_000,
        "notes":            "Deterministic, very sparse reward, long episodes",
    },
    {
        "id":               "ALE/Breakout-v5",
        "obs_type":         "ram",
        "n_features":       128,
        "stochastic":       True,
        "reward_type":      "dense",
        "episodes_per_round": 100,
        "max_steps":        5_000,
        "notes":            "Stochastic ball physics, dense brick rewards",
    },
    {
        "id":               "ALE/Frogger-v5",
        "obs_type":         "ram",
        "n_features":       128,
        "stochastic":       True,
        "reward_type":      "sparse",
        "episodes_per_round": 100,
        "max_steps":        4_000,
        "notes":            "Stochastic traffic, sparse crossing rewards",
    },
    {
        "id":               "CartPole-v1",
        "obs_type":         "state",
        "n_features":       4,
        "stochastic":       False,
        "reward_type":      "dense",
        "episodes_per_round": 200,
        "max_steps":        500,
        "notes":            "Deterministic, dense (+1/step), fully observable state",
    },
]

# ── Imports ────────────────────────────────────────────────────────────
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

from throng4.learning.portable_agent import PortableNNAgent, AgentConfig

try:
    from throng4.basal_ganglia.dreamer_engine import DreamerEngine
    from throng4.basal_ganglia.hypothesis_profiler import DreamerTeacher
    from throng4.basal_ganglia.compressed_state import CompressedStateEncoder, EncodingMode
    _DREAMER_OK = True
except ImportError as e:
    _DREAMER_OK = False
    print(f"[warn] throng4 dreamer not importable ({e}); running flat Q-learning only")

try:
    from throng4.meta_policy.meta_adapter import MetaAdapter
    _ADAPTER_OK = True
except ImportError:
    _ADAPTER_OK = False

_RESULTS = _ROOT / "benchmark_results"
_RESULTS.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────
# Per-game state (persisted across rounds)
# ──────────────────────────────────────────────────────────────────────

class GameState:
    def __init__(self, cfg: dict, seed: int = 42):
        self.cfg     = cfg
        self.game_id = cfg["id"]
        slug         = self.game_id.replace("/","_").replace("-","_")
        self.weights_path  = _RESULTS / f"{slug}_rr_weights.npz"
        self.episodes_path = _RESULTS / f"{slug}_rr_episodes.json"

        n_feat = cfg["n_features"]
        if cfg["obs_type"] == "ram":
            n_actions = gym.make(self.game_id, obs_type="ram").action_space.n
        else:
            env_tmp = gym.make(self.game_id)
            n_actions = env_tmp.action_space.n
            env_tmp.close()
        self.n_actions = n_actions

        feat_dim = n_feat + n_actions
        acfg = AgentConfig(
            epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995,
            train_freq=4, batch_size=32, gamma=0.97,
        )
        self.agent = PortableNNAgent(feat_dim, config=acfg)
        if self.weights_path.exists():
            self.agent.load_weights(str(self.weights_path))
            print(f"  [{slug}] Loaded weights: {self.weights_path.name}")

        # Dreamer components
        self.dreamer = self.dreamer_teacher = self.encoder = None
        self.hypotheses = None
        if _DREAMER_OK:
            self._init_dreamer(n_feat, n_actions)

        # Metadata for MetaAdapter
        self.meta_params = None
        if _ADAPTER_OK:
            self.meta_params = MetaAdapter.params_for(cfg)
            if self.dreamer and self.meta_params:
                self.dreamer.dream_interval = self.meta_params.dream_interval
                print(f"  [{slug}] MetaAdapter: dream_interval={self.meta_params.dream_interval}  "
                      f"advisory_rate={self.meta_params.advisory_rate:.2f}")

        self.all_episodes: list[dict] = []
        if self.episodes_path.exists():
            try:
                self.all_episodes = json.loads(
                    self.episodes_path.read_text(encoding="utf-8")
                )
                print(f"  [{slug}] Resumed {len(self.all_episodes)} prior episodes")
            except Exception:
                pass

    def _init_dreamer(self, n_feat: int, n_actions: int):
        enc = CompressedStateEncoder(mode=EncodingMode.QUANTIZED, n_quantize_levels=4)
        dummy = enc.encode(np.zeros(n_feat, dtype=np.float32)).data
        enc_size = int(dummy.size)
        self.encoder = enc
        self.dreamer = DreamerEngine(
            n_hypotheses=3, network_size="micro",
            state_size=enc_size, n_actions=n_actions, dream_interval=10,
        )
        self.hypotheses = self.dreamer.create_default_hypotheses(n_actions)
        self.dreamer_teacher = DreamerTeacher(n_actions=n_actions, state_dim=enc_size)

    def save(self):
        self.agent.save_weights(str(self.weights_path))
        self.episodes_path.write_text(
            json.dumps(self.all_episodes, indent=2), encoding="utf-8"
        )


# ──────────────────────────────────────────────────────────────────────
# Single episode runner (generic, RAM or state observation)
# ──────────────────────────────────────────────────────────────────────

def _q_values(agent: PortableNNAgent, obs: np.ndarray, n_actions: int) -> np.ndarray:
    obs_n = obs.astype(np.float32)
    if obs_n.max() > 1.0:
        obs_n = obs_n / 255.0
    q = np.empty(n_actions, dtype=np.float64)
    for a in range(n_actions):
        ah = np.zeros(n_actions, dtype=np.float32)
        ah[a] = 1.0
        q[a] = agent.forward(np.concatenate([obs_n, ah]))
    return q


def run_episode_generic(
    env,
    gs: GameState,
    seed: int,
    max_steps: int,
    use_dreamer: bool = True,
    advisory_rate: float = 0.25,
) -> dict:
    obs, _ = env.reset(seed=seed)
    obs = np.array(obs, dtype=np.float32)

    total_reward = 0.0
    steps        = 0
    done         = False
    n_actions    = gs.n_actions

    state_enc = np.zeros(1, dtype=np.float32)  # placeholder

    while not done and steps < max_steps:
        # State encoding for dreamer
        if use_dreamer and gs.encoder is not None:
            state_enc = gs.encoder.encode(
                obs if obs.max() <= 1.0 else obs / 255.0
            ).data

        # Action selection
        advisory = None
        if use_dreamer and gs.dreamer_teacher is not None:
            adv = gs.dreamer_teacher.get_best_action(state_enc)
            if adv is not None:
                advisory = adv[0]

        if gs.agent.rng.rand() < gs.agent.epsilon:
            action = int(gs.agent.rng.randint(n_actions))
        elif advisory is not None and gs.agent.rng.rand() < advisory_rate:
            action = int(advisory)
        else:
            q      = _q_values(gs.agent, obs, n_actions)
            action = int(np.argmax(q))

        obs_next_raw, reward, terminated, truncated, _ = env.step(action)
        obs_next = np.array(obs_next_raw, dtype=np.float32)
        done     = terminated or truncated

        # Features for replay
        obs_n      = obs if obs.max() <= 1.0 else obs / 255.0
        obs_next_n = obs_next if obs_next.max() <= 1.0 else obs_next / 255.0
        ah = np.zeros(n_actions, dtype=np.float32); ah[action] = 1.0
        feat      = np.concatenate([obs_n, ah])
        ah_next   = np.zeros(n_actions, dtype=np.float32)
        feat_next = np.concatenate([obs_next_n, ah_next])

        # Push to replay buffer and train (record_step handles push+train+target_sync)
        gs.agent.record_step(feat, reward, [feat_next], done)

        # Dreamer world model update
        if use_dreamer and gs.dreamer is not None and gs.encoder is not None:
            next_enc = gs.encoder.encode(
                obs_next if obs_next.max() <= 1.0 else obs_next / 255.0
            ).data
            gs.dreamer.learn(state_enc, action, next_enc, reward)

            if steps % gs.dreamer.dream_interval == 0 and gs.dreamer.is_calibrated:
                hyps      = gs.hypotheses or gs.dreamer.create_default_hypotheses(n_actions)
                dream_res = gs.dreamer.dream(state_enc, hyps, n_steps=8)
                gs.dreamer_teacher.process_dream_results(dream_res, state_enc, steps)

            # Record actual policy action vs. dreamer advisory so reliance evolves
            gs.dreamer_teacher.record_policy_action(
                state=state_enc,
                actual_action=action,
                dreamer_action=advisory,
                reward=float(reward),
            )

        total_reward += float(reward)
        obs   = obs_next
        steps += 1

    gs.agent.epsilon = max(gs.agent.config.epsilon_min,
                           gs.agent.epsilon * gs.agent.config.epsilon_decay)
    gs.agent.episode_count += 1

    return {
        "game":         gs.game_id,
        "episode":      len(gs.all_episodes),
        "game_reward":  total_reward,
        "steps":        steps,
        "epsilon":      float(gs.agent.epsilon),
        "dreamer_calibrated": bool(gs.dreamer.is_calibrated) if gs.dreamer else False,
        "dreamer_reliance":   round(gs.dreamer_teacher.dreamer_reliance, 4)
                              if gs.dreamer_teacher else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
# Round-robin orchestrator
# ──────────────────────────────────────────────────────────────────────

def run_round_robin(
    rounds: int     = 10,
    seed:   int     = 42,
    dry_run: bool   = False,
    games:  list[str] | None = None,
) -> None:
    roster = GAME_ROSTER
    if games:
        roster = [g for g in GAME_ROSTER if g["id"] in games]

    print(f"\n{'='*65}")
    print(f"  Round-Robin Cross-Game Training")
    print(f"  Games: {len(roster)}  Rounds: {rounds}")
    for g in roster:
        print(f"    {g['id']:40s}  {g['episodes_per_round']}ep/round  "
              f"{'stochastic' if g['stochastic'] else 'deterministic'}  {g['reward_type']}")
    print(f"{'='*65}\n")

    if dry_run:
        print("[dry-run] Would initialize game states. Exiting.")
        return

    # Build per-game state (loads weights + episode history)
    game_states: dict[str, GameState] = {}
    for cfg in roster:
        print(f"Initializing {cfg['id']}...")
        game_states[cfg["id"]] = GameState(cfg, seed=seed)

    t0 = time.time()
    for rnd in range(1, rounds + 1):
        print(f"\n--- Round {rnd}/{rounds} ---")
        for cfg in roster:
            gs      = game_states[cfg["id"]]
            n_ep    = cfg["episodes_per_round"]
            rate    = gs.meta_params.advisory_rate if gs.meta_params else 0.25
            use_d   = gs.dreamer is not None

            # Open environment
            obs_kw = {"obs_type": "ram"} if cfg["obs_type"] == "ram" else {}
            env = gym.make(cfg["id"], **obs_kw)

            ep_rewards = []
            for ep in range(n_ep):
                ep_seed = seed + rnd * 10000 + ep
                result  = run_episode_generic(
                    env, gs, seed=ep_seed,
                    max_steps=cfg["max_steps"],
                    use_dreamer=use_d,
                    advisory_rate=rate,
                )
                gs.all_episodes.append(result)
                ep_rewards.append(result["game_reward"])

            env.close()
            gs.save()

            avg_r = float(np.mean(ep_rewards)) if ep_rewards else 0.0
            slug  = cfg["id"].split("/")[-1]
            print(f"  {slug:30s}  {n_ep}ep  avg_r={avg_r:8.2f}  "
                  f"eps={gs.agent.epsilon:.3f}  "
                  f"dreamer_rel={gs.dreamer_teacher.dreamer_reliance:.2f}"
                  if gs.dreamer_teacher else
                  f"  {slug:30s}  {n_ep}ep  avg_r={avg_r:8.2f}  "
                  f"eps={gs.agent.epsilon:.3f}")

        elapsed = time.time() - t0
        print(f"  Round {rnd} done  ({elapsed/60:.1f}m elapsed)")

    print(f"\n{'='*65}")
    print(f"  Round-robin complete. {rounds} rounds x {len(roster)} games")
    for cfg in roster:
        gs   = game_states[cfg["id"]]
        best = max((e["game_reward"] for e in gs.all_episodes), default=0)
        print(f"  {cfg['id']:40s}  episodes={len(gs.all_episodes):5d}  best={best:.1f}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="Cross-game round-robin training")
    p.add_argument("--rounds",           type=int, default=10)
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--dry-run",          action="store_true")
    p.add_argument("--games",            nargs="*", default=None,
                   help="Subset of game IDs to run (defaults to full roster)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run_round_robin(
        rounds  = args.rounds,
        seed    = args.seed,
        dry_run = args.dry_run,
        games   = args.games,
    )
