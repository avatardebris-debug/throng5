"""
gauntlet_test.py — Comprehensive DQN vs Lean Brain comparison across
all compatible Gymnasium environments + Morris Water Maze.

Runs both bare DQN (all subsystems off) and lean brain (purged defaults)
side-by-side on every environment, producing a comparison report.

Usage:
  # Full gauntlet (all envs, 100 episodes each)
  python gauntlet_test.py --episodes 100

  # Quick gauntlet (50 episodes)
  python gauntlet_test.py --episodes 50

  # Specific categories only
  python gauntlet_test.py --categories classic_control toy_text custom

  # Morris Water Maze only
  python gauntlet_test.py --categories custom --episodes 50
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, ".")

import gymnasium as gym

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════

# ── Morris Water Maze (from throng1/2) ─────────────────────────────────

class MorrisWaterMaze:
    """Discrete-action Morris Water Maze adapted for WholeBrain."""
    def __init__(self, pool_radius=100, platform_radius=10, max_steps=100):
        self.pool_radius = pool_radius
        self.platform_radius = platform_radius
        self.max_steps = max_steps
        self.platform_pos = np.array([50.0, 50.0])  # Northeast quadrant
        self.n_actions = 8  # 8 compass directions
        self._angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self._directions = np.stack([np.cos(self._angles), np.sin(self._angles)], axis=1)

    def reset(self):
        angle = np.random.uniform(0, 2 * np.pi)
        self.position = np.array([
            self.pool_radius * np.cos(angle),
            self.pool_radius * np.sin(angle),
        ])
        self.steps = 0
        return self._obs()

    def step(self, action):
        action = int(action) % self.n_actions
        self.position += self._directions[action] * 5.0
        dist = np.linalg.norm(self.position)
        if dist > self.pool_radius:
            self.position = self.position / dist * self.pool_radius
        self.steps += 1

        dist_to_platform = np.linalg.norm(self.position - self.platform_pos)
        if dist_to_platform < self.platform_radius:
            return self._obs(), 1.0, True, {}
        elif self.steps >= self.max_steps:
            return self._obs(), -0.01, True, {}
        else:
            # Shaping: small reward for getting closer
            closeness = 1.0 - (dist_to_platform / (self.pool_radius * 2))
            return self._obs(), 0.01 * closeness, False, {}

    def _obs(self):
        return np.array([
            self.position[0] / self.pool_radius,
            self.position[1] / self.pool_radius,
        ], dtype=np.float32)


# ── Simple GridWorld ──────────────────────────────────────────────────

class GridWorldEnv:
    """5x5 grid with goal."""
    def __init__(self):
        self.size = 5
        self.goal = (4, 4)
        self.n_actions = 4

    def reset(self):
        self.pos = (0, 0)
        return self._obs()

    def step(self, action):
        x, y = self.pos
        if action == 0:   y = max(0, y - 1)
        elif action == 1: y = min(4, y + 1)
        elif action == 2: x = max(0, x - 1)
        elif action == 3: x = min(4, x + 1)
        self.pos = (x, y)
        if self.pos == self.goal:
            return self._obs(), 1.0, True, {}
        return self._obs(), -0.01, False, {}

    def _obs(self):
        return np.array([self.pos[0] / 4.0, self.pos[1] / 4.0], dtype=np.float32)


# ── Gymnasium Wrapper ────────────────────────────────────────────────

class GymEnv:
    """Wraps a Gymnasium env for normalized observations."""
    def __init__(self, env_id, obs_shape, n_actions, obs_low=None, obs_high=None,
                 discrete_obs=False, discrete_size=None, max_steps=500):
        self.env = gym.make(env_id)
        self.n_actions = n_actions
        self.max_steps = max_steps
        self.discrete_obs = discrete_obs
        self.discrete_size = discrete_size
        self.obs_low = np.asarray(obs_low, dtype=np.float32) if obs_low is not None else None
        self.obs_high = np.asarray(obs_high, dtype=np.float32) if obs_high is not None else None
        self._steps = 0

    def reset(self):
        obs, _ = self.env.reset()
        self._steps = 0
        return self._norm(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        self._steps += 1
        done = terminated or truncated or self._steps >= self.max_steps
        return self._norm(obs), reward, done, info

    def _norm(self, obs):
        if self.discrete_obs:
            s = self.discrete_size or 1
            # One-hot-ish encoding for discrete obs
            return np.array([float(obs) / max(s - 1, 1)], dtype=np.float32)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        if self.obs_low is not None and self.obs_high is not None:
            r = self.obs_high - self.obs_low
            r = np.where(r == 0, 1.0, r)
            return np.clip((obs - self.obs_low) / r, 0, 1)
        return obs


# ── Environment Registry ────────────────────────────────────────────

ENVIRONMENTS = {
    # ── Classic Control ──
    "CartPole-v1": {
        "category": "classic_control",
        "n_features": 4, "n_actions": 2, "max_steps": 500,
        "obs_low": [-4.8, -5.0, -0.42, -5.0],
        "obs_high": [4.8, 5.0, 0.42, 5.0],
        "description": "Balance a pole on a cart, 2 actions",
        "good_score": 200,
    },
    "MountainCar-v0": {
        "category": "classic_control",
        "n_features": 2, "n_actions": 3, "max_steps": 200,
        "obs_low": [-1.2, -0.07],
        "obs_high": [0.6, 0.07],
        "description": "Car must reach hilltop flag, 3 actions, sparse reward",
        "good_score": -110,
    },
    "Acrobot-v1": {
        "category": "classic_control",
        "n_features": 6, "n_actions": 3, "max_steps": 500,
        "obs_low": [-1, -1, -1, -1, -12.57, -28.27],
        "obs_high": [1, 1, 1, 1, 12.57, 28.27],
        "description": "Swing up double pendulum, 3 actions",
        "good_score": -100,
    },
    # ── Toy Text (Discrete obs) ──
    "FrozenLake-v1": {
        "category": "toy_text",
        "n_features": 2, "n_actions": 4, "max_steps": 100,
        "discrete_obs": True, "discrete_size": 16,
        "description": "4x4 slippery grid, reach goal without holes",
        "good_score": 0.7,
    },
    "Taxi-v3": {
        "category": "toy_text",
        "n_features": 1, "n_actions": 6, "max_steps": 200,
        "discrete_obs": True, "discrete_size": 500,
        "description": "Pick up and deliver passenger, 6 actions",
        "good_score": 8,
    },
    "CliffWalking-v0": {
        "category": "toy_text",
        "n_features": 1, "n_actions": 4, "max_steps": 200,
        "discrete_obs": True, "discrete_size": 48,
        "description": "Navigate grid cliff, avoid falling, 4 actions",
        "good_score": -13,
    },
    "Blackjack-v1": {
        "category": "toy_text",
        "n_features": 3, "n_actions": 2, "max_steps": 50,
        "description": "Hit or stand in blackjack",
        "good_score": 0.0,
    },
    # ── Custom ──
    "GridWorld": {
        "category": "custom",
        "n_features": 2, "n_actions": 4, "max_steps": 100,
        "description": "5x5 grid, goal at (4,4), shaped reward",
        "good_score": 0.8,
    },
    "MorrisWaterMaze": {
        "category": "custom",
        "n_features": 2, "n_actions": 8, "max_steps": 100,
        "description": "Circular pool, hidden platform NE quadrant, 8 compass dirs",
        "good_score": 0.5,
    },
}


def make_env(name: str) -> Any:
    """Create an environment instance."""
    config = ENVIRONMENTS[name]

    if name == "GridWorld":
        return GridWorldEnv()
    elif name == "MorrisWaterMaze":
        return MorrisWaterMaze()
    elif name == "FrozenLake-v1":
        env = gym.make("FrozenLake-v1", is_slippery=True)
        class FLWrap:
            def __init__(self, e):
                self.e = e
                self.n_actions = 4
            def reset(self):
                obs, _ = self.e.reset()
                return np.array([(obs % 4) / 3.0, (obs // 4) / 3.0], dtype=np.float32)
            def step(self, a):
                obs, r, t, tr, info = self.e.step(int(a))
                return np.array([(obs % 4) / 3.0, (obs // 4) / 3.0], dtype=np.float32), r, t or tr, info
        return FLWrap(env)
    elif name == "Blackjack-v1":
        env = gym.make("Blackjack-v1")
        class BJWrap:
            def __init__(self, e):
                self.e = e
                self.n_actions = 2
            def reset(self):
                obs, _ = self.e.reset()
                return np.array([obs[0] / 21.0, obs[1] / 10.0, float(obs[2])], dtype=np.float32)
            def step(self, a):
                obs, r, t, tr, info = self.e.step(int(a))
                return np.array([obs[0] / 21.0, obs[1] / 10.0, float(obs[2])], dtype=np.float32), r, t or tr, info
        return BJWrap(env)
    else:
        return GymEnv(
            name,
            obs_shape=config["n_features"],
            n_actions=config["n_actions"],
            obs_low=config.get("obs_low"),
            obs_high=config.get("obs_high"),
            discrete_obs=config.get("discrete_obs", False),
            discrete_size=config.get("discrete_size"),
            max_steps=config["max_steps"],
        )


# ════════════════════════════════════════════════════════════════════════
# RUNNERS
# ════════════════════════════════════════════════════════════════════════

def run_agent(agent_name: str, env_name: str, config: dict, n_episodes: int,
              brain_kwargs: Optional[dict] = None) -> Dict[str, Any]:
    """Run an agent on an environment and return results."""
    result = {
        "agent": agent_name,
        "env": env_name,
        "description": config["description"],
        "status": "UNKNOWN",
        "error": None,
        "episodes": 0,
        "all_rewards": [],
        "avg_reward": 0.0,
        "early_avg": 0.0,
        "late_avg": 0.0,
        "learning_slope": 0.0,
        "wall_time_sec": 0.0,
        "ms_per_step": 0.0,
        "total_steps": 0,
    }

    if agent_name == "random":
        return _run_random(env_name, config, n_episodes, result)

    # WholeBrain agent
    from brain.orchestrator import WholeBrain
    try:
        brain = WholeBrain(
            n_features=config["n_features"],
            n_actions=config["n_actions"],
            session_name=f"gauntlet_{agent_name}_{env_name}",
            enable_logging=False,
            use_torch=False,
            **(brain_kwargs or {}),
        )
    except Exception as e:
        result["status"] = "INIT_FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    all_rewards = []
    total_steps = 0
    step_times = []
    t_start = time.time()

    try:
        for ep in range(n_episodes):
            env = make_env(env_name)
            obs = env.reset()
            ep_reward = 0.0
            action = 0

            for step in range(config["max_steps"]):
                t_s = time.perf_counter()
                brain_out = brain.step(
                    obs=obs, prev_action=action,
                    reward=ep_reward if step == 0 else reward,
                    done=False,
                )
                step_times.append((time.perf_counter() - t_s) * 1000)

                action = brain_out.get("action", 0)
                action = max(0, min(action, config["n_actions"] - 1))
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                total_steps += 1
                if done:
                    brain.step(obs=obs, prev_action=action, reward=reward, done=True)
                    break

            all_rewards.append(ep_reward)

            if (ep + 1) % max(1, n_episodes // 5) == 0:
                recent = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards)
                ms = np.mean(step_times[-100:]) if step_times else 0
                print(f"      Ep {ep+1:3d}: reward={ep_reward:7.2f}  avg10={recent:7.2f}  {ms:.1f}ms/step")

    except Exception as e:
        result["status"] = "RUNTIME_FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        if not all_rewards:
            try: brain.close()
            except: pass
            return result

    elapsed = time.time() - t_start
    try: brain.close()
    except: pass

    return _analyze(result, all_rewards, total_steps, step_times, elapsed)


def _run_random(env_name, config, n_episodes, result):
    """Random agent baseline."""
    all_rewards = []
    total_steps = 0
    t_start = time.time()
    for _ in range(n_episodes):
        env = make_env(env_name)
        env.reset()
        ep_reward = 0.0
        for step in range(config["max_steps"]):
            action = np.random.randint(config["n_actions"])
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            total_steps += 1
            if done: break
        all_rewards.append(ep_reward)
    result["agent"] = "random"
    return _analyze(result, all_rewards, total_steps, [], time.time() - t_start)


def _analyze(result, all_rewards, total_steps, step_times, elapsed):
    """Analyze rewards and classify learning."""
    result["episodes"] = len(all_rewards)
    result["all_rewards"] = [float(r) for r in all_rewards]
    result["avg_reward"] = float(np.mean(all_rewards))
    result["total_steps"] = total_steps
    result["wall_time_sec"] = elapsed
    result["ms_per_step"] = float(np.mean(step_times)) if step_times else 0.0

    split = max(1, len(all_rewards) // 3)
    result["early_avg"] = float(np.mean(all_rewards[:split]))
    result["late_avg"] = float(np.mean(all_rewards[-split:]))

    if len(all_rewards) > 10:
        x = np.arange(len(all_rewards))
        slope, _ = np.polyfit(x, all_rewards, 1)
        result["learning_slope"] = float(slope)

    if result.get("error"):
        result["status"] = result.get("status", "RUNTIME_FAIL")
    elif result["late_avg"] > result["early_avg"] * 1.1 and result["learning_slope"] > 0:
        result["status"] = "LEARNING"
    elif result["late_avg"] > result["early_avg"]:
        result["status"] = "SLIGHT_IMPROVEMENT"
    else:
        result["status"] = "NO_LEARNING"

    return result


# ════════════════════════════════════════════════════════════════════════
# REPORT
# ════════════════════════════════════════════════════════════════════════

def generate_report(results: Dict[str, List[Dict]], out_dir: Path) -> str:
    """Generate comparison report."""
    lines = [
        "# WholeBrain Gauntlet Test Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\nComparing: **Random** vs **Bare DQN** (all subsystems off) vs **Lean Brain** (purged defaults)",
        "",
    ]

    # Group by category
    categories = {}
    for env_name, env_results in results.items():
        cat = ENVIRONMENTS[env_name]["category"]
        if cat not in categories:
            categories[cat] = {}
        categories[cat][env_name] = env_results

    overall_wins = {"bare_dqn": 0, "lean_brain": 0, "tie": 0}

    for cat_name, cat_envs in sorted(categories.items()):
        lines.append(f"\n## {cat_name.replace('_', ' ').title()}\n")
        lines.append("| Environment | Description | Random | Bare DQN | Lean Brain | Winner |")
        lines.append("|-------------|-------------|--------|----------|------------|--------|")

        for env_name, env_results in sorted(cat_envs.items()):
            desc = ENVIRONMENTS[env_name]["description"][:35]
            random_r = next((r for r in env_results if r["agent"] == "random"), None)
            dqn_r = next((r for r in env_results if r["agent"] == "bare_dqn"), None)
            lean_r = next((r for r in env_results if r["agent"] == "lean_brain"), None)

            def fmt(r):
                if r is None: return "—"
                if r.get("error"): return f"💥 FAIL"
                return f"{r['avg_reward']:.2f} ({r['early_avg']:.1f}→{r['late_avg']:.1f})"

            # Determine winner
            if dqn_r and lean_r and not dqn_r.get("error") and not lean_r.get("error"):
                if lean_r["avg_reward"] > dqn_r["avg_reward"] * 1.05:
                    winner = "🧠 Lean"
                    overall_wins["lean_brain"] += 1
                elif dqn_r["avg_reward"] > lean_r["avg_reward"] * 1.05:
                    winner = "⚡ DQN"
                    overall_wins["bare_dqn"] += 1
                else:
                    winner = "🤝 Tie"
                    overall_wins["tie"] += 1
            else:
                winner = "—"

            lines.append(f"| {env_name[:25]} | {desc} | {fmt(random_r)} | {fmt(dqn_r)} | {fmt(lean_r)} | {winner} |")

    # Overall summary
    lines.append("\n## Overall Score\n")
    total = sum(overall_wins.values())
    lines.append(f"- **Lean Brain wins**: {overall_wins['lean_brain']}/{total}")
    lines.append(f"- **Bare DQN wins**: {overall_wins['bare_dqn']}/{total}")
    lines.append(f"- **Ties**: {overall_wins['tie']}/{total}")

    # Speed comparison
    lines.append("\n## Speed Comparison\n")
    lines.append("| Environment | Bare DQN ms/step | Lean Brain ms/step | Overhead |")
    lines.append("|-------------|-----------------|-------------------|----------|")
    for env_name, env_results in sorted(results.items()):
        dqn_r = next((r for r in env_results if r["agent"] == "bare_dqn"), None)
        lean_r = next((r for r in env_results if r["agent"] == "lean_brain"), None)
        if dqn_r and lean_r:
            dqn_ms = dqn_r.get("ms_per_step", 0)
            lean_ms = lean_r.get("ms_per_step", 0)
            overhead = (lean_ms / max(dqn_ms, 0.001) - 1) * 100
            lines.append(f"| {env_name[:25]} | {dqn_ms:.1f} | {lean_ms:.1f} | {overhead:+.0f}% |")

    report = "\n".join(lines)
    report_path = out_dir / "gauntlet_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n📊 Report saved to: {report_path}")
    return report


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="WholeBrain Gauntlet — DQN vs Lean Brain")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filter by category: classic_control, toy_text, custom")
    parser.add_argument("--envs", nargs="+", default=None,
                        help="Run specific environments only")
    parser.add_argument("--out-dir", type=str, default="gauntlet_results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Filter environments
    envs_to_test = {}
    for name, config in ENVIRONMENTS.items():
        if args.envs and name not in args.envs:
            continue
        if args.categories and config["category"] not in args.categories:
            continue
        envs_to_test[name] = config

    agents = [
        ("random", None),
        ("bare_dqn", {
            "enabled_systems": {k: False for k in [
                "world_model", "dreams", "causal_model", "skill_library",
                "attribution", "stage_classifier", "counterfactual",
                "hippocampus_store", "threat_gating", "probe_runner",
            ]}
        }),
        ("lean_brain", {}),  # Uses purged defaults (everything remaining = on)
    ]

    total_runs = len(envs_to_test) * len(agents)
    print("=" * 70)
    print(f"  WHOLEBRAIN GAUNTLET — DQN vs Lean Brain")
    print(f"  Environments: {len(envs_to_test)}")
    print(f"  Agents: {len(agents)} (random, bare_dqn, lean_brain)")
    print(f"  Total runs: {total_runs}")
    print(f"  Episodes per run: {args.episodes}")
    print("=" * 70)

    all_results = {}  # env_name -> [results per agent]
    run_idx = 0

    for env_name, config in envs_to_test.items():
        all_results[env_name] = []
        for agent_name, brain_kwargs in agents:
            run_idx += 1
            print(f"\n{'─' * 70}")
            print(f"  [{run_idx}/{total_runs}] {agent_name} on {env_name}")
            print(f"    {config['description']}")
            print(f"{'─' * 70}")

            result = run_agent(
                agent_name=agent_name,
                env_name=env_name,
                config=config,
                n_episodes=args.episodes,
                brain_kwargs=brain_kwargs,
            )
            all_results[env_name].append(result)

            status_icon = {"LEARNING": "✅", "SLIGHT_IMPROVEMENT": "⚠️",
                          "NO_LEARNING": "❌", "INIT_FAIL": "💥",
                          "RUNTIME_FAIL": "💥"}.get(result["status"], "❓")
            print(f"\n    → {status_icon} {result['status']}: avg={result['avg_reward']:.2f} "
                  f"({result['early_avg']:.2f}→{result['late_avg']:.2f}) "
                  f"{result.get('ms_per_step', 0):.1f}ms/step")

    # Save raw results
    raw_path = out_dir / "gauntlet_results.json"
    with open(raw_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_episodes": args.episodes,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n💾 Raw results saved to: {raw_path}")

    # Generate report
    report = generate_report(all_results, out_dir)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
