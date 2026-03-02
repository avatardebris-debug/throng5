"""
ablation_test.py — Systematic ablation testing for WholeBrain.

Runs subtraction-first testing by default: starts with full brain, removes
one subsystem at a time, measures learning impact + compute cost.

Also supports additive testing (bare DQN + one system at a time).

Outputs:
  - JSON results file with per-configuration reward curves + timing
  - Markdown diagnostic report showing what worked, what broke, and why

Usage:
  # Subtraction sweep (recommended first — may find the bug)
  python ablation_test.py --mode subtract --envs cartpole gridworld --episodes 200

  # Additive sweep (bare DQN + one system at a time)
  python ablation_test.py --mode add --envs cartpole gridworld --episodes 200

  # Single config: full brain with curiosity disabled
  python ablation_test.py --mode custom --disable curiosity --episodes 200

  # Single config: bare DQN with world_model enabled
  python ablation_test.py --mode custom --bare --enable world_model --episodes 200

  # Quick smoke test (10 episodes per config)
  python ablation_test.py --mode subtract --quick
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, ".")

# ── Environments (no external dependencies beyond gymnasium) ──────────

import gymnasium as gym


class SimpleEnv:
    """Minimal env wrapper."""
    def __init__(self, env, obs_low, obs_high, n_actions):
        self.env = env
        self.obs_low = np.asarray(obs_low, dtype=np.float32)
        self.obs_high = np.asarray(obs_high, dtype=np.float32)
        self.n_actions = n_actions

    def reset(self):
        obs, _ = self.env.reset()
        return self._norm(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._norm(obs), reward, terminated or truncated, info

    def _norm(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        r = self.obs_high - self.obs_low
        r = np.where(r == 0, 1.0, r)
        return np.clip((obs - self.obs_low) / r, 0, 1)


class GridWorldEnv:
    """Simple 5x5 grid."""
    def __init__(self):
        self.size = 5
        self.goal = (4, 4)
        self.n_actions = 4

    def reset(self):
        self.pos = (0, 0)
        return self._obs()

    def step(self, action):
        x, y = self.pos
        if action == 0: y = max(0, y - 1)
        elif action == 1: y = min(4, y + 1)
        elif action == 2: x = max(0, x - 1)
        elif action == 3: x = min(4, x + 1)
        self.pos = (x, y)
        if self.pos == self.goal:
            return self._obs(), 1.0, True, {}
        return self._obs(), -0.01, False, {}

    def _obs(self):
        return np.array([self.pos[0] / 4.0, self.pos[1] / 4.0], dtype=np.float32)


ENVS = {
    "gridworld": {"n_features": 2, "n_actions": 4, "max_steps": 100,
                   "description": "5x5 grid, goal at (4,4), dense reward"},
    "cartpole":  {"n_features": 4, "n_actions": 2, "max_steps": 500,
                   "description": "Balance pole, 2 actions, dense reward"},
    "mountaincar": {"n_features": 2, "n_actions": 3, "max_steps": 200,
                     "description": "Reach flag, 3 actions, sparse reward"},
    "frozenlake": {"n_features": 2, "n_actions": 4, "max_steps": 100,
                    "description": "4x4 slippery grid, sparse reward"},
}


def make_env(name: str):
    if name == "gridworld":
        return GridWorldEnv()
    elif name == "cartpole":
        return SimpleEnv(gym.make("CartPole-v1"),
                         [-4.8, -5.0, -0.42, -5.0], [4.8, 5.0, 0.42, 5.0], 2)
    elif name == "mountaincar":
        return SimpleEnv(gym.make("MountainCar-v0"),
                         [-1.2, -0.07], [0.6, 0.07], 3)
    elif name == "frozenlake":
        env = gym.make("FrozenLake-v1", is_slippery=True)
        class FLWrap:
            def __init__(self, e):
                self.e = e; self.n_actions = 4
            def reset(self):
                obs, _ = self.e.reset()
                return np.array([(obs % 4) / 3.0, (obs // 4) / 3.0], dtype=np.float32)
            def step(self, a):
                obs, r, t, tr, info = self.e.step(a)
                return np.array([(obs % 4) / 3.0, (obs // 4) / 3.0], dtype=np.float32), r, t or tr, info
        return FLWrap(env)
    raise ValueError(f"Unknown env: {name}")


# ── All subsystems that can be toggled ─────────────────────────────────

ALL_SUBSYSTEMS = [
    "curiosity", "world_model", "dreams", "meta_controller",
    "causal_model", "dead_end_detector", "rehearsal", "surprise_tracker",
    "skill_library", "entropy_monitor", "attribution", "stage_classifier",
    "counterfactual", "hippocampus_store", "threat_gating",
    "dream_action_bias", "probe_runner",
]

# ── Run one configuration ──────────────────────────────────────────────

def run_config(
    config_name: str,
    enabled: Dict[str, bool],
    env_name: str,
    env_config: dict,
    n_episodes: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run WholeBrain with a specific config and return results."""
    from brain.orchestrator import WholeBrain

    result = {
        "config_name": config_name,
        "env": env_name,
        "enabled_systems": {k: v for k, v in enabled.items() if not v},  # disabled only
        "status": "UNKNOWN",
        "error": None,
        "init_errors": {},
        "episodes": 0,
        "all_rewards": [],
        "avg_reward": 0.0,
        "early_avg": 0.0,
        "late_avg": 0.0,
        "learning_slope": 0.0,
        "wall_time_sec": 0.0,
        "ms_per_step": 0.0,
        "total_steps": 0,
        "profiler": {},
        "diagnostic": {},
    }

    # ── Init brain ──
    t_init_start = time.time()
    try:
        brain = WholeBrain(
            n_features=env_config["n_features"],
            n_actions=env_config["n_actions"],
            session_name=f"ablation_{config_name}_{env_name}",
            enable_logging=False,
            use_torch=False,  # Use numpy DQN for consistency
            enabled_systems=enabled,
        )
        init_time = time.time() - t_init_start
        result["init_errors"] = dict(brain._init_errors)
        if brain._init_errors:
            print(f"    ⚠ Init errors: {brain._init_errors}")
    except Exception as e:
        result["status"] = "INIT_FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        print(f"    ❌ Brain init failed: {e}")
        traceback.print_exc()
        return result

    # ── Run episodes ──
    all_rewards = []
    total_steps = 0
    step_times = []  # ms per step
    t_start = time.time()
    
    try:
        for ep in range(n_episodes):
            env = make_env(env_name)
            obs = env.reset()
            ep_reward = 0.0
            action = 0

            for step in range(env_config["max_steps"]):
                t_step = time.perf_counter()
                brain_result = brain.step(
                    obs=obs,
                    prev_action=action,
                    reward=ep_reward if step == 0 else reward,
                    done=False,
                )
                step_times.append((time.perf_counter() - t_step) * 1000)
                
                action = brain_result.get("action", 0)
                action = max(0, min(action, env_config["n_actions"] - 1))

                obs, reward, done, info = env.step(action)
                ep_reward += reward
                total_steps += 1

                if done:
                    brain.step(obs=obs, prev_action=action, reward=reward, done=True)
                    break

            all_rewards.append(ep_reward)

            if verbose or (ep + 1) % max(1, n_episodes // 5) == 0:
                recent = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards)
                ms = np.mean(step_times[-100:]) if step_times else 0
                print(f"      Ep {ep+1:3d}: reward={ep_reward:7.2f}  avg10={recent:7.2f}  {ms:.1f}ms/step")

    except Exception as e:
        result["status"] = "RUNTIME_FAIL"
        result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        print(f"    ❌ Runtime error at ep {len(all_rewards)}: {e}")
        traceback.print_exc()
        if not all_rewards:
            try: brain.close()
            except: pass
            return result

    elapsed = time.time() - t_start

    # ── Analyze ──
    result["episodes"] = len(all_rewards)
    result["all_rewards"] = [float(r) for r in all_rewards]
    result["avg_reward"] = float(np.mean(all_rewards))
    result["total_steps"] = total_steps
    result["wall_time_sec"] = elapsed
    result["ms_per_step"] = float(np.mean(step_times)) if step_times else 0.0

    split = max(1, len(all_rewards) // 3)
    result["early_avg"] = float(np.mean(all_rewards[:split]))
    result["late_avg"] = float(np.mean(all_rewards[-split:]))

    # Learning slope (linear regression on rewards)
    if len(all_rewards) > 10:
        x = np.arange(len(all_rewards))
        slope, _ = np.polyfit(x, all_rewards, 1)
        result["learning_slope"] = float(slope)

    # Status classification
    if result["error"]:
        pass
    elif result["late_avg"] > result["early_avg"] * 1.1 and result["learning_slope"] > 0:
        result["status"] = "LEARNING"
    elif result["late_avg"] > result["early_avg"]:
        result["status"] = "SLIGHT_IMPROVEMENT"
    else:
        result["status"] = "NO_LEARNING"

    # Profiler + diagnostic info
    try:
        result["profiler"] = brain.profiler.report()
        result["diagnostic"] = brain.get_diagnostic_info()
    except:
        pass

    try:
        brain.close()
    except:
        pass

    return result


# ── Random Baseline ──────────────────────────────────────────────────

def run_random_baseline(env_name: str, config: dict, n_episodes: int) -> Dict[str, Any]:
    """Run random agent baseline."""
    rewards = []
    for _ in range(n_episodes):
        env = make_env(env_name)
        env.reset()
        ep_reward = 0.0
        for step in range(config["max_steps"]):
            action = np.random.randint(config["n_actions"])
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done: break
        rewards.append(ep_reward)
    return {
        "config_name": "random",
        "env": env_name,
        "avg_reward": float(np.mean(rewards)),
        "all_rewards": [float(r) for r in rewards],
        "status": "BASELINE",
    }


# ── Generate Configs ─────────────────────────────────────────────────

def gen_subtract_configs() -> List[Dict[str, Any]]:
    """Full brain, then remove one subsystem at a time."""
    configs = [{"name": "full", "enabled": {s: True for s in ALL_SUBSYSTEMS}}]
    for sys_name in ALL_SUBSYSTEMS:
        enabled = {s: True for s in ALL_SUBSYSTEMS}
        enabled[sys_name] = False
        configs.append({"name": f"full_minus_{sys_name}", "enabled": enabled})
    return configs


def gen_add_configs() -> List[Dict[str, Any]]:
    """Bare DQN, then add one subsystem at a time."""
    bare = {s: False for s in ALL_SUBSYSTEMS}
    configs = [{"name": "bare_dqn", "enabled": dict(bare)}]
    for sys_name in ALL_SUBSYSTEMS:
        enabled = dict(bare)
        enabled[sys_name] = True
        configs.append({"name": f"bare_plus_{sys_name}", "enabled": enabled})
    return configs


# ── Generate Report ──────────────────────────────────────────────────

def generate_report(results: List[Dict], baselines: Dict[str, Dict], out_dir: Path) -> str:
    """Generate markdown diagnostic report."""
    lines = [
        "# WholeBrain Ablation Test Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Group by environment
    by_env = {}
    for r in results:
        env = r["env"]
        if env not in by_env:
            by_env[env] = []
        by_env[env].append(r)

    for env_name, env_results in by_env.items():
        baseline = baselines.get(env_name, {})
        baseline_avg = baseline.get("avg_reward", 0.0)

        lines.append(f"\n## {env_name.upper()}")
        lines.append(f"\nRandom baseline: **{baseline_avg:.2f}**\n")

        # Results table
        lines.append("| Config | Status | Avg Reward | Early | Late | Slope | ms/step | Init Errors |")
        lines.append("|--------|--------|-----------|-------|------|-------|---------|-------------|")

        for r in sorted(env_results, key=lambda x: x["avg_reward"], reverse=True):
            errors = ", ".join(r.get("init_errors", {}).keys()) or "—"
            delta = r["avg_reward"] - baseline_avg
            status_icon = {
                "LEARNING": "✅",
                "SLIGHT_IMPROVEMENT": "⚠️",
                "NO_LEARNING": "❌",
                "INIT_FAIL": "💥",
                "RUNTIME_FAIL": "💥",
                "BASELINE": "📊",
            }.get(r["status"], "❓")
            lines.append(
                f"| {r['config_name'][:30]} | {status_icon} {r['status']} | "
                f"{r['avg_reward']:.2f} ({delta:+.2f}) | "
                f"{r.get('early_avg', 0):.2f} | {r.get('late_avg', 0):.2f} | "
                f"{r.get('learning_slope', 0):.4f} | "
                f"{r.get('ms_per_step', 0):.1f} | {errors} |"
            )

        # Highlight key findings
        lines.append("\n### Key Findings\n")
        
        # Find configs where removal IMPROVED things
        full_result = next((r for r in env_results if r["config_name"] == "full"), None)
        if full_result:
            full_avg = full_result["avg_reward"]
            for r in env_results:
                if r["config_name"].startswith("full_minus_"):
                    removed = r["config_name"].replace("full_minus_", "")
                    if r["avg_reward"] > full_avg * 1.05:
                        lines.append(f"- 🟢 **Removing `{removed}` IMPROVED performance** "
                                   f"({full_avg:.2f} → {r['avg_reward']:.2f})")
                    elif r["avg_reward"] < full_avg * 0.95:
                        lines.append(f"- 🔴 **Removing `{removed}` HARMED performance** "
                                   f"({full_avg:.2f} → {r['avg_reward']:.2f})")
                    elif r.get("ms_per_step", 0) < full_result.get("ms_per_step", 999) * 0.8:
                        speedup = (1 - r["ms_per_step"] / max(full_result["ms_per_step"], 0.01)) * 100
                        lines.append(f"- ⚡ **Removing `{removed}` gave {speedup:.0f}% speedup** "
                                   f"with no performance change")

        # Errors section
        error_configs = [r for r in env_results if r.get("init_errors")]
        if error_configs:
            lines.append("\n### Init Errors\n")
            for r in error_configs:
                for sys_name, err in r["init_errors"].items():
                    lines.append(f"- **{r['config_name']}** → `{sys_name}`: `{err}`")

        runtime_errors = [r for r in env_results if r.get("error")]
        if runtime_errors:
            lines.append("\n### Runtime Errors\n")
            for r in runtime_errors:
                lines.append(f"- **{r['config_name']}**: `{r['error'][:200]}`")

    # Timing summary (across all envs)
    lines.append("\n## Compute Cost Summary\n")
    lines.append("| Config | ms/step | Total Steps | Wall Time (s) |")
    lines.append("|--------|---------|-------------|---------------|")
    seen = set()
    for r in sorted(results, key=lambda x: x.get("ms_per_step", 0)):
        if r["config_name"] in seen:
            continue
        seen.add(r["config_name"])
        lines.append(
            f"| {r['config_name'][:35]} | {r.get('ms_per_step', 0):.1f} | "
            f"{r.get('total_steps', 0)} | {r.get('wall_time_sec', 0):.1f} |"
        )

    report = "\n".join(lines)

    # Save report
    report_path = out_dir / "ablation_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n📊 Report saved to: {report_path}")

    return report


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WholeBrain Ablation Tester")
    parser.add_argument("--mode", choices=["subtract", "add", "custom", "both"],
                        default="subtract",
                        help="subtract=remove one at a time (default), "
                             "add=bare DQN + one, both=both sweeps")
    parser.add_argument("--envs", nargs="+", default=["cartpole", "gridworld"],
                        choices=list(ENVS.keys()),
                        help="Environments to test (default: cartpole gridworld)")
    parser.add_argument("--episodes", type=int, default=200,
                        help="Episodes per config (default: 200)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 10 episodes per config")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disable", type=str, default="",
                        help="Comma-separated systems to disable (custom mode)")
    parser.add_argument("--enable", type=str, default="",
                        help="Comma-separated systems to enable (custom mode)")
    parser.add_argument("--bare", action="store_true",
                        help="Start from bare DQN (custom mode)")
    parser.add_argument("--out-dir", type=str, default="ablation_results",
                        help="Output directory for results")
    args = parser.parse_args()

    n_episodes = 10 if args.quick else args.episodes
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Generate configs ──
    if args.mode == "subtract":
        configs = gen_subtract_configs()
    elif args.mode == "add":
        configs = gen_add_configs()
    elif args.mode == "both":
        configs = gen_subtract_configs() + gen_add_configs()
    elif args.mode == "custom":
        if args.bare:
            enabled = {s: False for s in ALL_SUBSYSTEMS}
        else:
            enabled = {s: True for s in ALL_SUBSYSTEMS}
        for s in args.disable.split(","):
            s = s.strip()
            if s: enabled[s] = False
        for s in args.enable.split(","):
            s = s.strip()
            if s: enabled[s] = True
        configs = [{"name": "custom", "enabled": enabled}]

    total_runs = len(configs) * len(args.envs)
    print("=" * 70)
    print(f"  WHOLEBRAIN ABLATION TEST — {args.mode.upper()} mode")
    print(f"  Configs: {len(configs)} × Envs: {len(args.envs)} = {total_runs} runs")
    print(f"  Episodes per run: {n_episodes}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # ── Run baselines ──
    baselines = {}
    for env_name in args.envs:
        print(f"\n  📊 Running random baseline for {env_name}...")
        baselines[env_name] = run_random_baseline(env_name, ENVS[env_name], min(n_episodes, 50))
        print(f"     Random avg: {baselines[env_name]['avg_reward']:.2f}")

    # ── Run all configs ──
    all_results = []
    run_idx = 0
    for config in configs:
        for env_name in args.envs:
            run_idx += 1
            print(f"\n{'─' * 70}")
            print(f"  [{run_idx}/{total_runs}] {config['name']} on {env_name}")
            disabled = [k for k, v in config["enabled"].items() if not v]
            if disabled:
                print(f"    Disabled: {', '.join(disabled)}")
            else:
                print(f"    All systems enabled")
            print(f"{'─' * 70}")

            result = run_config(
                config_name=config["name"],
                enabled=config["enabled"],
                env_name=env_name,
                env_config=ENVS[env_name],
                n_episodes=n_episodes,
                verbose=args.verbose,
            )
            all_results.append(result)

            # Print quick summary
            print(f"\n    → {result['status']}: avg={result['avg_reward']:.2f} "
                  f"({result['early_avg']:.2f}→{result['late_avg']:.2f}) "
                  f"{result.get('ms_per_step', 0):.1f}ms/step")

    # ── Save raw results ──
    results_path = out_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "mode": args.mode,
            "n_episodes": n_episodes,
            "envs": args.envs,
            "baselines": baselines,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\n💾 Raw results saved to: {results_path}")

    # ── Generate report ──
    report = generate_report(all_results, baselines, out_dir)
    print(f"\n{report}")


if __name__ == "__main__":
    main()
