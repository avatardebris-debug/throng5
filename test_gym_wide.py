"""
test_gym_wide.py — Wide testing of WholeBrain across classic Gymnasium envs.

Tests that the brain can:
  1. Initialize with different obs/action dimensions
  2. Run episodes without crashing
  3. Select valid actions
  4. Show ANY learning signal (reward improvement over random)

Usage:
  python test_gym_wide.py                    # All envs, 50 episodes each
  python test_gym_wide.py --env cartpole     # Single env
  python test_gym_wide.py --episodes 200     # More episodes
  python test_gym_wide.py --quick            # 10 episodes each (smoke test)
"""

import argparse
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, ".")

# ── Standalone environment wrappers (no throng4 dependency) ──────────

import gymnasium as gym


class SimpleEnv:
    """Minimal env wrapper — just obs, step, reset."""
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
    """Simple 5x5 grid — no gym dependency."""
    def __init__(self):
        self.size = 5
        self.goal = (4, 4)
        self.n_actions = 4

    def reset(self):
        self.pos = (0, 0)
        return self._obs()

    def step(self, action):
        x, y = self.pos
        if action == 0: y = max(0, y - 1)      # up
        elif action == 1: y = min(4, y + 1)     # down
        elif action == 2: x = max(0, x - 1)     # left
        elif action == 3: x = min(4, x + 1)     # right
        self.pos = (x, y)
        if self.pos == self.goal:
            return self._obs(), 1.0, True, {}
        return self._obs(), -0.01, False, {}

    def _obs(self):
        return np.array([self.pos[0] / 4.0, self.pos[1] / 4.0], dtype=np.float32)


def make_env(name: str):
    """Create an environment by name."""
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
        # Wrap discrete obs → 2D position
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


ENVS = {
    "gridworld": {
        "n_features": 2,
        "n_actions": 4,
        "max_steps": 100,
        "description": "5x5 grid, goal at (4,4), dense reward",
    },
    "cartpole": {
        "n_features": 4,
        "n_actions": 2,
        "max_steps": 500,
        "description": "Balance pole, 2 actions, dense reward",
    },
    "mountaincar": {
        "n_features": 2,
        "n_actions": 3,
        "max_steps": 200,
        "description": "Reach flag, 3 actions, sparse reward",
    },
    "frozenlake": {
        "n_features": 2,
        "n_actions": 4,
        "max_steps": 100,
        "description": "4x4 slippery grid with holes, sparse reward",
    },
}


def run_random_baseline(env_name: str, config: dict, n_episodes: int, max_steps: int):
    """Run random agent to establish baseline."""
    rewards = []
    n_actions = config["n_actions"]
    for _ in range(n_episodes):
        env = make_env(env_name)
        env.reset()
        ep_reward = 0.0
        for step in range(max_steps):
            action = np.random.randint(n_actions)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)


def run_brain_test(env_name: str, config: dict, n_episodes: int, verbose: bool = False):
    """Run WholeBrain on one environment and return results."""
    from brain.orchestrator import WholeBrain

    print(f"\n{'─'*60}")
    print(f"  {env_name.upper()}: {config['description']}")
    print(f"  features={config['n_features']}, actions={config['n_actions']}, "
          f"episodes={n_episodes}")
    print(f"{'─'*60}")

    result = {
        "env": env_name,
        "status": "UNKNOWN",
        "error": None,
        "episodes": 0,
        "avg_reward": 0.0,
        "best_reward": float("-inf"),
        "random_baseline": 0.0,
        "improvement_pct": 0.0,
        "early_avg": 0.0,
        "late_avg": 0.0,
        "steps_per_sec": 0.0,
    }

    # ── 1. Initialize adapter ──
    try:
        adapter = make_env(env_name)
        print(f"  ✅ Adapter loaded")
    except Exception as e:
        result["status"] = "ADAPTER_FAIL"
        result["error"] = str(e)
        print(f"  ❌ Adapter failed: {e}")
        return result

    # ── 2. Random baseline ──
    try:
        baseline_mean, baseline_std = run_random_baseline(
            env_name, config, min(n_episodes, 20), config["max_steps"]
        )
        result["random_baseline"] = baseline_mean
        print(f"  Random baseline: {baseline_mean:.2f} ± {baseline_std:.2f}")
    except Exception as e:
        print(f"  ⚠ Random baseline failed: {e}")

    # ── 3. Initialize brain ──
    try:
        brain = WholeBrain(
            n_features=config["n_features"],
            n_actions=config["n_actions"],
            session_name=f"test_{env_name}",
            enable_logging=False,
            use_torch=True,
        )
        print(f"  ✅ WholeBrain initialized")
    except Exception as e:
        result["status"] = "BRAIN_INIT_FAIL"
        result["error"] = str(e)
        print(f"  ❌ Brain init failed: {e}")
        traceback.print_exc()
        return result

    # ── 4. Run episodes ──
    all_rewards = []
    total_steps = 0
    t_start = time.time()

    try:
        for ep in range(n_episodes):
            obs = adapter.reset()
            ep_reward = 0.0
            action = 0

            for step in range(config["max_steps"]):
                # Brain step
                brain_result = brain.step(
                    obs=obs,
                    prev_action=action,
                    reward=ep_reward if step == 0 else reward,
                    done=False,
                )
                action = brain_result.get("action", 0)

                # Clamp action to valid range
                action = max(0, min(action, config["n_actions"] - 1))

                # Env step
                obs, reward, done, info = adapter.step(action)
                ep_reward += reward
                total_steps += 1

                if done:
                    # Notify brain of episode end
                    brain.step(obs=obs, prev_action=action, reward=reward, done=True)
                    break

            all_rewards.append(ep_reward)

            if verbose or (ep + 1) % max(1, n_episodes // 5) == 0:
                recent = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards)
                print(f"    Ep {ep+1:3d}: reward={ep_reward:7.2f}  avg(last10)={recent:7.2f}")

    except Exception as e:
        result["status"] = "RUNTIME_FAIL"
        result["error"] = str(e)
        print(f"  ❌ Runtime error at ep {len(all_rewards)}: {e}")
        traceback.print_exc()
        if not all_rewards:
            return result

    elapsed = time.time() - t_start

    # ── 5. Analyze results ──
    result["episodes"] = len(all_rewards)
    result["avg_reward"] = np.mean(all_rewards)
    result["best_reward"] = max(all_rewards)
    result["steps_per_sec"] = total_steps / max(elapsed, 0.01)

    # Early vs late comparison (learning signal)
    split = max(1, len(all_rewards) // 3)
    result["early_avg"] = np.mean(all_rewards[:split])
    result["late_avg"] = np.mean(all_rewards[-split:])

    if result["random_baseline"] != 0:
        result["improvement_pct"] = (
            (result["avg_reward"] - result["random_baseline"]) /
            max(abs(result["random_baseline"]), 0.01) * 100
        )

    # Determine status
    if result["error"]:
        pass  # Already set
    elif result["late_avg"] > result["early_avg"] * 1.1:
        result["status"] = "LEARNING"
    elif result["avg_reward"] > result["random_baseline"]:
        result["status"] = "BETTER_THAN_RANDOM"
    else:
        result["status"] = "NO_LEARNING"

    print(f"\n  Results:")
    print(f"    Episodes: {result['episodes']}")
    print(f"    Avg reward: {result['avg_reward']:.2f} (random: {result['random_baseline']:.2f})")
    print(f"    Best reward: {result['best_reward']:.2f}")
    print(f"    Early avg: {result['early_avg']:.2f} → Late avg: {result['late_avg']:.2f}")
    print(f"    Speed: {result['steps_per_sec']:.0f} steps/sec")
    print(f"    Status: {result['status']}")

    try:
        brain.close()
    except:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Wide brain test across Gym envs")
    parser.add_argument("--env", choices=list(ENVS.keys()), default=None,
                        help="Test single env (default: all)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per env (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (10 episodes)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    n_episodes = 10 if args.quick else args.episodes
    envs_to_test = {args.env: ENVS[args.env]} if args.env else ENVS

    print("=" * 60)
    print("  THRONG5 — Wide Brain Test")
    print(f"  Environments: {', '.join(envs_to_test.keys())}")
    print(f"  Episodes per env: {n_episodes}")
    print("=" * 60)

    results = {}
    for env_name, config in envs_to_test.items():
        results[env_name] = run_brain_test(env_name, config, n_episodes, args.verbose)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Env':<15} {'Status':<20} {'Avg':<10} {'Random':<10} {'Δ%':<8}")
    print(f"  {'-'*13:<15} {'-'*18:<20} {'-'*8:<10} {'-'*8:<10} {'-'*6:<8}")
    for name, r in results.items():
        status_icon = {
            "LEARNING": "✅",
            "BETTER_THAN_RANDOM": "⚠️",
            "NO_LEARNING": "❌",
            "ADAPTER_FAIL": "💥",
            "BRAIN_INIT_FAIL": "💥",
            "RUNTIME_FAIL": "💥",
        }.get(r["status"], "❓")
        print(f"  {name:<15} {status_icon} {r['status']:<17} "
              f"{r['avg_reward']:<10.2f} {r['random_baseline']:<10.2f} "
              f"{r['improvement_pct']:>+6.1f}%")

    # Overall verdict
    n_pass = sum(1 for r in results.values() if r["status"] in ("LEARNING", "BETTER_THAN_RANDOM"))
    n_fail = sum(1 for r in results.values() if "FAIL" in r["status"])
    n_total = len(results)

    print(f"\n  {n_pass}/{n_total} environments show learning signal")
    if n_fail:
        print(f"  ⚠ {n_fail} environments had errors")
    print("=" * 60)


if __name__ == "__main__":
    main()
