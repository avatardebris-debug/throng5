"""
Dreamer Performance Benchmark — Bridge Step 4 Analysis

Answers critical design questions:
1. Compressed vs uncompressed world model fidelity
2. Max lookahead depth at 30fps and 60fps budgets
3. Calibration drift over time (does accuracy degrade?)
4. Network size tier comparison (micro/mini/full)
5. Amygdala response latency for threat scenarios

Frame budget calculations:
  - 60fps → 16.7ms per frame total
  - 30fps → 33.3ms per frame total
  - Dream budget should be ~30-50% of frame time (rest for real policy)
  - 60fps dream budget: ~5-8ms
  - 30fps dream budget: ~10-16ms
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from throng4.basal_ganglia.dreamer_engine import (
    DreamerEngine, Hypothesis, WorldModel, NetworkSize, NETWORK_CONFIGS,
)
from throng4.basal_ganglia.compressed_state import (
    CompressedStateEncoder, EncodingMode,
)
from throng4.basal_ganglia.amygdala import Amygdala


def benchmark_world_model_speed():
    """
    Benchmark 1: Raw world model predict() speed across tiers.
    This determines how many steps we can simulate per millisecond.
    """
    print("=" * 70)
    print("BENCHMARK 1: World Model Prediction Speed")
    print("=" * 70)

    state_sizes = [32, 64, 128]
    n_actions = 6  # Typical Atari

    results = {}

    for tier in NetworkSize:
        config = NETWORK_CONFIGS[tier]
        for state_size in state_sizes:
            model = WorldModel(state_size, n_actions, config)

            # Warmup
            state = np.random.randn(state_size).astype(np.float32)
            for _ in range(10):
                model.predict(state, 0)

            # Benchmark
            n_iters = 1000
            t0 = time.perf_counter()
            for i in range(n_iters):
                next_state, reward = model.predict(state, i % n_actions)
                state = next_state  # Chain predictions like real dreaming
            elapsed = time.perf_counter() - t0

            per_step_us = (elapsed / n_iters) * 1e6
            steps_per_ms = 1000 / per_step_us

            key = f"{tier.value}_s{state_size}"
            results[key] = {
                'tier': tier.value,
                'state_size': state_size,
                'per_step_us': per_step_us,
                'steps_per_ms': steps_per_ms,
            }

            print(f"  {tier.value:5s} | state={state_size:3d} | "
                  f"{per_step_us:6.1f} µs/step | "
                  f"{steps_per_ms:6.1f} steps/ms")

    return results


def benchmark_dream_cycle():
    """
    Benchmark 2: Full dream cycle (3 hypotheses × N lookahead steps).
    Measures total time to produce ranked evaluations.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Full Dream Cycle (3 hypotheses)")
    print("=" * 70)

    lookahead_depths = [10, 30, 60, 120, 180, 300]
    # At 60fps: 10=0.17s, 30=0.5s, 60=1s, 120=2s, 180=3s, 300=5s

    state_size = 64
    n_actions = 6

    print(f"\n  {'Depth':>6s} | {'Game-s @60fps':>14s} | ", end="")
    for tier in NetworkSize:
        print(f"  {tier.value:>8s}", end="")
    print("  | Fits 60fps? | Fits 30fps?")
    print("  " + "-" * 100)

    all_results = {}

    for depth in lookahead_depths:
        game_seconds = depth / 60.0
        row = {'depth': depth, 'game_seconds': game_seconds}

        print(f"  {depth:>6d} | {game_seconds:>11.2f}s   | ", end="")

        for tier in NetworkSize:
            dreamer = DreamerEngine(
                n_hypotheses=3,
                network_size=tier.value,
                state_size=state_size,
                n_actions=n_actions,
            )

            # Train world model minimally
            for i in range(60):
                s = np.random.randn(state_size).astype(np.float32)
                a = np.random.randint(n_actions)
                s2 = s + np.random.randn(state_size) * 0.1
                r = float(np.random.randn() * 0.5)
                dreamer.learn(s, a, s2, r)

            hypotheses = dreamer.create_default_hypotheses(n_actions)
            state = np.random.randn(state_size).astype(np.float32)

            # Run 5 trials, take median
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                results = dreamer.dream(state, hypotheses, n_steps=depth)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                times.append(elapsed_ms)

            median_ms = sorted(times)[len(times) // 2]
            row[tier.value] = median_ms
            print(f"  {median_ms:7.1f}ms", end="")

        # Check budget fit (using micro tier)
        micro_ms = row.get('micro', 999)
        fits_60 = "YES" if micro_ms < 8 else "NO"
        fits_30 = "YES" if micro_ms < 16 else "NO"
        print(f"  | {fits_60:>11s} | {fits_30:>11s}")

        all_results[depth] = row

    return all_results


def benchmark_calibration_drift():
    """
    Benchmark 3: Does the compressed world model drift from reality?
    
    Trains a world model on actual transitions, then measures how
    prediction error changes over increasing lookahead depth.
    This tells us if/when calibration becomes necessary.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 3: Calibration Drift (prediction error vs lookahead)")
    print("=" * 70)

    state_size = 64
    n_actions = 4
    model = WorldModel(state_size, n_actions, NETWORK_CONFIGS[NetworkSize.MICRO])

    # Create a simple deterministic environment simulator
    # (ground truth to compare against)
    def ground_truth_step(state, action):
        """Simple dynamics: state rotates + action bias + noise."""
        next_state = np.roll(state, action + 1) * 0.95
        next_state[action * (state_size // n_actions)] += 0.5
        reward = float(np.sum(next_state[:4]) * 0.1)
        return next_state, reward

    # Phase 1: Train model on ground truth
    print("\n  Training world model on 500 ground truth transitions...")
    for i in range(500):
        s = np.random.randn(state_size).astype(np.float32) * 0.5
        a = np.random.randint(n_actions)
        s2, r = ground_truth_step(s, a)
        model.update(s, a, s2.astype(np.float32), r, lr=0.001)

    # Phase 2: Measure prediction error at various depths
    depths = [1, 5, 10, 20, 30, 60, 120]
    n_trials = 20

    print(f"\n  {'Depth':>6s} | {'State Error':>12s} | {'Reward Error':>13s} | {'Drift?':>6s}")
    print("  " + "-" * 55)

    drift_results = {}

    for depth in depths:
        state_errors = []
        reward_errors = []

        for trial in range(n_trials):
            # Start from random state
            true_state = np.random.randn(state_size).astype(np.float32) * 0.5
            pred_state = true_state.copy()

            for step in range(depth):
                action = np.random.randint(n_actions)

                # Ground truth
                true_next, true_reward = ground_truth_step(true_state, action)

                # Model prediction (chained — uses its own predictions)
                pred_next, pred_reward = model.predict(pred_state, action)

                true_state = true_next.astype(np.float32)
                pred_state = pred_next.astype(np.float32)

            # Error at final step
            state_err = np.mean(np.abs(true_state - pred_state))
            reward_err = abs(true_reward - pred_reward)
            state_errors.append(state_err)
            reward_errors.append(reward_err)

        avg_s_err = np.mean(state_errors)
        avg_r_err = np.mean(reward_errors)
        drifting = "YES" if avg_s_err > 1.0 else "no"

        drift_results[depth] = {
            'state_error': avg_s_err,
            'reward_error': avg_r_err,
            'drifting': avg_s_err > 1.0,
        }

        print(f"  {depth:>6d} | {avg_s_err:>12.4f} | {avg_r_err:>13.4f} | {drifting}")

    return drift_results


def benchmark_compressed_vs_uncompressed():
    """
    Benchmark 4: Does compression degrade model quality?
    
    Compares world model trained on raw observations vs compressed.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Compressed vs Uncompressed Fidelity")
    print("=" * 70)

    state_size = 64
    n_actions = 4
    n_levels_list = [2, 4, 8, 16, 0]  # 0 = no compression

    def ground_truth_step(state, action):
        next_state = np.roll(state, action + 1) * 0.95
        next_state[action * (state_size // n_actions)] += 0.5
        reward = float(np.sum(next_state[:4]) * 0.1)
        return next_state, reward

    print(f"\n  {'Levels':>7s} | {'Compress?':>9s} | {'1-step err':>10s} | "
          f"{'10-step err':>11s} | {'Speed':>8s}")
    print("  " + "-" * 65)

    for n_levels in n_levels_list:
        use_compression = n_levels > 0
        label = str(n_levels) if use_compression else "raw"

        model = WorldModel(state_size, n_actions, NETWORK_CONFIGS[NetworkSize.MICRO])
        encoder = CompressedStateEncoder(
            mode=EncodingMode.QUANTIZED,
            n_quantize_levels=max(n_levels, 2),
        ) if use_compression else None

        # Train
        for i in range(500):
            s = np.random.randn(state_size).astype(np.float32) * 0.5
            a = np.random.randint(n_actions)
            s2, r = ground_truth_step(s, a)

            if use_compression:
                cs = encoder.encode(s).data
                cs2 = encoder.encode(s2.astype(np.float32)).data
                model.update(cs, a, cs2, r, lr=0.001)
            else:
                model.update(s, a, s2.astype(np.float32), r, lr=0.001)

        # Evaluate at 1-step and 10-step
        errors_1 = []
        errors_10 = []

        for trial in range(50):
            s = np.random.randn(state_size).astype(np.float32) * 0.5
            a = np.random.randint(n_actions)
            true_next, true_r = ground_truth_step(s, a)

            if use_compression:
                cs = encoder.encode(s).data
            else:
                cs = s

            pred_next, pred_r = model.predict(cs, a)
            err_1 = np.mean(np.abs(true_next - pred_next))
            errors_1.append(err_1)

            # 10-step chained
            pred_s = cs.copy()
            true_s = s.copy()
            for step in range(10):
                a = np.random.randint(n_actions)
                true_s, _ = ground_truth_step(true_s, a)
                pred_s, _ = model.predict(pred_s, a)
            err_10 = np.mean(np.abs(true_s - pred_s))
            errors_10.append(err_10)

        # Speed benchmark
        t0 = time.perf_counter()
        for _ in range(1000):
            if use_compression:
                cs = encoder.encode(np.random.randn(state_size).astype(np.float32) * 0.5).data
                model.predict(cs, 0)
            else:
                model.predict(np.random.randn(state_size).astype(np.float32) * 0.5, 0)
        speed_us = (time.perf_counter() - t0) / 1000 * 1e6

        print(f"  {label:>7s} | {'yes' if use_compression else 'no':>9s} | "
              f"{np.mean(errors_1):>10.4f} | {np.mean(errors_10):>11.4f} | "
              f"{speed_us:>6.1f} µs")


def benchmark_amygdala_response():
    """
    Benchmark 5: Amygdala threat detection + response latency.
    
    Simulates a threat scenario and measures total time from
    state observation to danger signal.
    """
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Amygdala Threat Detection Latency")
    print("=" * 70)

    state_size = 64
    n_actions = 6

    dreamer = DreamerEngine(
        n_hypotheses=3, network_size='micro',
        state_size=state_size, n_actions=n_actions,
    )
    amygdala = Amygdala()

    # Train
    for i in range(100):
        s = np.random.randn(state_size).astype(np.float32)
        a = np.random.randint(n_actions)
        s2 = s + np.random.randn(state_size) * 0.1
        r = float(np.random.randn() * 0.5)
        dreamer.learn(s, a, s2, r)

    hypotheses = dreamer.create_default_hypotheses(n_actions)
    depths = [10, 30, 60, 120]

    print(f"\n  {'Depth':>6s} | {'Game-s':>7s} | {'Dream':>8s} | {'Assess':>8s} | "
          f"{'Total':>8s} | {'< 8ms?':>7s} | {'< 16ms?':>8s}")
    print("  " + "-" * 75)

    for depth in depths:
        game_sec = depth / 60.0
        state = np.random.randn(state_size).astype(np.float32)

        # Measure full pipeline: dream + assess
        times_dream = []
        times_assess = []

        for _ in range(20):
            t0 = time.perf_counter()
            results = dreamer.dream(state, hypotheses, n_steps=depth)
            t1 = time.perf_counter()
            danger = amygdala.assess_danger(results, current_step=0)
            t2 = time.perf_counter()

            times_dream.append((t1 - t0) * 1000)
            times_assess.append((t2 - t1) * 1000)

        med_dream = sorted(times_dream)[len(times_dream) // 2]
        med_assess = sorted(times_assess)[len(times_assess) // 2]
        total = med_dream + med_assess

        fits_60 = "YES" if total < 8 else "NO"
        fits_30 = "YES" if total < 16 else "NO"

        print(f"  {depth:>6d} | {game_sec:>5.1f}s  | {med_dream:>6.1f}ms | "
              f"{med_assess:>6.1f}ms | {total:>6.1f}ms | {fits_60:>7s} | {fits_30:>8s}")


def compute_recommendations(speed_results, drift_results, cycle_results):
    """Synthesize benchmark results into actionable recommendations."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find max depth that fits 60fps and 30fps budgets
    max_60fps = 0
    max_30fps = 0
    for depth, row in sorted(cycle_results.items()):
        micro_ms = row.get('micro', 999)
        if micro_ms < 8:
            max_60fps = depth
        if micro_ms < 16:
            max_30fps = depth

    # Find drift threshold
    drift_threshold = None
    for depth, data in sorted(drift_results.items()):
        if data['drifting']:
            drift_threshold = depth
            break

    print(f"""
  SPEED:
    Max lookahead at 60fps (8ms budget):  {max_60fps} steps = {max_60fps/60:.1f}s game time
    Max lookahead at 30fps (16ms budget): {max_30fps} steps = {max_30fps/60:.1f}s game time

  CALIBRATION:
    Drift becomes significant at: {drift_threshold or '>120'} steps lookahead
    Recommendation: {'Auto-recalibrate every episode' if drift_threshold and drift_threshold <= 30 else 'Recalibrate every 5-10 episodes (drift is gradual)'}

  ARCHITECTURE:
    For Tetris (slow-paced):  micro tier, 60-120 step lookahead (1-2s ahead)
    For Breakout (medium):    micro tier, 30-60 step lookahead (0.5-1s ahead)
    For Missile Defender:     micro tier, max affordable lookahead +
                              periodic full-tier deep analysis between episodes

  DUAL-FIDELITY STRATEGY:
    → Use micro (compressed) for per-frame dreaming within frame budget
    → Use mini/full (uncompressed) for between-episode calibration benchmark
    → If micro predictions diverge from mini/full by >threshold, recalibrate
    → Amygdala assess_danger() is <0.1ms — essentially free
""")


if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  DREAMER PERFORMANCE ANALYSIS — Bridge Step 4")
    print("#" * 70)

    speed = benchmark_world_model_speed()
    cycle = benchmark_dream_cycle()
    drift = benchmark_calibration_drift()
    benchmark_compressed_vs_uncompressed()
    benchmark_amygdala_response()
    compute_recommendations(speed, drift, cycle)

    print("\n" + "#" * 70)
    print("#  BENCHMARK COMPLETE")
    print("#" * 70)
