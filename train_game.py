"""
train_game.py — Full end-to-end game training loop.

Wires a Gymnasium environment through the whole brain pipeline:
    Environment → Sensory Cortex → Basal Ganglia → Amygdala →
    Hippocampus → Striatum (TorchDQN) → Motor Cortex → Environment

Usage:
    python train_game.py                                        # CartPole, 500 episodes
    python train_game.py --env ALE/Pong-v5 --torch --episodes 1000
    python train_game.py --env ALE/Pong-v5 --torch --cnn --episodes 500  # CNN encoder!
    python train_game.py --env ALE/Pong-v5 --torch --cnn --fft --episodes 500  # FFT+CNN!
    python train_game.py --env ALE/Pong-v5 --fft --episodes 500  # Pure FFT (no CNN)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
try:
    import ale_py  # Register ALE/Atari environments
    gym.register_envs(ale_py)
except ImportError:
    pass
from brain.orchestrator import WholeBrain
from brain.scaling.brain_state import BrainState, AutoCheckpointer


def preprocess_obs(obs: np.ndarray, n_features: int = 84) -> np.ndarray:
    """
    Convert any observation to a fixed-size feature vector.

    For pixel observations (H, W, C): grayscale → downsample → flatten → project.
    For vector observations: pad/truncate to n_features.
    """
    obs = np.asarray(obs, dtype=np.float32)

    if obs.ndim >= 2:
        # Pixel observation (Atari)
        if obs.ndim == 3 and obs.shape[2] == 3:
            # RGB → grayscale
            gray = np.mean(obs, axis=2)
        else:
            gray = obs.squeeze()

        # Downsample by taking every Nth pixel
        h, w = gray.shape
        target_pixels = n_features * 4  # 336 pixels
        skip = max(1, int(np.sqrt(h * w / target_pixels)))
        downsampled = gray[::skip, ::skip].flatten()

        # Random projection to n_features
        rng = np.random.RandomState(42)
        if len(downsampled) > n_features:
            proj_matrix = rng.randn(len(downsampled), n_features).astype(np.float32)
            proj_matrix /= np.sqrt(len(downsampled))
            features = downsampled @ proj_matrix
        else:
            features = downsampled

    else:
        # Vector observation (CartPole, etc.)
        features = obs.flatten()

    # Pad or truncate to n_features
    if len(features) < n_features:
        features = np.pad(features, (0, n_features - len(features)))
    elif len(features) > n_features:
        features = features[:n_features]

    # Normalize to reasonable range
    std = np.std(features)
    if std > 0:
        features = features / (std * 3)  # Soft normalize
    features = np.clip(features, -5, 5)

    return features.astype(np.float32)


def train(
    env_name: str = "CartPole-v1",
    n_episodes: int = 500,
    n_features: int = 84,
    use_torch: bool = False,
    use_cnn: bool = False,
    use_fft: bool = False,
    render: bool = False,
    report_every: int = 25,
    checkpoint_every: int = 0,
    resume_path: str = "",
) -> None:
    """Run full training loop."""

    # ── Create environment ────────────────────────────────────────────
    render_mode = "human" if render else None
    env = gym.make(env_name, render_mode=render_mode)
    n_actions = env.action_space.n if hasattr(env.action_space, 'n') else 4

    # Detect if pixel environment
    sample_obs, _ = env.reset()
    is_pixel = np.asarray(sample_obs).ndim >= 2
    encoder_name = "cnn" if (use_cnn and is_pixel) else "random_projection"

    print("=" * 60)
    print(f"Training: {env_name}")
    print(f"  Actions: {n_actions}, Features: {n_features}")
    print(f"  Backend: {'PyTorch DQN' if use_torch else 'NumPy DQN'}")
    print(f"  Encoder: {encoder_name}")
    print(f"  Episodes: {n_episodes}")
    print("=" * 60)

    # ── Create brain ──────────────────────────────────────────────────
    brain = WholeBrain(
        n_features=n_features,
        n_actions=n_actions,
        enable_logging=False,
        use_torch=use_torch,
        use_cnn=use_cnn and is_pixel,
        use_fft=use_fft and is_pixel,
    )

    # ── Resume from checkpoint ────────────────────────────────────────
    start_episode = 0
    if resume_path:
        print(f"  Resuming from: {resume_path}")
        info = BrainState.info(resume_path)
        print(f"    Checkpoint: ep={info['episode_count']}, steps={info['step_count']}, "
              f"size={info['size_mb']}MB")
        BrainState.load(resume_path, brain)
        start_episode = brain._episode_count
        print(f"    Restored. Continuing from episode {start_episode}.")

    # ── Checkpointer ──────────────────────────────────────────────────
    checkpointer = None
    if checkpoint_every > 0:
        checkpointer = AutoCheckpointer(
            brain,
            interval_episodes=checkpoint_every,
            keep_last=5,
        )
        print(f"  Checkpointing every {checkpoint_every} episodes")

    backend = brain.striatum.report().get("backend", "numpy")
    sensory_report = brain.sensory.report()
    print(f"  Striatum backend: {backend}")
    print(f"  Sensory encoder:  {sensory_report.get('encoder', '?')}")
    if use_torch:
        print(f"  DQN params: {brain.striatum.report().get('n_params', '?'):,}")
    if use_cnn and 'cnn_params' in sensory_report:
        print(f"  CNN params: {sensory_report['cnn_params']:,}")
    print()

    # ── Training loop ─────────────────────────────────────────────────
    episode_rewards = deque(maxlen=100)
    best_reward = -float("inf")
    start_time = time.time()

    for episode in range(start_episode, n_episodes):
        obs, info = env.reset()
        # CNN mode: pass raw pixels. Non-CNN: preprocess to feature vector
        brain_obs = obs if (use_cnn and is_pixel) else preprocess_obs(obs, n_features)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        prev_action = 0

        while not done and not truncated:
            # Brain step
            result = brain.step(
                obs=brain_obs,
                prev_action=prev_action,
                reward=ep_reward if ep_steps == 0 else reward,
                done=False,
            )
            action = result["action"]

            # Environment step
            obs, reward, done, truncated, info = env.step(action)
            brain_obs = obs if (use_cnn and is_pixel) else preprocess_obs(obs, n_features)
            ep_reward += reward
            ep_steps += 1
            prev_action = action

        # Final done signal to brain
        brain.step(obs=brain_obs, prev_action=prev_action, reward=reward, done=True)

        episode_rewards.append(ep_reward)
        if ep_reward > best_reward:
            best_reward = ep_reward

        # ── Periodic report ───────────────────────────────────────────
        if (episode + 1) % report_every == 0 or episode == 0:
            avg_reward = np.mean(episode_rewards)
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed

            report = brain.striatum.report()
            epsilon = report.get("epsilon", 0)

            print(
                f"  Ep {episode+1:4d}/{n_episodes} | "
                f"Avg: {avg_reward:7.1f} | "
                f"Best: {best_reward:7.1f} | "
                f"ε: {epsilon:.3f} | "
                f"Steps: {brain._step_count:,} | "
                f"{eps_per_sec:.1f} ep/s"
            )
            # Profiler timing
            print(brain.profiler.report_str())

        # ── Auto-checkpoint ───────────────────────────────────────────
        if checkpointer is not None:
            ck = checkpointer.maybe_checkpoint()
            if ck:
                print(f"  [CHECKPOINT] Saved: {ck['filepath']} ({ck['size_mb']}MB)")

    # ── Final summary ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    final_avg = np.mean(episode_rewards)

    print()
    print("=" * 60)
    print(f"Training Complete: {env_name}")
    print("=" * 60)
    print(f"  Episodes:       {n_episodes}")
    print(f"  Total steps:    {brain._step_count:,}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Final avg (100): {final_avg:.1f}")
    print(f"  Best episode:   {best_reward:.1f}")
    print(f"  Episodes/sec:   {n_episodes/elapsed:.1f}")

    # Brain report
    report = brain.report()
    print(f"\n  Brain regions:")
    for rname, rdata in report.items():
        if isinstance(rdata, dict):
            items = []
            for k, v in rdata.items():
                if k in ("processed", "total_updates", "buffer_size"):
                    items.append(f"{k}={v}")
            if items:
                print(f"    {rname:24s} {', '.join(items)}")

    # Striatum detail
    sr = brain.striatum.report()
    print(f"\n  Striatum detail:")
    print(f"    Backend:      {sr.get('backend', 'unknown')}")
    print(f"    Updates:      {sr.get('total_updates', 0):,}")
    print(f"    Buffer:       {sr.get('buffer_size', 0):,}")
    print(f"    Epsilon:      {sr.get('epsilon', 0):.4f}")
    if use_torch:
        print(f"    Params:       {sr.get('n_params', 0):,}")
        print(f"    Avg loss:     {sr.get('avg_loss', 0):.6f}")

    # Sensory detail
    sensory_r = brain.sensory.report()
    print(f"\n  Sensory detail:")
    print(f"    Encoder:      {sensory_r.get('encoder', 'unknown')}")
    if 'cnn_params' in sensory_r:
        print(f"    CNN params:   {sensory_r['cnn_params']:,}")
        print(f"    Frame stack:  {sensory_r.get('frame_stack', 4)}")

    # World model detail
    bg_r = brain.basal_ganglia.report()
    print(f"\n  World Model:")
    print(f"    Has model:    {bg_r.get('has_world_model', False)}")
    print(f"    Confidence:   {bg_r.get('wm_confidence', 0):.3f}")
    print(f"    Train steps:  {bg_r.get('wm_train_steps', 0):,}")
    print(f"    Total dreams: {bg_r.get('total_dreams', 0):,}")
    if bg_r.get('wm_params', 0) > 0:
        print(f"    WM params:    {bg_r['wm_params']:,}")
        print(f"    WM avg loss:  {bg_r.get('wm_avg_loss', 0):.6f}")

    # Curiosity detail
    cur_stats = brain.curiosity.stats()
    print(f"\n  Curiosity:")
    print(f"    Prediction:   {cur_stats.get('prediction_source', 'unknown')}")
    print(f"    Avg pred err: {cur_stats.get('avg_pred_error', 0):.6f}")
    print(f"    Unique states:{cur_stats.get('unique_states', 0):,}")
    print(f"    Visit buckets:{cur_stats.get('visit_buckets', 0):,}")

    # Meta-controller detail
    if brain.meta_controller is not None:
        mc = brain.meta_controller
        print(f"\n  Meta-Controller:")
        print(f"    Learners:     {len(mc._slots)}")
        print(f"    Collapsed:    {mc._is_collapsed}")
        if mc._is_collapsed and mc._locked_learner:
            print(f"    Winner:       {mc._locked_learner}")
        for name, slot in mc._slots.items():
            mark = " << active" if name == brain._active_learner_name else ""
            print(f"    {name:15s} | steps={slot.total_steps:,} | avg={slot.mean_reward:.1f}{mark}")

    # Performance profile
    print(f"\n{brain.profiler.report_str()}")

    # Final checkpoint
    if checkpointer is not None:
        ck = checkpointer.checkpoint()
        print(f"\n  [CHECKPOINT] Final: {ck['filepath']} ({ck['size_mb']}MB)")

    env.close()
    brain.close()

    return final_avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train brain on a game")
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium env name")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--features", type=int, default=84, help="Feature vector size")
    parser.add_argument("--torch", action="store_true", help="Use PyTorch DQN")
    parser.add_argument("--cnn", action="store_true", help="Use CNN encoder for pixels")
    parser.add_argument("--fft", action="store_true", help="Use FFT compression (with or without CNN)")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--report-every", type=int, default=25, help="Report interval")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Checkpoint interval (0=off)")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint file")
    args = parser.parse_args()

    train(
        env_name=args.env,
        n_episodes=args.episodes,
        n_features=args.features,
        use_torch=args.torch,
        use_cnn=args.cnn,
        use_fft=args.fft,
        render=args.render,
        report_every=args.report_every,
        checkpoint_every=args.checkpoint_every,
        resume_path=args.resume,
    )
