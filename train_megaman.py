"""
train_megaman.py — Train on Mega Man 2 using RainbowDQN directly.

Bypasses MessageBus/Striatum routing overhead — clean direct loop.

Usage:
    python train_megaman.py            # headless (fast)
    python train_megaman.py --render   # show game window (slower)
    python train_megaman.py --episodes 200
"""

import argparse, time, sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from brain.environments.nes_adapter import MegaManEnv, N_ACTIONS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes",    type=int,   default=500)
    p.add_argument("--render",      action="store_true")
    p.add_argument("--n-features",  type=int,   default=84)
    p.add_argument("--checkpoint",  type=str,   default="checkpoints/megaman2.pt")
    p.add_argument("--lr",          type=float, default=6.25e-5)
    p.add_argument("--batch",       type=int,   default=32)
    p.add_argument("--buffer",      type=int,   default=50_000)
    p.add_argument("--frame-skip",  type=int,   default=4,
                   help="NES frames per agent action (default 4 = 4x speedup)")
    p.add_argument("--train-every", type=int,   default=4,
                   help="Train DQN every N steps (default 4)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)

    print("=" * 60)
    print("Throng5 × Mega Man 2  (RainbowDQN direct)")
    print(f"  Episodes : {args.episodes}")
    print(f"  Features : {args.n_features}")
    print(f"  Render   : {args.render}")
    print("=" * 60)

    # ── Environment ────────────────────────────────────────────────────
    print("Loading environment...", end="", flush=True)
    env = MegaManEnv(n_features=args.n_features, render=args.render)
    print(" OK")

    # ── Learner: RainbowDQN ────────────────────────────────────────────
    print("Loading RainbowDQN...", end="", flush=True)
    try:
        from brain.learning.torch_dqn import RainbowDQN
        dqn = RainbowDQN(
            n_features=args.n_features,
            n_actions=N_ACTIONS,
            hidden_sizes=(256, 256, 128),
            lr=args.lr,
            gamma=0.99,
            batch_size=args.batch,
        )
        print(f" OK  (device={dqn.device})")
        USE_DQN = True
    except Exception as e:
        print(f"\n  RainbowDQN unavailable ({e}) — using random policy")
        dqn = None
        USE_DQN = False

    # Simple replay buffer (list-based, no PER for now)
    replay = collections.deque(maxlen=args.buffer)

    # Screenshots for verification (saved after each reset)
    SAVE_SCREENSHOTS = True
    os.makedirs("checkpoints", exist_ok=True)

    # Shared memory frame buffer for live viewer (tools/watch_training.py)
    shm = None
    if args.render:
        try:
            from multiprocessing.shared_memory import SharedMemory
            SHM_NAME   = "throng5_mm2_frame"
            FRAME_BYTES = 256 * 240 * 3
            try:
                shm = SharedMemory(name=SHM_NAME, create=True, size=FRAME_BYTES)
            except FileExistsError:
                shm = SharedMemory(name=SHM_NAME, create=False)
            print(f"Shared memory ready. Open viewer: python tools/watch_training.py")
        except Exception as e:
            print(f"Shared memory unavailable: {e}")

    # Load checkpoint
    if USE_DQN and os.path.exists(args.checkpoint):
        try:
            dqn.load(args.checkpoint)
            print(f"Resumed from {args.checkpoint}")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")

    # ── Training loop ──────────────────────────────────────────────────
    best_reward = float("-inf")
    ep_rewards  = []
    ep_x_maxes  = []
    t_train_start = time.time()

    for episode in range(1, args.episodes + 1):
        obs   = env.reset()
        done  = False
        ep_r  = 0.0
        ep_x  = 0
        steps = 0
        t_ep  = time.time()

        # Show we're alive at start of episode
        print(f"Ep {episode:4d} | step 0...", end="\r", flush=True)

        # Save screenshot so you can verify game state (open checkpoints/epXXXX_start.bmp)
        if SAVE_SCREENSHOTS and episode <= 5:
            try:
                raw = env._env.render()
                if raw is not None:
                    import struct
                    h, w, _ = raw.shape
                    bmp = f"checkpoints/ep{episode:04d}_start.bmp"
                    row_sz, pad = w * 3, (4 - (w * 3) % 4) % 4
                    with open(bmp, "wb") as f:
                        f.write(b"BM")
                        f.write(struct.pack("<I", 54 + (row_sz + pad) * h))
                        f.write(b"\x00" * 4)
                        f.write(struct.pack("<II", 54, 40))
                        f.write(struct.pack("<ii", w, -h))
                        f.write(struct.pack("<HHI", 1, 24, 0))
                        f.write(struct.pack("<I", (row_sz + pad) * h))
                        f.write(struct.pack("<iiII", 2835, 2835, 0, 0))
                        for row in raw:
                            for px in row:
                                f.write(bytes([px[2], px[1], px[0]]))
                            f.write(b"\x00" * pad)
                    print(f"  Screenshot -> {bmp}")
            except Exception as e:
                pass

        while not done:
            # Action selection
            if USE_DQN and len(replay) >= args.batch:
                action, _ = dqn.select_action(obs, explore=True)
            else:
                action = env.action_space.sample()

            next_obs, reward, done, info = env.step(action)

            # No render loop needed (saves a screenshot after reset instead)

            # Write frame to shared memory for live viewer
            if shm is not None and steps % 4 == 0:
                try:
                    raw = env._last_raw   # (240,256,3) pixels stored by MegaManEnv.step()
                    if raw.shape == (240, 256, 3):
                        shm.buf[:raw.nbytes] = raw.tobytes()
                except Exception:
                    pass

            # Heartbeat every 100 steps
            if steps % 100 == 0:
                elapsed_so_far = time.time() - t_ep
                fps_so_far = steps / max(elapsed_so_far, 1e-6)
                print(
                    f"Ep {episode:4d} | step {steps:4d} | "
                    f"r={ep_r:6.2f} | x={ep_x:3d} | "
                    f"act={action} | lastr={reward:+.3f} | "
                    f"fps={fps_so_far:4.0f} | buf={len(replay):5d}",
                    end="\r", flush=True,
                )

            # Store transition
            replay.append((obs, action, reward, next_obs, done))

            # Train every step once buffer has enough
            if USE_DQN and len(replay) >= args.batch:
                # Sample mini-batch
                idx   = np.random.choice(len(replay), args.batch, replace=False)
                batch = [replay[i] for i in idx]
                s  = np.array([b[0] for b in batch], dtype=np.float32)
                a  = np.array([b[1] for b in batch], dtype=np.int64)
                r  = np.array([b[2] for b in batch], dtype=np.float32)
                ns = np.array([b[3] for b in batch], dtype=np.float32)
                d  = np.array([b[4] for b in batch], dtype=np.float32)

                import torch
                dev = dqn.device
                with torch.no_grad():
                    ns_t   = torch.FloatTensor(ns).to(dev)
                    next_q = dqn.target_net(ns_t)
                    best_a = dqn.online_net(ns_t).argmax(1)
                    next_v = next_q.gather(1, best_a.unsqueeze(1)).squeeze(1)
                    target = torch.FloatTensor(r).to(dev) + \
                             0.99 * next_v * (1 - torch.FloatTensor(d).to(dev))

                s_t  = torch.FloatTensor(s).to(dev)
                a_t  = torch.LongTensor(a).to(dev)
                q    = dqn.online_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                import torch.nn.functional as F
                loss = F.smooth_l1_loss(q, target.detach())

                dqn.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dqn.online_net.parameters(), 10.0)
                dqn.optimizer.step()
                dqn._total_updates += 1

                # Soft target update every 100 steps
                if dqn._total_updates % 100 == 0:
                    tau = 0.005
                    for tp, op in zip(dqn.target_net.parameters(),
                                      dqn.online_net.parameters()):
                        tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

            obs    = next_obs
            ep_r  += reward
            ep_x   = max(ep_x, info.get("x_pos", 0))
            steps += 1

        # Episode complete
        ep_rewards.append(ep_r)
        ep_x_maxes.append(ep_x)
        elapsed  = time.time() - t_ep
        fps      = steps / max(elapsed, 1e-6)
        mean100  = np.mean(ep_rewards[-100:])
        buf_size = len(replay)

        print(
            f"Ep {episode:4d}/{args.episodes} | "
            f"r={ep_r:7.1f} | "
            f"mean100={mean100:7.1f} | "
            f"x={ep_x:3d} | "
            f"steps={steps:4d} | "
            f"fps={fps:4.0f} | "
            f"buf={buf_size:6d} | "
            f"updates={dqn._total_updates if USE_DQN else 0}"
        )

        # Save best
        if USE_DQN and ep_r > best_reward:
            best_reward = ep_r
            try:
                dqn.save(args.checkpoint)
                print(f"  ✓ Best! ({best_reward:.1f}) saved → {args.checkpoint}")
            except Exception as e:
                print(f"  Save failed: {e}")

        # Periodic saves every 50 ep
        if USE_DQN and episode % 50 == 0:
            p = args.checkpoint.replace(".pt", f"_ep{episode}.pt")
            try:
                dqn.save(p)
            except Exception:
                pass

    total_time = time.time() - t_train_start
    env.close()
    print(f"\nDone in {total_time/60:.1f} min")
    print(f"Best reward  : {best_reward:.1f}")
    print(f"Best x_pos   : {max(ep_x_maxes)}")
    print(f"Final mean100: {np.mean(ep_rewards[-100:]):.1f}")


if __name__ == "__main__":
    main()
