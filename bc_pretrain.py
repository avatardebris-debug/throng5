"""
bc_pretrain.py
==============
Behavior Cloning pretrainer: takes your BEST human episode from
replay_db.sqlite and drills the exact state->action sequence into the
agent backbone with supervised learning BEFORE any RL starts.

This is the right approach for hard-exploration games like Montezuma:
  - No conflicting signals from other sessions
  - Sequence order matters — states are trained in context
  - Multiple epochs = the network sees the expert path many times
  - Then RL takes over from a backbone that already knows HOW to reach the key

After pretraining, weights are saved for benchmark_human.py or
eval_atari_agent.py to load as --weights.

Usage
-----
    # Pretrain from best Montezuma episode, 50 epochs
    python bc_pretrain.py --game ALE/MontezumaRevenge-v5 --epochs 50

    # Specify a particular episode
    python bc_pretrain.py --game ALE/MontezumaRevenge-v5 \\
        --episode-id ep_92b7324c-9ae6-4f86-85xx --epochs 100

    # Then evaluate
    python eval_atari_agent.py --game ALE/MontezumaRevenge-v5 \\
        --weights benchmark_results/bc_ALE_MontezumaRevenge_v5.npz --episodes 10
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from throng4.learning.portable_agent import PortableNNAgent, AgentConfig

DB_PATH     = _ROOT / "experiments" / "replay_db.sqlite"
RESULTS_DIR = _ROOT / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Load best episode from DB
# ─────────────────────────────────────────────────────────────────────

def get_best_episode(
    db_path: Path,
    game_id: str,
    episode_id: str | None = None,
) -> tuple[str, float, list[dict]]:
    """
    Returns (episode_id, score, steps_list).
    steps_list is ordered by step_idx with fields:
        abstract_vec_json, human_action, executed_action, reward, done
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row

    if episode_id is None:
        row = con.execute("""
            SELECT e.episode_id, e.final_score
            FROM episodes e
            JOIN sessions s ON s.session_id = e.session_id
            WHERE s.env_name = ?
              AND e.final_score IS NOT NULL
            ORDER BY e.final_score DESC LIMIT 1
        """, (game_id,)).fetchone()
        if row is None:
            con.close()
            raise ValueError(f"No episodes found for {game_id} in {db_path}")
        episode_id = row["episode_id"]
        score = row["final_score"]
    else:
        row = con.execute(
            "SELECT final_score FROM episodes WHERE episode_id = ?",
            (episode_id,)
        ).fetchone()
        score = row["final_score"] if row else 0.0

    # Load transitions in step order
    rows = con.execute("""
        SELECT t.step_idx, t.human_action, t.executed_action,
               t.reward, t.done,
               m.abstract_vec_json
        FROM transitions t
        JOIN transition_metrics m ON m.transition_id = t.id
        WHERE t.episode_id = ?
          AND m.abstract_vec_json IS NOT NULL
        ORDER BY t.step_idx ASC
    """, (episode_id,)).fetchall()
    con.close()

    steps = [{
        "step":        r["step_idx"],
        "human_action": r["human_action"] if r["human_action"] is not None
                        else r["executed_action"],
        "reward":      r["reward"],
        "done":        bool(r["done"]),
        "vec":         json.loads(r["abstract_vec_json"]),
    } for r in rows]

    return episode_id, score, steps


# ─────────────────────────────────────────────────────────────────────
# Behavior Cloning training
# ─────────────────────────────────────────────────────────────────────

def _bc_loss_step(
    agent: PortableNNAgent,
    feat: np.ndarray,
    target_action: int,
    n_actions: int,
    lr: float,
) -> float:
    """
    One supervised gradient step on the EXPERT ACTION ONLY.

    Target Q for the human action = +2.0 (high, unambiguous reward signal).
    Other actions are NOT touched — no conflicting gradients.
    Also trains imitation head cross-entropy for the same step.

    Returns squared error.
    """
    ram = np.array(feat[:128], dtype=np.float32)

    # Feature vector for the expert action
    expert_feat = np.zeros(128 + n_actions, dtype=np.float32)
    expert_feat[:128] = ram
    expert_feat[128 + target_action] = 1.0

    target_q = 2.0   # high positive target for the expert action

    agent._training = True
    try:
        pred = agent.forward(expert_feat)
        error = pred - target_q
        ce = np.clip(error, -2.0, 2.0)   # tighter clip for BC stability
        loss = float(error ** 2)

        # Backprop
        agent.W3 -= lr * ce * agent._last_h2
        agent.b3 -= lr * ce

        dh2 = ce * agent.W3[0] * (agent._last_h2 > 0)
        # Clip layer gradients to prevent blowup
        dh2 = np.clip(dh2, -1.0, 1.0)
        agent.W2 -= lr * np.outer(dh2, agent._last_h1)
        agent.b2 -= lr * dh2

        dh1 = (agent.W2.T @ dh2) * (agent._last_h1 > 0)
        dh1 = np.clip(dh1, -1.0, 1.0)
        agent.W1 -= lr * np.outer(dh1, agent._last_x)
        agent.b1 -= lr * dh1
    finally:
        agent._training = False

    # Imitation head cross-entropy (separate head, doesn't affect backbone Q)
    if agent.config.use_imitation_head:
        agent._train_imitation_step(expert_feat, target_action, loss_scale=1.0)

    return loss



def bc_pretrain(
    agent: PortableNNAgent,
    steps: list[dict],
    n_actions: int,
    n_epochs: int = 50,
    lr: float = 0.002,
    verbose: bool = True,
) -> list[float]:
    """
    Train agent on expert trajectory for n_epochs passes.
    Returns per-epoch mean loss.
    """
    # Only keep steps with a real action (not NOOP-only early frames)
    demo = [(s["vec"], s["human_action"]) for s in steps
            if s["human_action"] is not None and 0 <= s["human_action"] < n_actions]

    # Filter: focus more on reward steps and steps around rewards
    # (steps within 20 of a non-zero reward are high-value)
    reward_steps = {i for i, s in enumerate(steps) if s["reward"] != 0}
    high_value = set()
    for ri in reward_steps:
        high_value.update(range(max(0, ri - 20), min(len(steps), ri + 5)))

    # Over-sample high-value steps: include them 3x
    weighted_demo = demo.copy()
    for i in high_value:
        if i < len(demo):
            weighted_demo.extend([demo[i]] * 2)

    if verbose:
        print(f"  Demo steps: {len(demo)}  "
              f"  High-value steps: {len(high_value)} (reward-adjacent)")
        print(f"  Weighted demo: {len(weighted_demo)} steps  x {n_epochs} epochs")

    epoch_losses = []
    rng = np.random.RandomState(0)
    t0 = time.time()

    for epoch in range(n_epochs):
        # Shuffle demo each epoch
        idx = rng.permutation(len(weighted_demo))
        epoch_loss = 0.0

        for i in idx:
            vec, action = weighted_demo[i]
            feat = np.array(vec, dtype=np.float32)
            loss = _bc_loss_step(agent, feat, action, n_actions, lr)
            epoch_loss += loss

        mean_loss = epoch_loss / len(idx)
        epoch_losses.append(mean_loss)

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (epoch + 1) * (n_epochs - epoch - 1)
            print(f"  epoch {epoch+1:>4}/{n_epochs}  "
                  f"loss={mean_loss:.4f}  "
                  f"eta={eta/60:.1f}min")

    # Sync target network after pretraining
    agent.sync_target_network()
    return epoch_losses


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="Behavior cloning pretrainer from best human episode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--game", default="ALE/MontezumaRevenge-v5")
    p.add_argument("--episode-id", default=None,
                   help="Specific episode ID (default: highest-scoring)")
    p.add_argument("--epochs", type=int, default=50,
                   help="Training passes over the demo trajectory")
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--db", default=str(DB_PATH))
    p.add_argument("--out", default=None,
                   help="Output weights path (default: benchmark_results/bc_<game>.npz)")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    print(f"\nBehavior Cloning Pretrainer")
    print(f"  Game:   {args.game}")
    print(f"  DB:     {args.db}")
    print(f"  Epochs: {args.epochs}")

    # Load best episode
    ep_id, score, steps = get_best_episode(
        Path(args.db), args.game, args.episode_id
    )
    print(f"\n  Best episode: {ep_id[:32]}...")
    print(f"  Score: {score:.0f}  Steps: {len(steps)}")

    # Build environment for action space size
    import gymnasium as gym, ale_py
    gym.register_envs(ale_py)
    env = gym.make(args.game, obs_type="ram", render_mode=None)
    n_actions = env.action_space.n
    action_meanings = list(env.unwrapped.get_action_meanings())
    env.close()
    print(f"  n_actions: {n_actions}")

    # Action distribution in the demo
    from collections import Counter
    action_counts = Counter(s["human_action"] for s in steps if s["human_action"] is not None)
    print(f"  Top demo actions: "
          + ", ".join(f"{action_meanings[a]}:{c}"
                      for a, c in action_counts.most_common(5)))

    # Build agent
    n_features = 128 + n_actions
    cfg = AgentConfig(
        n_hidden=256, n_hidden2=128,
        epsilon=0.15,       # slight exploration after BC
        epsilon_min=0.02,
        use_imitation_head=True,
        imitation_n_actions=n_actions,
        imitation_lr=args.lr,
    )
    agent = PortableNNAgent(n_features, config=cfg, seed=42)

    # Run BC
    print(f"\nStarting behavior cloning...")
    losses = bc_pretrain(
        agent, steps, n_actions,
        n_epochs=args.epochs,
        lr=args.lr,
        verbose=not args.quiet,
    )

    # Save weights
    slug = args.game.replace("/", "_").replace("-", "_")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"bc_{slug}.npz"
    agent.save_weights(str(out_path))
    print(f"\nWeights saved -> {out_path}")
    print(f"Final loss: {losses[-1]:.4f}  (initial: {losses[0]:.4f})")
    print(f"\nNext step:")
    print(f"  python eval_atari_agent.py \\")
    print(f"    --game {args.game} \\")
    print(f"    --weights {out_path} \\")
    print(f"    --episodes 10")
