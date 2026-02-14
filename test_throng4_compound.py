"""
Test: Compound Learning Experiment

Tests whether MetaStackPipeline demonstrates compound transfer:
1. Extended Tetris curriculum (levels 1→7) — does the gap widen?
2. Path richness: 1→2→3→4 vs 1→4 vs 3→4 (more steps = better?)
3. Cross-domain boost: GridWorld→Tetris1→2→3 vs Tetris1→2→3

Key insight from user:
- Episodes per level may NOT decrease (complexity grows exponentially)
- The GAP between fresh and progressive should WIDEN at higher levels
- More intermediate steps should produce better results
- Cross-domain pretraining should boost even different-dimension tasks
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
import time
from throng4.metastack_pipeline import MetaStackPipeline
from throng4.layers.meta1_synapse import DualHeadSynapseConfig
from throng4.layers.meta3_maml import DualHeadMAMLConfig
from throng3.environments.gym_envs import GridWorldAdapter
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv


# ============================================================
# Constants: fixed dimensions for consistent pipeline sizing
# ============================================================

# Max across all 7 levels (L7: state=210, actions=34)
MAX_STATE_DIM = 220   # Padded for safety
MAX_ACTION_DIM = 40   # Padded for safety

# ============================================================
# Shared helpers
# ============================================================

def make_pipeline(n_hidden=64, n_inputs=None, n_outputs=None):
    """Create a fresh MetaStackPipeline with fixed max dimensions."""
    return MetaStackPipeline(
        n_inputs=n_inputs or MAX_STATE_DIM,
        n_outputs=n_outputs or MAX_ACTION_DIM,
        n_hidden=n_hidden,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3),
        buffer_size=5000, batch_size=16, target_update_freq=20,
        gamma=0.99
    )


def pad_state(state, target_dim=MAX_STATE_DIM):
    """Zero-pad state to fixed dimension."""
    if len(state) >= target_dim:
        return state[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:len(state)] = state
    return padded


def select_tetris_action(pipeline, state, valid_actions):
    """Select action from valid Tetris actions using pipeline Q-values with masking."""
    padded = pad_state(state)
    q_values = pipeline.get_q_values(padded)
    
    n_valid = len(valid_actions)
    if np.random.rand() < pipeline.epsilon:
        return np.random.randint(n_valid), padded
    else:
        # Only consider Q-values for valid action indices
        action_idx = int(np.argmax(q_values[:n_valid]))
        return action_idx, padded


def train_tetris_level(pipeline, level, max_episodes=200, target_lines=2.0,
                       verbose=False):
    """
    Train pipeline on a single Tetris level with padded states.
    
    Returns:
        (episodes_used, final_mean_lines, rewards_history)
    """
    env = TetrisCurriculumEnv(level=level, max_pieces=100)
    
    rewards = []
    for episode in range(max_episodes):
        state = env.reset()
        total_lines = 0
        
        for step in range(100):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            
            action_idx, padded_state = select_tetris_action(pipeline, state, valid_actions)
            action = valid_actions[action_idx]
            
            next_state, reward, done, info = env.step(action)
            padded_next = pad_state(next_state)
            pipeline.update(padded_state, action_idx, reward, padded_next, done)
            
            total_lines = info.get('lines_cleared', 0)
            state = next_state
            
            if done:
                break
        
        rewards.append(total_lines)
        
        # Check target
        if len(rewards) >= 20 and np.mean(rewards[-20:]) >= target_lines:
            if verbose:
                print(f"    Reached {target_lines:.1f} at episode {episode + 1}")
            return episode + 1, np.mean(rewards[-20:]), rewards
        
        if verbose and (episode + 1) % 50 == 0:
            mean = np.mean(rewards[-50:])
            print(f"    Ep {episode + 1}: mean_50={mean:.2f} lines")
    
    final = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
    if verbose:
        print(f"    Finished {max_episodes} episodes, final={final:.2f} lines")
    return max_episodes, final, rewards


def transfer_maml(source, target):
    """Transfer MAML LR multipliers from source to target pipeline."""
    mults = source.maml.get_lr_multipliers()
    target.maml.meta_params['rl']['lr_multipliers'] = mults.copy()
    return mults


def train_gridworld(pipeline, n_episodes=200, verbose=False):
    """Train on GridWorld with padded states. Returns mean reward."""
    env = GridWorldAdapter(size=5)
    rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        padded = pad_state(state)
        total_reward = 0.0
        
        for _ in range(100):
            action = pipeline.select_action(padded, explore=True)
            # Clamp action to valid GridWorld range (4 actions)
            action = action % 4
            next_state, reward, done, _ = env.step(action)
            padded_next = pad_state(next_state)
            pipeline.update(padded, action, reward, padded_next, done)
            total_reward += reward
            padded = padded_next
            state = next_state
            if done:
                break
        
        rewards.append(total_reward)
        
        if verbose and (episode + 1) % 50 == 0:
            print(f"    Ep {episode + 1}: mean_50={np.mean(rewards[-50:]):.3f}")
    
    return np.mean(rewards[-20:])


# ============================================================
# Test 1: Extended Curriculum — Gap Analysis
# ============================================================

def test_extended_curriculum():
    """
    Test compound learning by measuring the GAP between fresh and progressive
    training at each level. If learning compounds, the gap should widen.
    """
    print("=" * 70)
    print("TEST 1: Extended Tetris Curriculum (Gap Analysis)")
    print("=" * 70)
    print("\nMeasuring gap between fresh vs progressive at each level")
    print("Compound learning = gap widens at higher levels\n")
    
    max_level = 7
    target_lines = 1.5  # Realistic target for higher levels
    max_ep_per_level = 150
    
    # --- Fresh training (each level from scratch) ---
    print("[FRESH] Training each level from scratch...")
    fresh_results = {}
    
    for level in range(1, max_level + 1):
        print(f"\n  Level {level}:")
        pipeline = make_pipeline()
        episodes, final_mean, history = train_tetris_level(
            pipeline, level, max_episodes=max_ep_per_level,
            target_lines=target_lines, verbose=True
        )
        fresh_results[level] = {
            'episodes': episodes,
            'final_mean': final_mean,
            'history': history,
        }
    
    # --- Progressive training (transfer between levels) ---
    print("\n\n[PROGRESSIVE] Training with transfer between levels...")
    progressive_results = {}
    prev_pipeline = None
    
    for level in range(1, max_level + 1):
        print(f"\n  Level {level}:")
        pipeline = make_pipeline()
        
        # Transfer full weight initialization from previous level
        if prev_pipeline is not None:
            pipeline.transfer_weights(prev_pipeline)
            print(f"    Transferred weights + LR mults from level {level - 1}")
        
        episodes, final_mean, history = train_tetris_level(
            pipeline, level, max_episodes=max_ep_per_level,
            target_lines=target_lines, verbose=True
        )
        progressive_results[level] = {
            'episodes': episodes,
            'final_mean': final_mean,
            'history': history,
            'lr_multipliers': pipeline.maml.get_lr_multipliers(),
        }
        prev_pipeline = pipeline
    
    # --- Analysis ---
    print("\n\n" + "=" * 70)
    print("RESULTS: Gap Analysis")
    print("=" * 70)
    
    print(f"\n{'Level':<8} {'Fresh Ep':<12} {'Prog Ep':<12} {'Gap':<10} {'Fresh Mean':<12} {'Prog Mean':<12} {'Mean Gap':<10}")
    print("-" * 76)
    
    gaps = []
    mean_gaps = []
    for level in range(1, max_level + 1):
        f = fresh_results[level]
        p = progressive_results[level]
        ep_gap = f['episodes'] - p['episodes']
        mean_gap = p['final_mean'] - f['final_mean']
        gaps.append(ep_gap)
        mean_gaps.append(mean_gap)
        
        print(f"  {level:<6} {f['episodes']:<12} {p['episodes']:<12} {ep_gap:<10} "
              f"{f['final_mean']:<12.2f} {p['final_mean']:<12.2f} {mean_gap:<+10.2f}")
    
    # Check for widening gap
    print("\nGap trend (episode savings):", gaps)
    print("Gap trend (mean improvement):", [f"{g:+.2f}" for g in mean_gaps])
    
    # Simple trend: is the gap generally increasing?
    later_gaps = mean_gaps[len(mean_gaps)//2:]
    early_gaps = mean_gaps[:len(mean_gaps)//2]
    avg_early = np.mean(early_gaps) if early_gaps else 0
    avg_later = np.mean(later_gaps) if later_gaps else 0
    
    print(f"\nAvg gap (early levels): {avg_early:+.3f}")
    print(f"Avg gap (later levels): {avg_later:+.3f}")
    
    if avg_later > avg_early:
        print("[PASS] Gap widens at higher levels - compound learning detected!")
    else:
        print("[INFO] Gap does not widen - transfer is linear, not compound")
    
    print()
    return fresh_results, progressive_results


# ============================================================
# Test 2: Path Richness
# ============================================================

def test_path_richness():
    """
    Test whether more intermediate steps produce better transfer.
    
    Compare:
    - Direct:  1→4  (skip levels 2,3)
    - Partial: 3→4  (one intermediate)
    - Full:    1→2→3→4  (all intermediates)
    """
    print("=" * 70)
    print("TEST 2: Path Richness (1->4 vs 3->4 vs 1->2->3->4)")
    print("=" * 70)
    print("\nMore intermediate steps should produce better transfer\n")
    
    target_level = 4
    target_lines = 1.5
    max_ep = 150
    
    # --- Path A: Fresh (no transfer) ---
    print("[Fresh] Level 4 from scratch...")
    fresh = make_pipeline()
    fresh_ep, fresh_mean, _ = train_tetris_level(
        fresh, target_level, max_episodes=max_ep,
        target_lines=target_lines, verbose=True
    )
    
    # --- Path B: Direct 1→4 ---
    print("\n[1->4] Train level 1, transfer to level 4...")
    p1 = make_pipeline()
    train_tetris_level(p1, 1, max_episodes=100, target_lines=target_lines, verbose=True)
    
    p1_to_4 = make_pipeline()
    p1_to_4.transfer_weights(p1)
    direct_ep, direct_mean, _ = train_tetris_level(
        p1_to_4, target_level, max_episodes=max_ep,
        target_lines=target_lines, verbose=True
    )
    
    # --- Path C: 3→4 ---
    print("\n[3->4] Train level 3, transfer to level 4...")
    p3 = make_pipeline()
    train_tetris_level(p3, 3, max_episodes=100, target_lines=target_lines, verbose=True)
    
    p3_to_4 = make_pipeline()
    p3_to_4.transfer_weights(p3)
    partial_ep, partial_mean, _ = train_tetris_level(
        p3_to_4, target_level, max_episodes=max_ep,
        target_lines=target_lines, verbose=True
    )
    
    # --- Path D: 1→2→3→4 (full chain) ---
    print("\n[1->2->3->4] Full progressive chain...")
    prev = None
    for level in [1, 2, 3]:
        print(f"  Training level {level}...")
        p = make_pipeline()
        if prev is not None:
            p.transfer_weights(prev)
        train_tetris_level(p, level, max_episodes=100,
                          target_lines=target_lines, verbose=True)
        prev = p
    
    print("  Training level 4 (with full chain transfer)...")
    full_chain = make_pipeline()
    full_chain.transfer_weights(prev)
    chain_ep, chain_mean, _ = train_tetris_level(
        full_chain, target_level, max_episodes=max_ep,
        target_lines=target_lines, verbose=True
    )
    
    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS: Path Richness")
    print("=" * 70)
    
    paths = [
        ("Fresh (no transfer)", fresh_ep, fresh_mean),
        ("1->4 (direct)", direct_ep, direct_mean),
        ("3->4 (one step)", partial_ep, partial_mean),
        ("1->2->3->4 (full)", chain_ep, chain_mean),
    ]
    
    print(f"\n{'Path':<25} {'Episodes':<12} {'Final Mean':<12}")
    print("-" * 49)
    for name, ep, mean in paths:
        print(f"  {name:<23} {ep:<12} {mean:<12.2f}")
    
    # Check ordering: full chain should be best
    if chain_mean >= direct_mean and chain_mean >= partial_mean:
        print("\n[PASS] Full chain (1->2->3->4) achieves best result")
    else:
        print("\n[INFO] Full chain is not best - path richness not confirmed")
    
    if chain_ep <= fresh_ep:
        speedup = fresh_ep / max(chain_ep, 1)
        print(f"[PASS] Full chain {speedup:.1f}x faster than fresh")
    
    print()
    return paths


# ============================================================
# Test 3: Cross-Domain Boost
# ============================================================

def test_cross_domain_boost():
    """
    Test whether GridWorld pretraining boosts Tetris curriculum.
    
    Compare:
    - Tetris only:   1→2→3  (Tetris curriculum only)
    - Cross-domain:  GridWorld→1→2→3  (GridWorld first, then Tetris)
    """
    print("=" * 70)
    print("TEST 3: Cross-Domain Boost (GridWorld -> Tetris)")
    print("=" * 70)
    print("\nDoes GridWorld pretraining help Tetris learning?\n")
    
    target_lines = 1.5
    max_ep = 150
    
    # --- Tetris only: 1→2→3 ---
    print("[Tetris Only] 1->2->3...")
    tetris_only_total = 0
    prev = None
    tetris_only_results = {}
    
    for level in [1, 2, 3]:
        print(f"\n  Level {level}:")
        p = make_pipeline()
        if prev is not None:
            p.transfer_weights(prev)
        ep, mean, _ = train_tetris_level(
            p, level, max_episodes=max_ep,
            target_lines=target_lines, verbose=True
        )
        tetris_only_total += ep
        tetris_only_results[level] = {'episodes': ep, 'final_mean': mean}
        prev = p
    
    print(f"\n  Tetris-only total: {tetris_only_total} episodes")
    
    # --- Cross-domain: GridWorld→1→2→3 ---
    print("\n\n[Cross-Domain] GridWorld->1->2->3...")
    
    # Train on GridWorld first
    print("\n  GridWorld pretraining (200 episodes):")
    gw_pipeline = make_pipeline()
    gw_mean = train_gridworld(gw_pipeline, n_episodes=200, verbose=True)
    print(f"    GridWorld final: {gw_mean:.3f}")
    gw_maml_mults = gw_pipeline.maml.get_lr_multipliers()
    print(f"    Learned LR mults: {gw_maml_mults}")
    
    cross_domain_total = 0
    prev = gw_pipeline
    cross_domain_results = {}
    
    for level in [1, 2, 3]:
        print(f"\n  Level {level}:")
        p = make_pipeline()
        p.transfer_weights(prev)
        ep, mean, _ = train_tetris_level(
            p, level, max_episodes=max_ep,
            target_lines=target_lines, verbose=True
        )
        cross_domain_total += ep
        cross_domain_results[level] = {'episodes': ep, 'final_mean': mean}
        prev = p
    
    print(f"\n  Cross-domain total: {cross_domain_total} episodes")
    
    # --- Results ---
    print("\n" + "=" * 70)
    print("RESULTS: Cross-Domain Boost")
    print("=" * 70)
    
    print(f"\n{'Level':<8} {'Tetris Only':<15} {'Cross-Domain':<15} {'Boost':<10}")
    print("-" * 48)
    
    for level in [1, 2, 3]:
        t = tetris_only_results[level]
        c = cross_domain_results[level]
        boost = t['final_mean'] - c['final_mean']
        ep_diff = t['episodes'] - c['episodes']
        print(f"  {level:<6} {t['episodes']} ep ({t['final_mean']:.2f})  "
              f"{c['episodes']} ep ({c['final_mean']:.2f})  "
              f"{ep_diff:+d} ep")
    
    print(f"\nTotal: Tetris-only={tetris_only_total}, Cross-domain={cross_domain_total}")
    
    if cross_domain_total < tetris_only_total:
        savings = tetris_only_total - cross_domain_total
        pct = savings / tetris_only_total * 100
        print(f"Savings: {savings} episodes ({pct:.1f}%)")
        print("[PASS] Cross-domain pretraining boosts Tetris learning!")
    else:
        print("[INFO] Cross-domain pretraining did not help Tetris")
    
    print()
    return tetris_only_results, cross_domain_results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPOUND LEARNING EXPERIMENT")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Test 1: Extended curriculum with gap analysis
        fresh, progressive = test_extended_curriculum()
        
        # Test 2: Path richness
        paths = test_path_richness()
        
        # Test 3: Cross-domain boost
        tetris_only, cross_domain = test_cross_domain_boost()
        
        elapsed = time.time() - start_time
        
        print("=" * 70)
        print("COMPOUND LEARNING EXPERIMENT COMPLETE")
        print(f"Total time: {elapsed / 60:.1f} minutes")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
