"""
Phase F6: Hyperparameter Sweep for Compound Learning

Tests different network configurations to find one that shows compound learning
at higher Tetris levels (L4-L7).

Focus: Network size (64 → 128 → 256 hidden units)
Hypothesis: Larger networks will show widening gap at higher levels.
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
import time
from throng4.metastack_pipeline import MetaStackPipeline
from throng4.layers.meta1_synapse import DualHeadSynapseConfig
from throng4.layers.meta3_maml import DualHeadMAMLConfig
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv


# Fixed dimensions for all Tetris levels
MAX_STATE_DIM = 220
MAX_ACTION_DIM = 40


def pad_state(state):
    """Pad state to MAX_STATE_DIM."""
    padded = np.zeros(MAX_STATE_DIM)
    padded[:len(state)] = state
    return padded


def select_tetris_action(pipeline, state, valid_actions):
    """Select action with masking for valid Tetris actions."""
    padded = pad_state(state)
    q_values = pipeline.get_q_values(padded)
    
    n_valid = len(valid_actions)
    if n_valid == 0:
        return 0, padded
    else:
        action_idx = int(np.argmax(q_values[:n_valid]))
        return action_idx, padded


def train_tetris_level(pipeline, level, max_episodes=150, target_lines=1.5, verbose=False):
    """Train on a Tetris level. Returns (episodes_used, final_mean, history)."""
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
            
            total_lines += info.get('lines_cleared', 0)
            state = next_state
            
            if done:
                break
        
        rewards.append(total_lines)
        
        if verbose and (episode + 1) % 50 == 0:
            mean_50 = np.mean(rewards[-50:])
            print(f"    Ep {episode + 1}: mean_50={mean_50:.2f} lines")
        
        # Early stopping if target reached
        if len(rewards) >= 20:
            recent_mean = np.mean(rewards[-20:])
            if recent_mean >= target_lines:
                if verbose:
                    print(f"    Reached {target_lines} at episode {episode + 1}")
                return episode + 1, recent_mean, rewards
    
    final = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
    if verbose:
        print(f"    Finished {max_episodes} episodes, final={final:.2f} lines")
    return max_episodes, final, rewards


def make_pipeline(n_hidden):
    """Create pipeline with specified hidden layer size."""
    return MetaStackPipeline(
        n_inputs=MAX_STATE_DIM,
        n_outputs=MAX_ACTION_DIM,
        n_hidden=n_hidden,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3),
        buffer_size=5000, batch_size=16, target_update_freq=20,
        gamma=0.99
    )


def test_network_size(n_hidden, max_level=7):
    """
    Test extended curriculum with specified network size.
    Returns gap analysis results.
    """
    print(f"\n{'='*70}")
    print(f"TESTING: {n_hidden} Hidden Units")
    print(f"{'='*70}\n")
    
    target_lines = 1.5
    max_ep = 150
    
    # Fresh training
    print("[FRESH] Training each level from scratch...")
    fresh_results = {}
    for level in range(1, max_level + 1):
        print(f"\n  Level {level}:")
        pipeline = make_pipeline(n_hidden)
        episodes, final_mean, _ = train_tetris_level(
            pipeline, level, max_episodes=max_ep,
            target_lines=target_lines, verbose=True
        )
        fresh_results[level] = {'episodes': episodes, 'final_mean': final_mean}
    
    # Progressive training with weight transfer
    print("\n\n[PROGRESSIVE] Training with weight transfer...")
    progressive_results = {}
    prev_pipeline = None
    
    for level in range(1, max_level + 1):
        print(f"\n  Level {level}:")
        pipeline = make_pipeline(n_hidden)
        
        if prev_pipeline is not None:
            pipeline.transfer_weights(prev_pipeline)
            print(f"    Transferred weights from level {level - 1}")
        
        episodes, final_mean, _ = train_tetris_level(
            pipeline, level, max_episodes=max_ep,
            target_lines=target_lines, verbose=True
        )
        progressive_results[level] = {'episodes': episodes, 'final_mean': final_mean}
        prev_pipeline = pipeline
    
    # Analysis
    print(f"\n\n{'='*70}")
    print(f"RESULTS: {n_hidden} Hidden Units")
    print(f"{'='*70}\n")
    
    print(f"{'Level':<8} {'Fresh Ep':<12} {'Prog Ep':<12} {'Gap':<10} {'Fresh Mean':<12} {'Prog Mean':<12}")
    print("-" * 70)
    
    gaps = []
    for level in range(1, max_level + 1):
        f = fresh_results[level]
        p = progressive_results[level]
        ep_gap = f['episodes'] - p['episodes']
        gaps.append(ep_gap)
        
        print(f"  {level:<6} {f['episodes']:<12} {p['episodes']:<12} {ep_gap:<10} "
              f"{f['final_mean']:<12.2f} {p['final_mean']:<12.2f}")
    
    # Check for widening gap
    early_gap = np.mean(gaps[:3])  # L1-L3
    late_gap = np.mean(gaps[4:])   # L5-L7
    
    print(f"\nGap trend:")
    print(f"  Early levels (L1-L3): {early_gap:+.1f} episodes")
    print(f"  Later levels (L5-L7): {late_gap:+.1f} episodes")
    print(f"  Widening: {late_gap - early_gap:+.1f} episodes")
    
    compound_detected = late_gap > early_gap + 10  # At least 10 episode improvement
    
    if compound_detected:
        print(f"\n✓ COMPOUND LEARNING DETECTED with {n_hidden} hidden units!")
    else:
        print(f"\n✗ No clear compounding with {n_hidden} hidden units")
    
    return {
        'n_hidden': n_hidden,
        'gaps': gaps,
        'early_gap': early_gap,
        'late_gap': late_gap,
        'compound_detected': compound_detected,
        'fresh_results': fresh_results,
        'progressive_results': progressive_results,
    }


if __name__ == '__main__':
    print("="*70)
    print("PHASE F6: HYPERPARAMETER SWEEP")
    print("="*70)
    print("\nTesting network sizes: 64, 128, 256 hidden units")
    print("Goal: Find configuration that shows compound learning at L4-L7\n")
    
    start_time = time.time()
    
    # Test different network sizes
    results = []
    for n_hidden in [64, 128, 256]:
        result = test_network_size(n_hidden, max_level=7)
        results.append(result)
    
    # Summary
    elapsed = (time.time() - start_time) / 60
    
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n{'Hidden Units':<15} {'Early Gap':<12} {'Late Gap':<12} {'Widening':<12} {'Compound?':<12}")
    print("-" * 70)
    
    for r in results:
        widening = r['late_gap'] - r['early_gap']
        status = "✓ YES" if r['compound_detected'] else "✗ NO"
        print(f"  {r['n_hidden']:<13} {r['early_gap']:+<11.1f} {r['late_gap']:+<11.1f} "
              f"{widening:+<11.1f} {status:<12}")
    
    print(f"\nTotal runtime: {elapsed:.1f} minutes")
    
    # Recommendation
    best = max(results, key=lambda r: r['late_gap'] - r['early_gap'])
    print(f"\n✓ RECOMMENDATION: Use {best['n_hidden']} hidden units for compound learning")
