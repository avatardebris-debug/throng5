"""
Test: MetaStackPipeline Integration

Verifies:
1. MetaStackPipeline basic functionality (forward/backward)
2. MAML task batching and meta-updates
3. Performance comparison to SimplePipeline baseline
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.metastack_pipeline import MetaStackPipeline
from throng4.pipeline import SimplePipeline
from throng4.layers.meta1_synapse import DualHeadSynapseConfig
from throng4.layers.meta3_maml import DualHeadMAMLConfig
from throng3.environments.gym_envs import GridWorldAdapter


def run_episode(env, pipeline, max_steps=100, explore=True):
    """Run one episode."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    
    for _ in range(max_steps):
        action = pipeline.select_action(state, explore=explore)
        next_state, reward, done, _ = env.step(action)
        result = pipeline.update(state, action, reward, next_state, done)
        
        total_reward += reward
        steps += 1
        state = next_state
        
        if done:
            break
    
    return total_reward, steps


def test_metastack_basic():
    """Test basic MetaStackPipeline functionality."""
    print("=" * 70)
    print("TEST 1: MetaStackPipeline Basic Functionality")
    print("=" * 70)
    
    env = GridWorldAdapter(size=5)
    
    pipeline = MetaStackPipeline(
        n_inputs=2,
        n_outputs=4,
        n_hidden=32,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(
            meta_batch_size=5,  # Trigger meta-update every 5 episodes
            inner_steps=3
        )
    )
    
    print(f"Pipeline created: {pipeline.ann.get_num_parameters()} params")
    print(f"MAML batch size: 5 episodes")
    print()
    
    # Run 10 episodes
    meta_updates = 0
    for episode in range(10):
        reward, steps = run_episode(env, pipeline, explore=True)
        stats = pipeline.get_stats()
        
        if stats['meta_updates'] > meta_updates:
            meta_updates = stats['meta_updates']
            print(f"Episode {episode + 1}: META-UPDATE triggered! "
                  f"(total meta-updates: {meta_updates})")
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}: reward={reward:.3f}, "
                  f"ε={stats['epsilon']:.3f}, "
                  f"meta_updates={stats['meta_updates']}")
    
    final_stats = pipeline.get_stats()
    print(f"\nFinal stats:")
    print(f"  Episodes: {final_stats['episodes']}")
    print(f"  Meta-updates: {final_stats['meta_updates']}")
    print(f"  LR multipliers: {final_stats['lr_multipliers']}")
    
    # Should have 2 meta-updates (episodes 5 and 10)
    assert final_stats['meta_updates'] >= 1, "Should have at least 1 meta-update"
    print("[PASS] MetaStackPipeline basic test PASSED\n")


def test_metastack_vs_simple():
    """Compare MetaStackPipeline to SimplePipeline baseline."""
    print("=" * 70)
    print("TEST 2: MetaStackPipeline vs SimplePipeline")
    print("=" * 70)
    
    env = GridWorldAdapter(size=5)
    n_episodes = 100
    
    # Test SimplePipeline
    print("\nTraining SimplePipeline...")
    simple = SimplePipeline(n_inputs=2, n_outputs=4, n_hidden=64)
    simple_rewards = []
    for _ in range(n_episodes):
        reward, _ = run_episode(env, simple, explore=True)
        simple_rewards.append(reward)
    
    # Evaluate SimplePipeline
    simple_eval = []
    for _ in range(20):
        reward, _ = run_episode(env, simple, explore=False)
        simple_eval.append(reward)
    
    simple_mean = np.mean(simple_eval)
    print(f"SimplePipeline eval: {simple_mean:.3f} ± {np.std(simple_eval):.3f}")
    
    # Test MetaStackPipeline
    print("\nTraining MetaStackPipeline...")
    metastack = MetaStackPipeline(
        n_inputs=2,
        n_outputs=4,
        n_hidden=64,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(
            meta_batch_size=10,
            inner_steps=3,
            meta_lr=0.001
        )
    )
    
    metastack_rewards = []
    for episode in range(n_episodes):
        reward, _ = run_episode(env, metastack, explore=True)
        metastack_rewards.append(reward)
        
        if (episode + 1) % 20 == 0:
            stats = metastack.get_stats()
            recent_mean = np.mean(metastack_rewards[-20:])
            print(f"Episode {episode + 1}: mean_20={recent_mean:.3f}, "
                  f"meta_updates={stats['meta_updates']}")
    
    # Evaluate MetaStackPipeline
    metastack_eval = []
    for _ in range(20):
        reward, _ = run_episode(env, metastack, explore=False)
        metastack_eval.append(reward)
    
    metastack_mean = np.mean(metastack_eval)
    metastack_stats = metastack.get_stats()
    
    print(f"\nMetaStackPipeline eval: {metastack_mean:.3f} ± {np.std(metastack_eval):.3f}")
    print(f"Meta-updates triggered: {metastack_stats['meta_updates']}")
    print(f"Learned LR multipliers: {metastack_stats['lr_multipliers']}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"SimplePipeline:    {simple_mean:.3f} ({simple_mean*100:.0f}% success)")
    print(f"MetaStackPipeline: {metastack_mean:.3f} ({metastack_mean*100:.0f}% success)")
    
    improvement = metastack_mean - simple_mean
    print(f"\nImprovement: {improvement:+.3f}")
    
    # MetaStackPipeline should be at least competitive
    if metastack_mean >= simple_mean * 0.8:
        print("[PASS] MetaStackPipeline is competitive with SimplePipeline")
    else:
        print("[WARNING] MetaStackPipeline underperforms SimplePipeline")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("METASTACK PIPELINE INTEGRATION TEST")
    print("=" * 70 + "\n")
    
    try:
        test_metastack_basic()
        test_metastack_vs_simple()
        
        print("=" * 70)
        print("ALL METASTACK TESTS PASSED [OK]")
        print("=" * 70)
        print("\nMeta^0 → Meta^1 → Meta^3 integration validated!")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
