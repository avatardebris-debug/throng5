"""
Test 3-Region Integration: Striatum + Cortex + Hippocampus

Validates that all three regions work together coordinated by Executive.
"""

import sys
sys.path.insert(0, 'throng35')
sys.path.insert(0, 'throng3')

import numpy as np
from throng35.regions.striatum import StriatumRegion
from throng35.regions.cortex import CortexRegion
from throng35.regions.hippocampus import HippocampusRegion
from throng35.coordination.executive import ExecutiveController
from throng35.learning.qlearning import QLearningConfig
from throng35.learning.hebbian import HebbianConfig
from throng35.learning.stdp import STDPConfig
from throng3.environments import GridWorldAdapter

print("="*70)
print("3-Region Integration Test")
print("="*70)
print("Striatum (Q-learning) + Cortex (Hebbian) + Hippocampus (STDP)")
print("Target: Maintain high success with 3-region coordination")
print()

# Create environment
env = GridWorldAdapter()

# Create all three regions
striatum = StriatumRegion(
    n_states=2,
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

cortex = CortexRegion(
    n_neurons=100,
    n_features=10,
    config=HebbianConfig(learning_rate=0.01)
)

hippocampus = HippocampusRegion(
    n_neurons=50,
    sequence_length=10,
    config=STDPConfig()
)

# Create executive with all three regions
executive = ExecutiveController({
    'striatum': striatum,
    'cortex': cortex,
    'hippocampus': hippocampus
})

print("3-region architecture created:")
print(f"  Striatum: Q-learning (2 states, 4 actions)")
print(f"  Cortex: Hebbian (100 neurons, 10 features)")
print(f"  Hippocampus: STDP (50 neurons, sequence learning)")
print(f"  Executive: Coordinating all three regions")
print()

# ============================================================
# PHASE 1: Curriculum Learning
# ============================================================
print("="*70)
print("PHASE 1: Curriculum Learning (5 demonstrations)")
print("="*70)

optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]  # right×4, down×4

for demo_ep in range(5):
    obs = env.reset()
    prev_reward = 0
    activations = np.random.randn(100) * 0.1
    total_reward = 0
    
    for step, action in enumerate(optimal_actions):
        # All regions step together
        result = executive.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        total_reward += reward
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    print(f"  Demo {demo_ep+1}: reward={total_reward:.3f}")

print()
stats = executive.get_stats()
print(f"After demonstrations:")
print(f"  Striatum updates: {stats['striatum']['n_updates']}")
print(f"  Cortex pattern strength: {stats['cortex']['avg_pattern_strength']:.3f}")
print(f"  Hippocampus episodes: {stats['hippocampus']['episodic_memories']}")
print()

# ============================================================
# PHASE 2: Exploration & Learning
# ============================================================
print("="*70)
print("PHASE 2: Exploration & Learning (95 episodes)")
print("="*70)

successes = 0
returns = []

for episode in range(95):
    obs = env.reset()
    done = False
    steps = 0
    prev_reward = 0
    episode_reward = 0
    activations = np.random.randn(100) * 0.1
    
    while not done and steps < 100:
        result = executive.step(
            raw_observation=obs,
            reward=prev_reward,
            done=False,
            activations=activations
        )
        
        action = result['action']
        obs, reward, done, info = env.step(action)
        prev_reward = reward
        episode_reward += reward
        steps += 1
        activations += np.random.randn(100) * 0.05
    
    executive.step(obs, prev_reward, True, activations)
    executive.reset()
    
    if info['pos'] == info['goal']:
        successes += 1
    
    returns.append(episode_reward)
    
    if (episode + 1) % 10 == 0:
        recent_success = sum(1 for i in range(max(0, episode-9), episode+1) 
                            if returns[i] > 0) / min(10, episode+1)
        print(f"  Episode {episode+1}: return={episode_reward:.3f}, "
              f"avg={np.mean(returns[-10:]):.3f}, success={recent_success:.0%}")

# ============================================================
# RESULTS
# ============================================================
print()
print("="*70)
print("RESULTS")
print("="*70)

early_avg = np.mean(returns[:10])
late_avg = np.mean(returns[-10:])

print(f"Early episodes (1-10):  {early_avg:.3f}")
print(f"Late episodes (86-95):  {late_avg:.3f}")
print(f"Improvement:            {late_avg - early_avg:+.3f}")
print(f"Total successes:        {successes}/95 ({successes/95:.1%})")
print()

stats = executive.get_stats()

print("Regional statistics:")
for region_name, region_stats in stats.items():
    if region_name == 'step_count':
        continue  # Skip step_count
    print(f"  {region_name.capitalize()}:")
    if isinstance(region_stats, dict):
        if 'epsilon' in region_stats:
            print(f"    Epsilon: {region_stats['epsilon']:.4f}")
        if 'n_updates' in region_stats:
            print(f"    Updates: {region_stats['n_updates']}")
        if 'avg_pattern_strength' in region_stats:
            print(f"    Pattern strength: {region_stats['avg_pattern_strength']:.3f}")
        if 'episodic_memories' in region_stats:
            print(f"    Episodic memories: {region_stats['episodic_memories']}")
        if 'resource_usage' in region_stats:
            print(f"    Compute: {region_stats['resource_usage']['compute_ms']:.2f}ms/step")

print()
print("="*70)
print("VALIDATION")
print("="*70)
print()

baseline_2region = 97.9  # From test_integrated_regions.py
current_3region = successes / 95 * 100

print(f"Baseline (2 regions):  {baseline_2region:.1f}%")
print(f"Current (3 regions):   {current_3region:.1f}%")
print()

if successes >= 85:  # ~90%
    print("✓ SUCCESS! 3-region integration works!")
    print("  Striatum + Cortex + Hippocampus validated.")
    print("  Ready for Phase C: Enhanced Executive")
elif successes >= 70:
    print("⚠ PARTIAL SUCCESS")
    print("  3 regions working but performance degraded")
    print("  May need tuning")
else:
    print("✗ NEEDS WORK")
    print("  Performance issues with 3-region coordination")

print("="*70)
