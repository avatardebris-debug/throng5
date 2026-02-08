"""
Q-Learning with CORRECT State Representation

This test fixes the state representation bug - Q-learner uses raw observations
instead of 100-dim neuron activations.

Expected result: ~100% success (matching standalone)
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter
from throng3.core.fractal_stack import FractalStack
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig
from throng3.learning.qlearning import QLearner, QLearningConfig

print("="*60)
print("Q-Learning with CORRECT State Representation")
print("="*60)
print("Using raw observations [x,y] instead of neuron activations")
print()

# Create pipeline (Q-learning disabled in synapse - we'll manage separately)
synapse_config = SynapseConfig(
    use_qlearning=False,  # Disable broken integration
    default_rule='both',  # Keep STDP+Hebbian for their benefits
)

stack = FractalStack(config={'holographic_dim': 64})
stack.add_layer(NeuronLayer(NeuronConfig(n_neurons=100, n_inputs=2, n_outputs=4)))
stack.add_layer(SynapseOptimizer(synapse_config))
stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))

pipeline = MetaNPipeline(stack)

# Create Q-learner with CORRECT state dimension (2D observations)
qlearner = QLearner(
    n_states=2,  # Raw [x, y] observation
    n_actions=4,
    config=QLearningConfig(
        learning_rate=0.3,
        gamma=0.95,
        epsilon=0.3,
        epsilon_decay=0.99,
    )
)

print(f"Pipeline + Separate Q-learner created:")
print(f"  Q-learner state dim: 2 (raw observations)")
print(f"  STDP/Hebbian: ENABLED (on neurons)")
print()

env = GridWorldAdapter()

# ============================================================
# PHASE 1: CURRICULUM (5 demonstrations)
# ============================================================
print("="*60)
print("PHASE 1: Curriculum Learning (5 demonstrations)")
print("="*60)

optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]

for demo_ep in range(5):
    obs = env.reset()
    episode_reward = 0
    prev_obs = None
    prev_action = None
    
    for step, action in enumerate(optimal_actions):
        # Pipeline processes for bio-inspired learning
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Q-learner update with RAW observations
        if prev_obs is not None:
            qlearner.update(prev_obs, prev_action, episode_reward, obs, False)
        
        prev_obs = obs.copy()
        prev_action = action
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            qlearner.update(prev_obs, prev_action, reward, obs, True)
            break
    
    qlearner.reset_episode()
    print(f"  Demo {demo_ep+1}: reward={episode_reward:.3f}")

stats = qlearner.get_stats()
print(f"\nAfter demonstrations:")
print(f"  Q-learner updates: {stats['n_updates']}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# ============================================================
# PHASE 2: EXPLORATION & LEARNING (95 episodes)
# ============================================================
print(f"\n{'='*60}")
print("PHASE 2: Exploration & Learning (95 episodes)")
print("="*60)

episode_returns = []
successes = 0

for episode in range(95):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    prev_obs = None
    prev_action = None
    
    while not done and steps < 100:
        # Pipeline processes (bio-inspired learning continues)
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Q-learner selects action using RAW observations
        action = qlearner.select_action(obs, explore=True)
        
        # Q-learning update
        if prev_obs is not None:
            qlearner.update(prev_obs, prev_action, episode_reward, obs, False)
        
        prev_obs = obs.copy()
        prev_action = action
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
    
    # Final update
    if prev_obs is not None:
        qlearner.update(prev_obs, prev_action, reward, obs, done=True)
    
    qlearner.reset_episode()
    
    if info['pos'] == info['goal']:
        successes += 1
    
    episode_returns.append(episode_reward)
    
    if (episode + 1) % 10 == 0:
        recent = episode_returns[-10:]
        success_rate = sum(1 for r in recent if r > 0) / 10
        print(f"  Episode {episode+1}: return={episode_reward:.3f}, "
              f"avg={np.mean(recent):.3f}, success={success_rate:.0%}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

early = np.mean(episode_returns[:10])
late = np.mean(episode_returns[-10:])
improvement = late - early

print(f"Early episodes (1-10):  {early:.3f}")
print(f"Late episodes (86-95):  {late:.3f}")
print(f"Improvement:            {improvement:+.3f}")
print(f"Total successes:        {successes}/95 ({successes/95:.1%})")

stats = qlearner.get_stats()
print(f"\nQ-learning stats:")
print(f"  Epsilon: {stats['epsilon']:.4f}")
print(f"  Updates: {stats['n_updates']}")
print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# ============================================================
# COMPARISON
# ============================================================
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")

print(f"\n| Configuration              | Success Rate |")
print(f"|----------------------------|--------------|")
print(f"| Standalone Q-learning      | 100%         |")
print(f"| Mixed (broken integration) | 29.5%        |")
print(f"| Q-only pipeline (broken)   | 8.4%         |")
print(f"| Fixed Q + Bio (this test)  | {successes/95:.1%}         |")

if successes / 95 > 0.9:
    print(f"\n✓ STATE REPRESENTATION FIX CONFIRMED!")
    print(f"  Q-learning works when given raw observations")
    print(f"  → Throng3.5 regional architecture is the right path")
    print(f"  → Each region needs appropriate state representation")
elif successes / 95 > 0.5:
    print(f"\n⚠ PARTIAL FIX - better but not 100%")
    print(f"  May need further tuning")
else:
    print(f"\n✗ FIX INSUFFICIENT - other issues present")

print("="*60)
