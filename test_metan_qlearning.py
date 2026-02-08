"""
Meta^N with Q-Learning and Curriculum Learning

Complete integration test:
1. Demonstrate optimal path to Meta^N
2. Let Meta^N + Q-learning explore and learn
3. Validate learning improvement
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter
from throng3.core.fractal_stack import FractalStack
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig

print("="*60)
print("Meta^N + Q-Learning + Curriculum Learning")
print("="*60)
print()

# Create pipeline with Q-learning enabled
synapse_config = SynapseConfig(
    use_qlearning=True,
    q_learning_rate=0.3,  # Higher for faster learning from demos
    q_gamma=0.95,
    q_epsilon=0.3,
    q_epsilon_decay=0.99,
)

stack = FractalStack(config={'holographic_dim': 64})
stack.add_layer(NeuronLayer(NeuronConfig(n_neurons=100, n_inputs=2, n_outputs=4)))
stack.add_layer(SynapseOptimizer(synapse_config))
stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))

pipeline = MetaNPipeline(stack)

print(f"Pipeline created with Q-learning enabled")
print(f"  Q-learning rate: {synapse_config.q_learning_rate}")
print(f"  Gamma: {synapse_config.q_gamma}")
print(f"  Epsilon: {synapse_config.q_epsilon}")
print()

env = GridWorldAdapter()

# ============================================================
# PHASE 1: CURRICULUM LEARNING (Demonstrate optimal path)
# ============================================================
print("="*60)
print("PHASE 1: Curriculum Learning (5 demonstrations)")
print("="*60)
print("Demonstrating optimal path: right×4, down×4")
print()

optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]  # right×4, down×4

for demo_ep in range(5):
    obs = env.reset()
    episode_reward = 0
    
    for step, action in enumerate(optimal_actions):
        # Get current state from pipeline
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Take optimal action (override pipeline output)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"  Demo {demo_ep+1}: reward={episode_reward:.3f}, steps={step+1}")

# Get Q-learning stats after demos
meta1 = pipeline.stack.layers[1]
if hasattr(meta1, 'qlearner') and meta1.qlearner is not None:
    stats = meta1.qlearner.get_stats()
    print(f"\nAfter demonstrations:")
    print(f"  Q-learner updates: {stats['n_updates']}")
    print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# ============================================================
# PHASE 2: EXPLORATION & LEARNING
# ============================================================
print(f"\n{'='*60}")
print("PHASE 2: Exploration & Learning (95 episodes)")
print("="*60)
print()

episode_returns = []
episode_lengths = []
successes = 0

for episode in range(95):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 100:
        # Get pipeline output (for other learning rules)
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Use Q-learner with RAW OBSERVATION (not activations!)
        # This is critical - Q-learner trained on [x,y], not 100-dim activations
        if hasattr(meta1, 'qlearner') and meta1.qlearner is not None:
            action = meta1.qlearner.select_action(obs)  # Use raw obs!
        else:
            action = np.argmax(result['output'])
        
        # Take action
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
    
    # CRITICAL: Reset Q-learner at episode end (for epsilon decay)
    if hasattr(meta1, 'qlearner') and meta1.qlearner is not None:
        meta1.qlearner.reset_episode()
    
    # Track success
    if info['pos'] == info['goal']:
        successes += 1
    
    episode_returns.append(episode_reward)
    episode_lengths.append(steps)
    
    if (episode + 1) % 10 == 0:
        recent_returns = episode_returns[-10:]
        recent_lengths = episode_lengths[-10:]
        recent_success = sum(1 for i in range(len(episode_returns)-10, len(episode_returns)) 
                            if episode_returns[i] > 0) / 10
        print(f"  Episode {episode+1}: "
              f"return={episode_reward:.3f}, "
              f"avg_return={np.mean(recent_returns):.3f}, "
              f"avg_length={np.mean(recent_lengths):.1f}, "
              f"success_rate={recent_success:.1%}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")

early_returns = np.mean(episode_returns[:10])
late_returns = np.mean(episode_returns[-10:])
improvement = late_returns - early_returns

print(f"Early episodes (1-10):   {early_returns:.3f}")
print(f"Late episodes (86-95):   {late_returns:.3f}")
print(f"Improvement:             {improvement:+.3f}")
print(f"Total successes:         {successes}/95 ({successes/95:.1%})")

# Get final Q-learning stats
if hasattr(meta1, 'qlearner') and meta1.qlearner is not None:
    stats = meta1.qlearner.get_stats()
    print(f"\nQ-learning stats:")
    print(f"  Epsilon: {stats['epsilon']:.4f}")
    print(f"  Total updates: {stats['n_updates']}")
    print(f"  Avg TD error: {stats['avg_td_error']:.4f}")
    print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

# Get Meta^1 stats
print(f"\nMeta^1 stats:")
print(f"  Active rule: {meta1.active_rule}")
print(f"  STDP usage: {meta1._rule_usage.get('stdp', 0)}")
print(f"  Hebbian usage: {meta1._rule_usage.get('hebbian', 0)}")

if successes / 95 > 0.8:
    print(f"\n✓ SUCCESS! Meta^N + Q-learning + Curriculum works!")
    print(f"  {successes}/95 episodes reached goal ({successes/95:.1%})")
elif successes / 95 > 0.5:
    print(f"\n⚠ Moderate success. {successes}/95 episodes reached goal.")
else:
    print(f"\n✗ Limited success. Only {successes}/95 episodes reached goal.")

print("="*60)
