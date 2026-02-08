"""
Test Q-learning integration with Meta^N
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter
from throng3.layers.meta1_synapse import SynapseConfig

print("="*60)
print("Q-Learning Integration Test")
print("="*60)
print("Testing Q-learning with Meta^N on GridWorld")
print()

# Create pipeline manually with Q-learning enabled
from throng3.core.fractal_stack import FractalStack
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig

synapse_config = SynapseConfig(
    use_qlearning=True,
    q_learning_rate=0.1,
    q_gamma=0.95,
    q_epsilon=0.2,
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
episode_returns = []
episode_lengths = []

print("Training for 100 episodes...")

for episode in range(100):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:
        # Get pipeline output
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Use Q-values for action selection if available
        q_values = result.get('q_values')
        if q_values is not None and np.any(q_values != None):
            # ε-greedy action selection using Q-values
            meta1 = pipeline.stack.layers[1]
            if hasattr(meta1, 'qlearner') and meta1.qlearner is not None:
                # Use Q-learner's epsilon-greedy selection
                activations = result.get('activations', np.zeros(100))
                action = meta1.qlearner.select_action(activations)
            else:
                action = np.argmax(result['output'])
        else:
            action = np.argmax(result['output'])
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
    
    episode_returns.append(episode_reward)
    episode_lengths.append(steps)
    
    if (episode + 1) % 10 == 0:
        recent_returns = episode_returns[-10:]
        recent_lengths = episode_lengths[-10:]
        print(f"  Episode {episode+1}: "
              f"return={episode_reward:.3f}, "
              f"avg_return={np.mean(recent_returns):.3f}, "
              f"avg_length={np.mean(recent_lengths):.1f}")

# Analysis
early_returns = np.mean(episode_returns[:10])
late_returns = np.mean(episode_returns[-10:])
improvement = late_returns - early_returns

print(f"\n{'='*60}")
print("Results:")
print(f"{'='*60}")
print(f"Early episodes (1-10):  {early_returns:.3f}")
print(f"Late episodes (91-100): {late_returns:.3f}")
print(f"Improvement:            {improvement:+.3f}")

# Get Q-learning stats
meta1 = pipeline.stack.layers[1]
if hasattr(meta1, 'qlearner') and meta1.qlearner is not None:
    stats = meta1.qlearner.get_stats()
    print(f"\nQ-learning stats:")
    print(f"  Epsilon: {stats['epsilon']:.4f}")
    print(f"  Updates: {stats['n_updates']}")
    print(f"  Avg TD error: {stats['avg_td_error']:.4f}")
    print(f"  Mean Q weight: {stats['mean_q_weight']:.6f}")

if improvement > 0.1:
    print(f"\n✓ SUCCESS! Q-learning shows improvement!")
elif improvement > 0.01:
    print(f"\n⚠ Slight improvement, may need tuning.")
else:
    print(f"\n✗ No significant learning. Check configuration.")

print("="*60)
