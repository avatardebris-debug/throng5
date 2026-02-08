"""
STDP + Hebbian ONLY (No Q-Learning) - Baseline Test

Test whether bio-inspired learning rules can learn GridWorld on their own.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter
from throng3.core.fractal_stack import FractalStack
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig

print("="*60)
print("STDP + Hebbian ONLY (No Q-Learning)")
print("="*60)
print("Testing if bio-inspired rules can learn GridWorld alone")
print()

# Create pipeline WITHOUT Q-learning
synapse_config = SynapseConfig(
    use_qlearning=False,  # DISABLE Q-learning
    default_rule='both',  # STDP + Hebbian enabled
)

stack = FractalStack(config={'holographic_dim': 64})
stack.add_layer(NeuronLayer(NeuronConfig(n_neurons=100, n_inputs=2, n_outputs=4)))
stack.add_layer(SynapseOptimizer(synapse_config))
stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))

pipeline = MetaNPipeline(stack)

print(f"Pipeline created:")
print(f"  Q-learning: DISABLED")
print(f"  STDP: ENABLED")
print(f"  Hebbian: ENABLED")
print()

env = GridWorldAdapter()

# ============================================================
# PHASE 1: CURRICULUM (demonstrate optimal path)
# ============================================================
print("="*60)
print("PHASE 1: Curriculum Learning (5 demonstrations)")
print("="*60)

optimal_actions = [3, 3, 3, 3, 1, 1, 1, 1]  # right×4, down×4

for demo_ep in range(5):
    obs = env.reset()
    episode_reward = 0
    
    for step, action in enumerate(optimal_actions):
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    print(f"  Demo {demo_ep+1}: reward={episode_reward:.3f}")

# ============================================================
# PHASE 2: EXPLORATION (bio-inspired learning)
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
    
    while not done and steps < 100:
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Use pipeline output for action
        action = np.argmax(result['output'])
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
    
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

meta1 = pipeline.stack.layers[1]
print(f"\nMeta^1 stats:")
print(f"  Active rule: {meta1.active_rule}")
print(f"  STDP usage: {meta1._rule_usage.get('stdp', 0)}")
print(f"  Hebbian usage: {meta1._rule_usage.get('hebbian', 0)}")

if successes / 95 > 0.8:
    print(f"\n✓ STDP+Hebbian can learn GridWorld!")
elif successes / 95 > 0.3:
    print(f"\n⚠ Partial learning - bio-inspired rules help but not sufficient")
else:
    print(f"\n✗ Bio-inspired rules alone cannot solve GridWorld")
    print(f"  (This is expected - they don't understand goals)")

print("="*60)
