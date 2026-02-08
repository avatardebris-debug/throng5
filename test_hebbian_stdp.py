"""
Test if Hebbian+STDP helps learning over longer training.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter

print("="*60)
print("Hebbian+STDP Learning Test (50 episodes)")
print("="*60)

# Create minimal pipeline
pipeline = MetaNPipeline.create_minimal(
    n_neurons=100,
    n_inputs=2,
    n_outputs=4
)

meta1 = pipeline.stack.layers[1]
print(f"\nMeta^1 active rule: {meta1.active_rule}")
print(f"Learning mechanisms: STDP + Hebbian + Dopamine modulation")

env = GridWorldAdapter()
episode_returns = []
episode_lengths = []

print(f"\nTraining for 50 episodes...")

for episode in range(50):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        output = result['output']
        action = np.argmax(output)
        
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
print(f"Late episodes (41-50):  {late_returns:.3f}")
print(f"Improvement:            {improvement:+.3f}")

print(f"\nMeta^1 final stats:")
print(f"  STDP usage: {meta1._rule_usage['stdp']}")
print(f"  Hebbian usage: {meta1._rule_usage['hebbian']}")
print(f"  Dopamine level: {meta1.dopamine.level:.3f}")

if improvement > 0.1:
    print(f"\n✓ Learning detected! Hebbian+STDP is helping.")
elif improvement > 0.01:
    print(f"\n⚠ Slight improvement, but not strong learning.")
else:
    print(f"\n✗ No significant learning. Need additional mechanisms (Q-learning?).")

print("="*60)
