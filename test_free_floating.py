"""
Test free-floating neurons to see if they can learn GridWorld.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter
from throng3.layers.free_floating import FreeFloatingLayer, FreeFloatingConfig

print("="*60)
print("Free-Floating Neurons Test")
print("="*60)
print("Testing if neurons outside holographic structure can learn")
print()

# Create pipeline with free-floating layer
pipeline = MetaNPipeline.create_minimal(
    n_neurons=100,
    n_inputs=2,
    n_outputs=4
)

# Add free-floating layer
ff_config = FreeFloatingConfig(
    n_neurons=256,
    learning_rate=0.02,
    sparsity=0.9,
    reward_modulation=True,
    hebbian_learning=True
)
ff_layer = FreeFloatingLayer(ff_config)

# Add free-floating layer to stack
pipeline.stack.layers[0.5] = ff_layer  # Insert at level 0.5 (between Meta^0 and Meta^1)

print(f"Pipeline layers: {len(pipeline.stack.layers)}")
for level, layer in sorted(pipeline.stack.layers.items()):
    print(f"  Level {level}: {layer.name}")
print()

env = GridWorldAdapter()
episode_returns = []
episode_lengths = []

print("Training for 50 episodes...")

for episode in range(50):
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:
        # Get base output from pipeline
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        base_output = result['output']
        
        # Add free-floating contribution
        ff_contribution = ff_layer.get_output_contribution(4)
        combined_output = base_output + ff_contribution * 0.5
        
        action = np.argmax(combined_output)
        
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Update free-floating output weights based on reward
        if reward != 0:
            output_error = np.zeros(4)
            output_error[action] = reward
            ff_layer.update_output_weights(output_error, learning_rate=0.01)
        
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

print(f"\nFree-floating layer stats:")
last_result = ff_layer.optimize({'holographic_state': np.zeros(128), 'reward': 0})
print(f"  Sparsity: {last_result['sparsity']:.3f}")
print(f"  Active neurons: {np.sum(np.abs(ff_layer.activations) > 0.1)}/{ff_layer.n}")
print(f"  Mean weight: {np.mean(np.abs(ff_layer.W_local)):.6f}")

if improvement > 0.1:
    print(f"\n✓ BREAKTHROUGH! Free-floating neurons ARE learning!")
    print(f"  This confirms holographic interference hypothesis.")
elif improvement > 0.01:
    print(f"\n⚠ Slight improvement, free-floating neurons may help.")
else:
    print(f"\n✗ No significant learning. May need additional mechanisms.")

print("="*60)
