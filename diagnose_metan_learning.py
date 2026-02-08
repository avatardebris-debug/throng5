"""
Diagnostic: Check if Meta^N learning mechanisms are being activated.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter

print("="*60)
print("Meta^N Learning Mechanisms Diagnostic")
print("="*60)

# Create minimal pipeline for debugging
print("\nCreating minimal Meta^N pipeline (100 neurons)...")
pipeline = MetaNPipeline.create_minimal(
    n_neurons=100,
    n_inputs=2,
    n_outputs=4
)

print(f"✓ Pipeline created")
print(f"  Layers: {len(pipeline.stack.layers)}")

# Get references to layers
layers_dict = pipeline.stack.layers
meta0 = layers_dict[0]  # NeuronLayer
meta1 = layers_dict[1]  # SynapseOptimizer
meta2 = layers_dict[2]  # LearningRuleSelector

print(f"\nMeta^1 (SynapseOptimizer) configuration:")
print(f"  Active rule: {meta1.active_rule}")
print(f"  Learning rate: {meta1.synapse_config.learning_rate}")
print(f"  Dopamine modulation: {meta1.synapse_config.dopamine_modulation}")
print(f"  Prune interval: {meta1.synapse_config.prune_interval}")

# Run a few episodes and track what's happening
print(f"\n{'='*60}")
print("Running 5 episodes on GridWorld...")
print(f"{'='*60}")

env = GridWorldAdapter()

for episode in range(5):
    print(f"\n--- Episode {episode+1} ---")
    obs = env.reset()
    episode_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 20:
        # Step pipeline
        result = pipeline.step(
            input_data=obs,
            reward=episode_reward,
            episode_return=episode_reward
        )
        
        # Get action
        output = result['output']
        action = np.argmax(output)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
    
    # Check what happened in Meta^1
    print(f"  Steps: {steps}, Reward: {episode_reward:.3f}")
    print(f"  Meta^1 active rule: {meta1.active_rule}")
    print(f"  Meta^1 rule usage: {meta1._rule_usage}")
    print(f"  Meta^1 dopamine level: {meta1.dopamine.level:.3f}")
    print(f"  Meta^1 total weight updates: {meta1._total_weight_updates}")
    
    # Check if weights are changing
    if hasattr(meta0, 'W_recurrent'):
        W = meta0.W_recurrent
        print(f"  Meta^0 weight stats: mean={np.mean(np.abs(W)):.4f}, max={np.max(np.abs(W)):.4f}")
        print(f"  Meta^0 active connections: {np.sum(np.abs(W) > 1e-6)}/{W.size}")

print(f"\n{'='*60}")
print("Diagnostic Summary")
print(f"{'='*60}")

print(f"\nLearning mechanisms status:")
print(f"  STDP: {'✓ ACTIVE' if meta1._rule_usage['stdp'] > 0 else '✗ INACTIVE'}")
print(f"  Hebbian: {'✓ ACTIVE' if meta1._rule_usage['hebbian'] > 0 else '✗ INACTIVE'}")
print(f"  Dopamine modulation: {'✓ ENABLED' if meta1.synapse_config.dopamine_modulation else '✗ DISABLED'}")
print(f"  Nash pruning: Runs every {meta1.synapse_config.prune_interval} steps")

print(f"\nWeight update stats:")
print(f"  Total updates: {meta1._total_weight_updates}")
print(f"  STDP usage: {meta1._rule_usage['stdp']}")
print(f"  Hebbian usage: {meta1._rule_usage['hebbian']}")

print(f"\nPotential issues:")
issues = []

if meta1._total_weight_updates == 0:
    issues.append("✗ No weight updates happening - Meta^1 not being called?")

if meta1._rule_usage['stdp'] == 0 and meta1._rule_usage['hebbian'] == 0:
    issues.append("✗ Neither STDP nor Hebbian being used")

if meta1.dopamine.level == 0:
    issues.append("⚠ Dopamine level is zero - reward signal not reaching Meta^1?")

if hasattr(meta0, 'W_recurrent'):
    if np.all(np.abs(meta0.W_recurrent) < 1e-6):
        issues.append("✗ All weights near zero - learning not effective")

if not issues:
    issues.append("✓ No obvious issues detected")

for issue in issues:
    print(f"  {issue}")

print("="*60)
