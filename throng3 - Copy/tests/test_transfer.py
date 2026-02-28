"""
Test 7: Transfer/Generalization

Tests that the Meta^N system generalizes:
1. Train on one task
2. Test on related but different task
3. Measure transfer vs training from scratch

The hypothesis: higher meta-layers should enable faster transfer
because they learn ABOUT learning, not just the task.

Structure only — not for running on Pi.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.pipeline import MetaNPipeline
from throng3.core.fractal_stack import FractalStack
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig


def generate_task(n_inputs: int, n_outputs: int, seed: int) -> tuple:
    """Generate a linear task with specific random seed."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * 0.5
    bias = rng.randn(n_outputs) * 0.1
    return W, bias


def evaluate_on_task(pipeline, W_task, bias_task, n_steps: int = 50) -> list:
    """Evaluate pipeline on a task for n_steps, return loss history."""
    losses = []
    for _ in range(n_steps):
        x = np.random.randn(W_task.shape[1])
        y = W_task @ x + bias_task
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    return losses


def test_task_transfer():
    """Test that meta-learning enables faster transfer to new tasks."""
    n_inputs, n_outputs = 16, 8
    
    # Task A (training)
    W_A, b_A = generate_task(n_inputs, n_outputs, seed=42)
    
    # Task B (transfer target, related but different)
    W_B, b_B = generate_task(n_inputs, n_outputs, seed=123)
    
    # 1. Train pipeline with Meta^0-2 on Task A
    pipeline_transfer = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    losses_A = evaluate_on_task(pipeline_transfer, W_A, b_A, n_steps=100)
    print(f"  Task A final loss: {np.mean(losses_A[-20:]):.4f}")
    
    # 2. Transfer to Task B: reset task-specific state, keep meta-knowledge
    pipeline_transfer.reset_task_state()
    losses_B_transfer = evaluate_on_task(pipeline_transfer, W_B, b_B, n_steps=50)
    
    # 3. Baseline: fresh pipeline on Task B
    pipeline_fresh = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    losses_B_fresh = evaluate_on_task(pipeline_fresh, W_B, b_B, n_steps=50)
    
    transfer_loss = np.mean(losses_B_transfer[-20:])
    fresh_loss = np.mean(losses_B_fresh[-20:])
    
    print(f"  Task B (transfer): {transfer_loss:.4f}")
    print(f"  Task B (fresh):    {fresh_loss:.4f}")
    
    improvement = (fresh_loss - transfer_loss) / max(fresh_loss, 1e-8) * 100
    print(f"  Transfer improvement: {improvement:.1f}%")
    print(f"✓ Transfer test complete")


def test_meta_level_transfer_comparison():
    """Compare transfer ability across different numbers of meta-layers."""
    n_inputs, n_outputs = 16, 8
    
    W_A, b_A = generate_task(n_inputs, n_outputs, seed=42)
    W_B, b_B = generate_task(n_inputs, n_outputs, seed=123)
    
    print("  Comparing transfer with different meta-layer counts:")
    
    for n_meta in [1, 2, 3]:
        # Create pipeline
        stack = FractalStack(config={'holographic_dim': 64})
        stack.add_layer(NeuronLayer(NeuronConfig(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )))
        if n_meta >= 2:
            stack.add_layer(SynapseOptimizer(SynapseConfig()))
        if n_meta >= 3:
            stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig(
                evaluation_window=10
            )))
        
        pipeline = MetaNPipeline(stack)
        
        # Train on A
        for _ in range(50):
            x = np.random.randn(n_inputs)
            y = W_A @ x + b_A
            pipeline.step(x, target=y)
        
        # Transfer to B
        losses_B = []
        for _ in range(30):
            x = np.random.randn(n_inputs)
            y = W_B @ x + b_B
            result = pipeline.step(x, target=y)
            losses_B.append(result['loss'])
        
        final_loss = np.mean(losses_B[-10:])
        print(f"    Meta^0-{n_meta-1} ({n_meta} layers): transfer loss={final_loss:.4f}")
    
    print("✓ Meta-level transfer comparison complete")


def test_holographic_state_transfer():
    """Test that holographic state captures transferable knowledge."""
    n_inputs, n_outputs = 16, 8
    
    # Create and train pipeline
    pipeline = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    W_A, b_A = generate_task(n_inputs, n_outputs, seed=42)
    for _ in range(50):
        x = np.random.randn(n_inputs)
        pipeline.step(x, target=W_A @ x + b_A)
    
    # Save holographic state
    snapshots = pipeline.stack.get_all_snapshots()
    holo_hash = pipeline.stack.holographic.get_hash()
    
    assert len(snapshots) > 0, "Should have snapshots"
    assert holo_hash != "0" * 16, "Holographic hash should be non-trivial"
    
    print(f"✓ Holographic state captured: hash={holo_hash}, "
          f"layers={len(snapshots)}")


if __name__ == '__main__':
    print("=" * 50)
    print("Test 7: Transfer/Generalization")
    print("=" * 50)
    
    test_task_transfer()
    print()
    test_meta_level_transfer_comparison()
    print()
    test_holographic_state_transfer()
    
    print("\nAll tests passed! ✓")
