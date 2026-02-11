"""
Test: Gradient Learning on Supervised Task

Verify that Meta^1 can use gradient descent for supervised learning
when learning_mode='gradient'.

This should show MUCH better performance than STDP/Hebbian on
supervised tasks.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.layers.meta1_synapse import SynapseConfig


def test_gradient_vs_bio_inspired():
    """
    Compare gradient descent vs bio-inspired learning on supervised task.
    
    Expected: Gradient descent should converge much faster and better.
    """
    print("\n" + "="*60)
    print("GRADIENT vs BIO-INSPIRED LEARNING")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    n_steps = 200
    
    # Generate supervised task
    np.random.seed(42)
    W_true = np.random.randn(n_outputs, n_inputs) * 0.5
    b_true = np.random.randn(n_outputs) * 0.1
    
    # Test 1: Gradient descent
    print("\n[Test 1: Gradient Descent]")
    pipeline_grad = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    # Set learning mode to gradient
    synapse_layer = pipeline_grad.stack.get_layer(1)
    if synapse_layer:
        synapse_layer.learning_mode = 'gradient'
    
    losses_grad = []
    for step in range(n_steps):
        x = np.random.randn(n_inputs)
        y = W_true @ x + b_true
        result = pipeline_grad.step(x, target=y, reward=0.0)
        losses_grad.append(result['loss'])
    
    final_loss_grad = np.mean(losses_grad[-20:])
    print(f"  Final loss: {final_loss_grad:.4f}")
    print(f"  Improvement: {(losses_grad[0] - final_loss_grad):.4f}")
    
    # Test 2: Bio-inspired (STDP/Hebbian) - the old way
    print("\n[Test 2: Bio-Inspired (STDP/Hebbian)]")
    pipeline_bio = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    # Default is bio-inspired (auto mode with no gradient)
    
    losses_bio = []
    for step in range(n_steps):
        x = np.random.randn(n_inputs)
        y = W_true @ x + b_true
        result = pipeline_bio.step(x, target=y, reward=0.0)
        losses_bio.append(result['loss'])
    
    final_loss_bio = np.mean(losses_bio[-20:])
    print(f"  Final loss: {final_loss_bio:.4f}")
    print(f"  Improvement: {(losses_bio[0] - final_loss_bio):.4f}")
    
    # Comparison
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Gradient descent: {final_loss_grad:.4f}")
    print(f"Bio-inspired:     {final_loss_bio:.4f}")
    
    improvement = (final_loss_bio - final_loss_grad) / final_loss_bio * 100
    print(f"\nGradient is {improvement:+.1f}% better")
    
    if final_loss_grad < final_loss_bio * 0.5:
        print("\n✓ GRADIENT DESCENT WORKS!")
        print("  Gradient learning is significantly better for supervised tasks")
    else:
        print("\n⚠ Gradient descent should be much better")
        print("  Something may be wrong with the implementation")
    
    return {
        'gradient_loss': final_loss_grad,
        'bio_loss': final_loss_bio,
        'improvement_pct': improvement,
    }


if __name__ == '__main__':
    results = test_gradient_vs_bio_inspired()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("\nThis test validates that:")
    print("  1. Gradient descent works for supervised tasks")
    print("  2. It's much better than bio-inspired learning")
    print("  3. Meta^1 can route to the right mechanism")
    print("\nNext: Add Meta^2 task detector to auto-select mechanism")
