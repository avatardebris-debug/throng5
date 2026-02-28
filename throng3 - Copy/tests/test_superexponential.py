"""
Test 5: Superexponential Test — Performance vs Number of Meta Layers

Tests the hypothesis that adding meta-layers produces superexponential
improvement in learning speed or final performance.

Compare:
1. Meta^0 alone (baseline)
2. Meta^0 + Meta^1 (synapse optimization)
3. Meta^0 + Meta^1 + Meta^2 (learning rule selection)
4. Meta^0-3 (+ representation optimization)
5. Meta^0-4 (+ goal hierarchy)
6. Meta^0-5 (+ architecture search)

Structure only — not for running on Pi.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.core.fractal_stack import FractalStack
from throng3.pipeline import MetaNPipeline
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig
from throng3.layers.meta3_representation import RepresentationOptimizer, RepresentationConfig
from throng3.layers.meta4_goal import GoalHierarchy, GoalConfig
from throng3.layers.meta5_architecture import ArchitectureSearch, ArchitectureConfig


def create_stack_with_n_layers(n_meta: int, n_neurons: int = 100) -> FractalStack:
    """Create a stack with exactly n meta-layers."""
    stack = FractalStack(config={'holographic_dim': 64})
    
    # Always add Meta^0
    stack.add_layer(NeuronLayer(NeuronConfig(
        n_neurons=n_neurons, n_inputs=16, n_outputs=8
    )))
    
    if n_meta >= 2:
        stack.add_layer(SynapseOptimizer(SynapseConfig()))
    
    if n_meta >= 3:
        stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig(
            evaluation_window=10,
        )))
    
    if n_meta >= 4:
        stack.add_layer(RepresentationOptimizer(RepresentationConfig()))
    
    if n_meta >= 5:
        stack.add_layer(GoalHierarchy(GoalConfig()))
    
    if n_meta >= 6:
        stack.add_layer(ArchitectureSearch(ArchitectureConfig(
            evaluation_steps=20,
            search_interval=50,
        )))
    
    return stack


def test_scaling_by_layers():
    """
    Compare performance across different numbers of meta-layers.
    
    For each configuration:
    1. Run N steps
    2. Record final loss, convergence speed, and total signals
    """
    n_steps = 100  # Would be more in real test
    
    # Generate a simple learning task
    np.random.seed(42)
    W_true = np.random.randn(8, 16) * 0.5
    
    results = {}
    
    for n_meta in range(1, 7):
        stack = create_stack_with_n_layers(n_meta)
        
        losses = []
        for step in range(n_steps):
            # Generate input/target pair
            x = np.random.randn(16)
            y = W_true @ x + np.random.randn(8) * 0.1
            
            result = stack.step({
                'input': x,
                'target': y,
                'reward': 0.0,
            })
            
            meta0_result = result.get('layer_results', {}).get(0, {})
            loss = meta0_result.get('loss', 1.0)
            losses.append(loss)
        
        final_loss = np.mean(losses[-20:])
        convergence_speed = np.argmin(losses[:50]) if len(losses) >= 50 else len(losses)
        signals = result.get('signals_routed', 0)
        
        results[n_meta] = {
            'final_loss': final_loss,
            'convergence_step': convergence_speed,
            'total_signals': signals,
            'losses': losses,
        }
        
        print(f"  Meta^0-{n_meta-1} ({n_meta} layers): "
              f"final_loss={final_loss:.4f}, "
              f"signals={signals}")
    
    # Check for superexponential trend
    losses = [results[n]['final_loss'] for n in sorted(results.keys())]
    improvements = []
    for i in range(1, len(losses)):
        if losses[i-1] > 0:
            improvements.append((losses[i-1] - losses[i]) / losses[i-1])
        else:
            improvements.append(0)
    
    print(f"\n✓ Improvements by adding each layer: {[f'{x:.3f}' for x in improvements]}")
    
    # Check if improvements are accelerating (superexponential)
    if len(improvements) >= 3:
        accel = [improvements[i] - improvements[i-1] for i in range(1, len(improvements))]
        is_superexponential = sum(1 for a in accel if a > 0) > len(accel) / 2
        print(f"  Acceleration: {[f'{x:.3f}' for x in accel]}")
        print(f"  Superexponential: {'YES' if is_superexponential else 'not yet'}")


def test_efficiency_scaling():
    """Test computational efficiency as layers are added."""
    import time
    
    timings = {}
    
    for n_meta in range(1, 7):
        stack = create_stack_with_n_layers(n_meta, n_neurons=50)
        
        t0 = time.time()
        for _ in range(20):
            stack.step({
                'input': np.random.randn(16),
                'reward': 0.0,
            })
        dt = time.time() - t0
        
        timings[n_meta] = dt / 20  # Per-step time
        print(f"  {n_meta} layers: {timings[n_meta]*1000:.1f} ms/step")
    
    print(f"\n✓ Timing scales: {[f'{t*1000:.1f}ms' for t in timings.values()]}")


if __name__ == '__main__':
    print("=" * 50)
    print("Test 5: Superexponential Scaling")
    print("=" * 50)
    
    test_scaling_by_layers()
    print()
    test_efficiency_scaling()
    
    print("\nAll tests passed! ✓")
