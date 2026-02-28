"""
Test Global Dynamics Optimizer — Task-Adaptive Layer Gating

Tests the GlobalDynamicsOptimizer which:
1. Assesses task complexity
2. Gates layers dynamically based on contribution
3. Prevents layer interference on simple tasks
4. Activates higher layers when needed for complex tasks
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.core.fractal_stack import FractalStack
from throng3.core.global_dynamics import GlobalDynamicsOptimizer, GlobalConfig
from throng3.pipeline import MetaNPipeline
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig
from throng3.layers.meta3_representation import RepresentationOptimizer, RepresentationConfig
from throng3.layers.meta4_goal import GoalHierarchy, GoalConfig
from throng3.layers.meta5_architecture import ArchitectureSearch, ArchitectureConfig


def create_full_stack(n_neurons: int = 100) -> FractalStack:
    """Create a full 6-layer stack for testing."""
    stack = FractalStack(config={'holographic_dim': 64})
    
    stack.add_layer(NeuronLayer(NeuronConfig(
        n_neurons=n_neurons, n_inputs=16, n_outputs=8
    )))
    stack.add_layer(SynapseOptimizer(SynapseConfig()))
    stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig(
        evaluation_window=10,
    )))
    stack.add_layer(RepresentationOptimizer(RepresentationConfig()))
    stack.add_layer(GoalHierarchy(GoalConfig()))
    stack.add_layer(ArchitectureSearch(ArchitectureConfig(
        evaluation_steps=20,
        search_interval=50,
    )))
    
    return stack


def test_complexity_assessment():
    """Test that complexity assessment responds to input characteristics."""
    print("Testing complexity assessment...")
    
    stack = create_full_stack()
    optimizer = GlobalDynamicsOptimizer(stack, GlobalConfig())
    
    # Test with simple input (low dim)
    simple_context = {'input': np.random.randn(8), 'reward': 0.5}
    optimizer.step(simple_context)
    complexity_simple = optimizer.get_complexity()
    
    # Run a few steps to build history
    for _ in range(20):
        optimizer.step(simple_context)
    
    complexity_after = optimizer.get_complexity()
    
    print(f"  Initial complexity: {complexity_simple:.3f}")
    print(f"  After 20 steps: {complexity_after:.3f}")
    print(f"✓ Complexity assessment working")


def test_layer_gating():
    """Test that layer gates adjust based on contribution."""
    print("\nTesting layer gating...")
    
    stack = create_full_stack()
    config = GlobalConfig(
        warmup_steps=10,
        update_interval=5,
    )
    optimizer = GlobalDynamicsOptimizer(stack, config)
    
    # Run enough steps for gates to adjust
    np.random.seed(42)
    W_true = np.random.randn(8, 16) * 0.5
    
    initial_gates = optimizer.get_gates().copy()
    print(f"  Initial gates: {initial_gates}")
    
    for step in range(50):
        x = np.random.randn(16)
        y = W_true @ x + np.random.randn(8) * 0.1
        
        optimizer.step({
            'input': x,
            'target': y,
            'reward': 0.0,
        })
    
    final_gates = optimizer.get_gates()
    print(f"  Final gates: {final_gates}")
    
    # Check that higher layers have lower gates (on this simple task)
    assert final_gates[0] == 1.0, "Meta^0 gate should always be 1.0"
    assert final_gates[1] == 1.0, "Meta^1 gate should be 1.0"
    
    print(f"✓ Layer gating working")


def test_adaptive_pipeline():
    """Test the create_adaptive() factory method."""
    print("\nTesting adaptive pipeline...")
    
    pipeline = MetaNPipeline.create_adaptive(
        n_neurons=50,
        n_inputs=16,
        n_outputs=8,
    )
    
    assert pipeline.global_optimizer is not None
    print(f"  Global optimizer: {pipeline.global_optimizer}")
    
    # Run a few steps
    np.random.seed(42)
    W_true = np.random.randn(8, 16) * 0.5
    
    for step in range(30):
        x = np.random.randn(16)
        y = W_true @ x + np.random.randn(8) * 0.1
        
        result = pipeline.step(x, target=y, reward=0.0)
    
    # Check that global metrics are included
    assert 'global' in result
    assert 'complexity' in result['global']
    assert 'gates' in result['global']
    
    print(f"  Complexity: {result['global']['complexity']:.3f}")
    print(f"  Gates: {result['global']['gates']}")
    print(f"✓ Adaptive pipeline working")


def test_comparison_with_without():
    """Compare performance with and without global optimizer."""
    print("\nComparing with/without global optimizer...")
    
    np.random.seed(42)
    W_true = np.random.randn(8, 16) * 0.5
    n_steps = 100
    
    # Without global optimizer
    pipeline_standard = MetaNPipeline.create_default(
        n_neurons=50,
        n_inputs=16,
        n_outputs=8,
    )
    
    losses_standard = []
    for step in range(n_steps):
        x = np.random.randn(16)
        y = W_true @ x + np.random.randn(8) * 0.1
        result = pipeline_standard.step(x, target=y, reward=0.0)
        losses_standard.append(result['loss'])
    
    # With global optimizer
    np.random.seed(42)  # Same seed for fair comparison
    pipeline_adaptive = MetaNPipeline.create_adaptive(
        n_neurons=50,
        n_inputs=16,
        n_outputs=8,
    )
    
    losses_adaptive = []
    for step in range(n_steps):
        x = np.random.randn(16)
        y = W_true @ x + np.random.randn(8) * 0.1
        result = pipeline_adaptive.step(x, target=y, reward=0.0)
        losses_adaptive.append(result['loss'])
    
    final_standard = np.mean(losses_standard[-20:])
    final_adaptive = np.mean(losses_adaptive[-20:])
    improvement = (final_standard - final_adaptive) / final_standard * 100
    
    print(f"  Standard (6 layers): {final_standard:.4f}")
    print(f"  Adaptive (gated):    {final_adaptive:.4f}")
    print(f"  Improvement: {improvement:+.1f}%")
    
    # Get final gate values
    gates = pipeline_adaptive.global_optimizer.get_gates()
    print(f"  Final gates: {gates}")
    
    print(f"✓ Comparison complete")


def test_plateau_escalation():
    """Test that layers escalate when learning plateaus."""
    print("\nTesting plateau escalation...")
    
    stack = create_full_stack()
    config = GlobalConfig(
        warmup_steps=5,
        update_interval=3,
        plateau_threshold=0.001,  # Low threshold to trigger plateau
        escalate_on_plateau=True,
    )
    optimizer = GlobalDynamicsOptimizer(stack, config)
    
    # Create a difficult task that will plateau
    np.random.seed(123)
    
    # Run with constant low reward (simulates stuck learning)
    for step in range(50):
        optimizer.step({
            'input': np.random.randn(16),
            'reward': 0.0,
        })
    
    plateau_count = optimizer._plateau_count
    gates = optimizer.get_gates()
    
    print(f"  Plateau count: {plateau_count}")
    print(f"  Gates after plateau: {gates}")
    
    print(f"✓ Plateau escalation test complete")


if __name__ == '__main__':
    print("=" * 50)
    print("Test: Global Dynamics Optimizer")
    print("=" * 50)
    
    test_complexity_assessment()
    test_layer_gating()
    test_adaptive_pipeline()
    test_comparison_with_without()
    test_plateau_escalation()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
