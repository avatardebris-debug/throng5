"""
Test 1: Single-Layer Self-Optimization Baseline

Tests that Meta^0 (NeuronLayer) can process input and produce output,
and that basic self-optimization works at the weight level.

NOT meant to be run on Pi — structure only.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.core.fractal_stack import FractalStack


def test_neuron_layer_forward():
    """Test that NeuronLayer produces output from input."""
    config = NeuronConfig(n_neurons=100, n_inputs=16, n_outputs=8)
    layer = NeuronLayer(config)
    
    # Forward pass
    input_data = np.random.randn(16)
    output = layer.forward(input_data)
    
    assert output.shape == (8,), f"Expected shape (8,), got {output.shape}"
    assert not np.all(output == 0), "Output should not be all zeros"
    print("✓ Forward pass produces valid output")


def test_neuron_layer_optimize():
    """Test that optimize() runs and returns metrics."""
    config = NeuronConfig(n_neurons=100, n_inputs=16, n_outputs=8)
    layer = NeuronLayer(config)
    
    context = {
        'input': np.random.randn(16),
        'target': np.random.randn(8),
    }
    
    result = layer.optimize(context)
    
    assert 'loss' in result, "Result should contain 'loss'"
    assert 'output' in result, "Result should contain 'output'"
    assert isinstance(result['loss'], float), "Loss should be float"
    print(f"✓ Optimize returns loss={result['loss']:.4f}")


def test_self_optimization():
    """Test that self_optimize() doesn't crash and modifies weights."""
    config = NeuronConfig(n_neurons=50, n_inputs=8, n_outputs=4)
    layer = NeuronLayer(config)
    
    weights_before = layer.W_recurrent.copy()
    layer.self_optimize()
    weights_after = layer.W_recurrent.copy()
    
    # Weights should change slightly (due to decay)
    diff = np.sum(np.abs(weights_after - weights_before))
    assert diff > 0, "Self-optimization should modify weights"
    print(f"✓ Self-optimization modified weights (diff={diff:.6f})")


def test_holographic_snapshot():
    """Test that snapshot produces valid state vector."""
    config = NeuronConfig(n_neurons=100, n_inputs=16, n_outputs=8)
    layer = NeuronLayer(config)
    
    # Run a few steps to populate state
    for _ in range(5):
        layer.forward(np.random.randn(16))
    
    snapshot = layer.snapshot()
    
    assert 'level' in snapshot, "Snapshot should contain 'level'"
    assert 'state_vector' in snapshot, "Snapshot should contain 'state_vector'"
    assert isinstance(snapshot['state_vector'], np.ndarray), "State vector should be ndarray"
    assert len(snapshot['state_vector']) > 0, "State vector should not be empty"
    print(f"✓ Snapshot has state_vector of dim {len(snapshot['state_vector'])}")


def test_single_layer_in_stack():
    """Test NeuronLayer operating inside a FractalStack."""
    stack = FractalStack()
    config = NeuronConfig(n_neurons=50, n_inputs=8, n_outputs=4)
    stack.add_layer(NeuronLayer(config))
    
    result = stack.step({
        'input': np.random.randn(8),
        'target': np.random.randn(4),
    })
    
    assert 'step' in result, "Result should contain 'step'"
    assert 'layer_results' in result, "Result should contain 'layer_results'"
    assert 0 in result['layer_results'], "Should have Meta^0 results"
    print(f"✓ Single layer in stack: step={result['step']}")


if __name__ == '__main__':
    print("=" * 50)
    print("Test 1: Single-Layer Self-Optimization")
    print("=" * 50)
    
    test_neuron_layer_forward()
    test_neuron_layer_optimize()
    test_self_optimization()
    test_holographic_snapshot()
    test_single_layer_in_stack()
    
    print("\nAll tests passed! ✓")
