"""
Test 2: Layer Self-Optimization (Meta^1)

Tests that Meta^1 (SynapseOptimizer) can modify Meta^0's weights
using STDP, Hebbian learning, and pruning.

Structure only — not for running on Pi.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.core.fractal_stack import FractalStack
from throng3.learning.stdp import STDPRule
from throng3.learning.hebbian import HebbianRule
from throng3.learning.pruning import NashPruner


def test_stdp_learning():
    """Test STDP weight updates."""
    stdp = STDPRule()
    
    N = 50
    weights = np.random.randn(N, N) * 0.1
    pre_spikes = (np.random.random(N) > 0.8).astype(float)
    post_spikes = (np.random.random(N) > 0.8).astype(float)
    
    dW = stdp.batch_update(weights, pre_spikes, post_spikes)
    
    assert dW.shape == weights.shape, "dW shape should match weights"
    assert np.any(dW != 0), "STDP should produce non-zero updates"
    print(f"✓ STDP produced updates: mean |dW|={np.mean(np.abs(dW)):.6f}")


def test_hebbian_learning():
    """Test Hebbian weight updates."""
    hebbian = HebbianRule()
    
    N = 50
    weights = np.random.randn(N, N) * 0.1
    pre = np.random.rand(N)
    post = np.random.rand(N)
    
    dW = hebbian.batch_update(weights, pre, post)
    
    assert dW.shape == weights.shape
    assert np.any(dW != 0), "Hebbian should produce non-zero updates"
    print(f"✓ Hebbian produced updates: mean |dW|={np.mean(np.abs(dW)):.6f}")


def test_pruning():
    """Test Nash pruning."""
    pruner = NashPruner()
    
    N = 50
    weights = np.random.randn(N, N) * 0.1
    activities = np.random.rand(N)
    
    pruned, stats = pruner.prune(weights, activities)
    
    assert pruned.shape == weights.shape
    assert stats['n_pruned'] >= 0
    assert 0 <= stats['sparsity'] <= 1
    print(f"✓ Pruning: removed {stats['n_pruned']}, sparsity={stats['sparsity']:.3f}")


def test_synapse_optimizer_in_stack():
    """Test Meta^1 operating with Meta^0 in a stack."""
    stack = FractalStack()
    
    neuron_config = NeuronConfig(n_neurons=100, n_inputs=16, n_outputs=8)
    stack.add_layer(NeuronLayer(neuron_config))
    stack.add_layer(SynapseOptimizer(SynapseConfig()))
    
    # Run multiple steps
    for i in range(10):
        result = stack.step({
            'input': np.random.randn(16),
            'target': np.random.randn(8),
            'reward': np.random.randn() * 0.1,
        })
    
    assert 1 in result['layer_results'], "Should have Meta^1 results"
    meta1_result = result['layer_results'][1]
    assert 'active_rule' in meta1_result, "Should report active rule"
    print(f"✓ Meta^1 in stack: active_rule={meta1_result['active_rule']}")


def test_dopamine_modulation():
    """Test that dopamine modulates learning rates."""
    synapse = SynapseOptimizer(SynapseConfig())
    
    # Simulate positive reward
    rpe = synapse.dopamine.compute_rpe(1.0)
    modulated_lr = synapse.dopamine.modulate_learning_rate(0.01)
    
    assert modulated_lr > 0.01, "Positive reward should increase learning rate"
    print(f"✓ Positive reward: base_lr=0.01, modulated={modulated_lr:.4f}")
    
    # Simulate negative reward
    rpe = synapse.dopamine.compute_rpe(-1.0)
    modulated_lr = synapse.dopamine.modulate_learning_rate(0.01)
    
    assert modulated_lr < 0.01, "Negative reward should decrease learning rate"
    print(f"✓ Negative reward: base_lr=0.01, modulated={modulated_lr:.4f}")


if __name__ == '__main__':
    print("=" * 50)
    print("Test 2: Synapse Optimization (Meta^1)")
    print("=" * 50)
    
    test_stdp_learning()
    test_hebbian_learning()
    test_pruning()
    test_synapse_optimizer_in_stack()
    test_dopamine_modulation()
    
    print("\nAll tests passed! ✓")
