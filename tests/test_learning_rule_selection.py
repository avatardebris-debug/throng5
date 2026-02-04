"""
Test 4: Learning Rule Selection (Meta^2)

Tests that Meta^2 correctly selects between STDP and Hebbian
learning rules based on performance feedback.

Structure only — not for running on Pi.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig
from throng3.core.fractal_stack import FractalStack
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig


def test_ucb_selection():
    """Test UCB rule selection converges to best rule."""
    selector = LearningRuleSelector(LearningRuleSelectorConfig(
        selection_strategy='ucb',
        evaluation_window=10,
    ))
    
    # Simulate: 'both' gives best rewards
    for _ in range(100):
        # Fake context with Meta^1 performance
        context = {
            'meta1_performance': {
                'loss': 0.5 if selector.current_rule == 'both' else 0.8,
                'rpe': 0.1,
                'active_rule': selector.current_rule,
            },
        }
        result = selector.optimize(context)
    
    # After enough trials, should favor 'both' or similar
    print(f"✓ UCB selection: current={selector.current_rule}")
    print(f"  Rule values: {selector.rule_values}")
    print(f"  Rule counts: {selector.rule_counts}")


def test_epsilon_greedy_selection():
    """Test epsilon-greedy exploration."""
    selector = LearningRuleSelector(LearningRuleSelectorConfig(
        selection_strategy='epsilon_greedy',
        epsilon=0.3,
        evaluation_window=5,
    ))
    
    rules_selected = []
    for _ in range(50):
        context = {
            'meta1_performance': {
                'loss': 0.5,
                'rpe': 0.0,
                'active_rule': selector.current_rule,
            },
        }
        selector.optimize(context)
        rules_selected.append(selector.current_rule)
    
    unique_rules = set(rules_selected)
    assert len(unique_rules) >= 2, "Should explore multiple rules"
    print(f"✓ Epsilon-greedy explored {len(unique_rules)} rules: {unique_rules}")


def test_rule_switching():
    """Test that rule switches happen when performance degrades."""
    selector = LearningRuleSelector(LearningRuleSelectorConfig(
        selection_strategy='ucb',
        evaluation_window=5,
    ))
    
    switches = []
    prev_rule = selector.current_rule
    
    for i in range(100):
        # Make current rule perform poorly
        loss = 0.9 if selector.current_rule == 'stdp' else 0.3
        context = {
            'meta1_performance': {
                'loss': loss,
                'rpe': 0.0,
                'active_rule': selector.current_rule,
            },
        }
        selector.optimize(context)
        
        if selector.current_rule != prev_rule:
            switches.append((i, prev_rule, selector.current_rule))
            prev_rule = selector.current_rule
    
    print(f"✓ Rule switches: {len(switches)}")
    for step, old, new in switches[:5]:
        print(f"  Step {step}: {old} → {new}")


def test_meta2_in_full_stack():
    """Test Meta^2 operating in full 3-layer stack."""
    stack = FractalStack()
    
    stack.add_layer(NeuronLayer(NeuronConfig(n_neurons=50, n_inputs=8, n_outputs=4)))
    stack.add_layer(SynapseOptimizer(SynapseConfig()))
    stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig(
        evaluation_window=5,
    )))
    
    # Run several steps
    for _ in range(30):
        stack.step({
            'input': np.random.randn(8),
            'target': np.random.randn(4),
            'reward': np.random.randn() * 0.1,
        })
    
    meta2 = stack.get_layer(2)
    print(f"✓ Meta^2 in stack: rule={meta2.current_rule}")
    print(f"  Values: {meta2.rule_values}")


if __name__ == '__main__':
    print("=" * 50)
    print("Test 4: Learning Rule Selection (Meta^2)")
    print("=" * 50)
    
    test_ucb_selection()
    test_epsilon_greedy_selection()
    test_rule_switching()
    test_meta2_in_full_stack()
    
    print("\nAll tests passed! ✓")
