"""
Test 6: LLM-in-the-Loop

Tests the Meta^N LLM interface:
- Observation generation
- Alert detection
- Suggestion generation
- External callback integration
- Accept/reject tracking

Structure only — not for running on Pi.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from throng3.pipeline import MetaNPipeline
from throng3.layers.meta_n_llm import LLMInterface, LLMConfig, Observation
from throng3.core.signal import Signal, SignalDirection, SignalType


def test_llm_observation():
    """Test that LLM layer generates observations."""
    pipeline = MetaNPipeline.create_default(
        n_neurons=50, n_inputs=8, n_outputs=4, include_llm=True
    )
    
    # Run some steps
    for _ in range(15):
        pipeline.step(np.random.randn(8), reward=0.1)
    
    llm = pipeline.stack.get_layer(99)
    assert llm is not None, "LLM layer should exist at level 99"
    assert len(llm.observations) > 0, "Should have observations"
    
    latest = llm.observations[-1]
    assert isinstance(latest, Observation)
    assert latest.human_readable != ""
    print(f"✓ LLM observation: {latest.human_readable[:100]}...")


def test_llm_alerts():
    """Test alert detection."""
    pipeline = MetaNPipeline.create_default(
        n_neurons=50, n_inputs=8, n_outputs=4, include_llm=True
    )
    
    # Run many steps without improvement (should trigger stagnation alert)
    for _ in range(100):
        pipeline.step(np.random.randn(8), reward=0.0)
    
    llm = pipeline.stack.get_layer(99)
    report = llm.get_system_report()
    print(f"✓ System report:\n{report[:200]}...")


def test_llm_callback():
    """Test external LLM callback integration."""
    callback_calls = []
    
    def mock_llm(observation):
        callback_calls.append(observation)
        # Return a suggestion
        return {
            'suggestions': [{
                'target_level': 4,
                'payload': {'exploration_rate': 0.2},
            }],
        }
    
    pipeline = MetaNPipeline.create_default(
        n_neurons=50, n_inputs=8, n_outputs=4,
        include_llm=True, llm_callback=mock_llm
    )
    
    for _ in range(20):
        pipeline.step(np.random.randn(8), reward=0.1)
    
    assert len(callback_calls) > 0, "Callback should have been called"
    print(f"✓ LLM callback called {len(callback_calls)} times")


def test_llm_suggestion_tracking():
    """Test that suggestions are tracked with accept/reject counts."""
    pipeline = MetaNPipeline.create_default(
        n_neurons=50, n_inputs=8, n_outputs=4, include_llm=True
    )
    
    llm = pipeline.stack.get_layer(99)
    
    # Submit manual suggestions
    llm.submit_suggestion(0, {'threshold': 0.8})
    llm.submit_suggestion(1, {'active_rule': 'hebbian'})
    
    # Run step to route signals
    pipeline.step(np.random.randn(8))
    
    print(f"✓ Suggestions: sent={len(llm._suggestions_sent)}, "
          f"accepted={llm._suggestions_accepted}, "
          f"rejected={llm._suggestions_rejected}")


def test_llm_system_report_api():
    """Test the public API for system reports."""
    pipeline = MetaNPipeline.create_default(
        n_neurons=50, n_inputs=8, n_outputs=4, include_llm=True
    )
    
    for _ in range(10):
        pipeline.step(np.random.randn(8), reward=0.1)
    
    llm = pipeline.stack.get_layer(99)
    
    report = llm.get_system_report()
    history = llm.get_observation_history(5)
    
    assert isinstance(report, str)
    assert isinstance(history, list)
    assert len(history) <= 5
    print(f"✓ API: report length={len(report)}, history entries={len(history)}")


if __name__ == '__main__':
    print("=" * 50)
    print("Test 6: LLM-in-the-Loop")
    print("=" * 50)
    
    test_llm_observation()
    test_llm_alerts()
    test_llm_callback()
    test_llm_suggestion_tracking()
    test_llm_system_report_api()
    
    print("\nAll tests passed! ✓")
