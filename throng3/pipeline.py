"""
Integration Pipeline — Wire Everything Together

Provides a high-level API to create and run a Meta^N fractal stack.
Handles the boilerplate of connecting layers, routing context,
and managing the optimization loop.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Callable
import time
import logging

from throng3.core.fractal_stack import FractalStack
from throng3.core.signal import Signal, SignalDirection, SignalType
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig
from throng3.layers.meta3_representation import RepresentationOptimizer, RepresentationConfig
from throng3.layers.meta4_goal import GoalHierarchy, GoalConfig
from throng3.layers.meta5_architecture import ArchitectureSearch, ArchitectureConfig
from throng3.layers.meta_n_llm import LLMInterface, LLMConfig

logger = logging.getLogger(__name__)


class MetaNPipeline:
    """
    High-level pipeline for running Meta^N recursive optimization.
    
    Usage:
        pipeline = MetaNPipeline.create_default(n_neurons=1000)
        
        for step in range(1000):
            result = pipeline.step(input_data, target, reward)
            print(result['system_state'])
    """
    
    def __init__(self, stack: FractalStack):
        self.stack = stack
        self._step_count = 0
        self._history: List[Dict] = []
    
    @classmethod
    def create_default(cls, 
                       n_neurons: int = 1000,
                       n_inputs: int = 64,
                       n_outputs: int = 32,
                       include_llm: bool = False,
                       llm_callback: Optional[Callable] = None) -> 'MetaNPipeline':
        """
        Create a default Meta^N pipeline with all layers.
        
        Args:
            n_neurons: Number of neurons in Meta^0
            n_inputs: Input dimension
            n_outputs: Output dimension
            include_llm: Whether to include Meta^N LLM interface
            llm_callback: Optional callback for LLM integration
        """
        stack = FractalStack(config={'holographic_dim': 128})
        
        # Meta^0: Neural substrate
        stack.add_layer(NeuronLayer(NeuronConfig(
            n_neurons=n_neurons,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
        )))
        
        # Meta^1: Synapse optimization
        stack.add_layer(SynapseOptimizer(SynapseConfig()))
        
        # Meta^2: Learning rule selection
        stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))
        
        # Meta^3: Representation optimization
        stack.add_layer(RepresentationOptimizer(RepresentationConfig()))
        
        # Meta^4: Goal hierarchy
        stack.add_layer(GoalHierarchy(GoalConfig()))
        
        # Meta^5: Architecture search
        stack.add_layer(ArchitectureSearch(ArchitectureConfig()))
        
        # Meta^N: LLM interface (optional)
        if include_llm:
            stack.add_layer(LLMInterface(LLMConfig(
                callback=llm_callback,
            )))
        
        return cls(stack)
    
    @classmethod
    def create_minimal(cls,
                       n_neurons: int = 100,
                       n_inputs: int = 16,
                       n_outputs: int = 8) -> 'MetaNPipeline':
        """
        Create a minimal pipeline with just Meta^0-2.
        
        Good for testing and understanding the core dynamics.
        """
        stack = FractalStack(config={'holographic_dim': 64})
        
        stack.add_layer(NeuronLayer(NeuronConfig(
            n_neurons=n_neurons,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
        )))
        stack.add_layer(SynapseOptimizer(SynapseConfig()))
        stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))
        
        return cls(stack)
    
    def step(self, input_data: np.ndarray, 
             target: Optional[np.ndarray] = None,
             reward: float = 0.0) -> Dict[str, Any]:
        """
        Run one full pipeline step.
        
        Args:
            input_data: Input to the neural network
            target: Optional target output (for supervised learning)
            reward: External reward signal
            
        Returns:
            Dict with output, metrics, and system state
        """
        # Build context for all layers
        neuron_layer = self.stack.get_layer(0)
        
        # Run Meta^0 forward pass directly to get weights/activations
        # that other layers need
        if neuron_layer:
            output = neuron_layer.forward(input_data)
            weights = neuron_layer.get_weights()
            activations = neuron_layer.activations.copy()
            spikes = neuron_layer.spikes.copy()
        else:
            output = np.zeros(1)
            weights = {}
            activations = np.array([])
            spikes = np.array([])
        
        # Build context enriched with neuron state
        context = {
            'input': input_data,
            'target': target,
            'reward': reward,
            'output': output,
            'weights': weights,
            'activations': activations,
            'spikes': spikes,
            'outputs': output,
            'step': self._step_count,
        }
        
        # If we have Meta^1 results from last step, include them
        if self._history:
            last = self._history[-1]
            if 'layer_results' in last:
                meta1_result = last['layer_results'].get(1, {})
                context['meta1_performance'] = meta1_result
        
        # Run the full stack step
        result = self.stack.step(context)
        
        # Extract output from Meta^0
        meta0_result = result.get('layer_results', {}).get(0, {})
        final_output = meta0_result.get('output', output)
        
        # Build summary
        step_result = {
            'output': final_output,
            'loss': meta0_result.get('loss', float('inf')),
            'reward': reward,
            'system_state': self.stack.get_system_state(),
            'holographic': result.get('holographic', {}),
            'step': self._step_count,
            'layer_results': result.get('layer_results', {}),
        }
        
        self._history.append(step_result)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        
        self._step_count += 1
        return step_result
    
    def reset_task_state(self):
        """
        Reset task-specific state while preserving meta-learned knowledge.
        
        This enables transfer learning:
        - Meta^0 weights are reset (task-specific)
        - Meta^1 learning rule parameters persist (meta-knowledge)
        - Meta^2 bandit statistics persist (meta-knowledge)
        - Meta^3-5 hyperparameters persist (meta-knowledge)
        """
        # Reset Meta^0 neural state
        neuron_layer = self.stack.get_layer(0)
        if neuron_layer:
            # Reinitialize weights
            neuron_layer._init_weights()
            # Reset dynamic state
            neuron_layer.membrane_potential[:] = 0
            neuron_layer.activations[:] = 0
            neuron_layer.spikes[:] = False
            neuron_layer.refractory_timer[:] = 0
            neuron_layer._spike_history.clear()
            neuron_layer._activity_history.clear()
            neuron_layer._time = 0.0
        
        # Reset Meta^1 pending changes but keep learned parameters
        synapse_layer = self.stack.get_layer(1)
        if synapse_layer:
            synapse_layer._pending_dW = {}
            # Reset spike history for STDP
            synapse_layer._prev_spikes = None
            synapse_layer._prev_activations = None
            # Keep dopamine system state (it's meta-knowledge about reward dynamics)
            # Keep STDP/Hebbian parameters (tuned by Meta^2)
        
        # Meta^2 and higher: keep everything (this IS the meta-knowledge)
        # Their bandit statistics, hyperparameters, etc. are what transfer
        
        # Reset step counter for new task
        self._step_count = 0
        # Keep history for meta-analysis but mark boundary
        if self._history:
            self._history.append({'task_boundary': True})

    
    def get_report(self) -> str:
        """Get a human-readable system report."""
        state = self.stack.get_system_state()
        
        lines = [
            "=" * 60,
            f"  Meta^N Pipeline Report — Step {self._step_count}",
            "=" * 60,
            f"  Active layers: {state['n_layers']}",
            f"  Signals routed: {state['total_signals_routed']}",
            f"  Avg step time: {state['avg_step_time']*1000:.1f}ms",
            "",
        ]
        
        for level, layer_state in sorted(state['layers'].items()):
            metrics = layer_state.get('metrics', {})
            lines.append(
                f"  Meta^{level} ({layer_state['name']}): "
                f"loss={metrics.get('loss', '?'):.4f}  "
                f"acc={metrics.get('accuracy', '?'):.4f}  "
                f"stability={metrics.get('stability', '?'):.3f}"
            )
        
        holo = state.get('holographic', {})
        lines.extend([
            "",
            f"  Holographic coherence: {holo.get('coherence', '?')}",
            f"  Signal stats: {state.get('signal_stats', {})}",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def save_state(self) -> Dict:
        """Save complete pipeline state."""
        return {
            'stack': self.stack.save_state(),
            'step_count': self._step_count,
        }
    
    def __repr__(self) -> str:
        return f"MetaNPipeline(step={self._step_count}, {self.stack})"
