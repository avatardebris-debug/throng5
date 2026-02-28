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
from throng3.core.global_dynamics import GlobalDynamicsOptimizer, GlobalConfig
from throng3.layers.meta0_neuron import NeuronLayer, NeuronConfig
from throng3.layers.meta1_synapse import SynapseOptimizer, SynapseConfig
from throng3.layers.meta2_learning_rule import LearningRuleSelector, LearningRuleSelectorConfig
from throng3.layers.meta3_representation import RepresentationOptimizer, RepresentationConfig
from throng3.layers.meta3_consolidation import WeightConsolidation, ConsolidationConfig
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
    
    def __init__(self, stack: FractalStack, 
                 global_optimizer: Optional[GlobalDynamicsOptimizer] = None):
        self.stack = stack
        self.global_optimizer = global_optimizer
        self._step_count = 0
        self._history: List[Dict] = []
        self._experience_buffer: List[Dict] = []  # For MAML meta-updates
    
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
    def create_with_ewc(cls,
                        n_neurons: int = 100,
                        n_inputs: int = 16,
                        n_outputs: int = 8,
                        ewc_lambda: float = 1000.0) -> 'MetaNPipeline':
        """
        Create pipeline with EWC (Elastic Weight Consolidation) for compound transfer.
        
        This uses Meta^3 consolidation to prevent catastrophic interference.
        """
        stack = FractalStack(config={'holographic_dim': 64})
        
        stack.add_layer(NeuronLayer(NeuronConfig(
            n_neurons=n_neurons,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
        )))
        stack.add_layer(SynapseOptimizer(SynapseConfig()))
        stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))
        stack.add_layer(WeightConsolidation(ConsolidationConfig(
            ewc_lambda=ewc_lambda,
        )))
        
        return cls(stack)
    
    @classmethod
    def create_with_maml(cls,
                        n_neurons: int = 100,
                        n_inputs: int = 16,
                        n_outputs: int = 8,
                        meta_lr: float = 0.001) -> 'MetaNPipeline':
        """
        Create pipeline with task-conditioned MAML for meta-learning.
        
        This uses Meta^3 MAML to learn task-type-specific optimization strategies.
        """
        from throng3.layers.meta3_maml import TaskConditionedMAML
        from throng3.config.maml_config import MAMLConfig
        
        stack = FractalStack(config={'holographic_dim': 64})
        
        stack.add_layer(NeuronLayer(NeuronConfig(
            n_neurons=n_neurons,
            n_inputs=n_inputs,
            n_outputs=n_outputs,
        )))
        stack.add_layer(SynapseOptimizer(SynapseConfig()))
        stack.add_layer(LearningRuleSelector(LearningRuleSelectorConfig()))
        stack.add_layer(TaskConditionedMAML(MAMLConfig(
            meta_lr=meta_lr,
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
    
    @classmethod
    def create_adaptive(cls,
                        n_neurons: int = 1000,
                        n_inputs: int = 64,
                        n_outputs: int = 32,
                        global_config: Optional[GlobalConfig] = None,
                        include_llm: bool = False,
                        llm_callback: Optional[Callable] = None) -> 'MetaNPipeline':
        """
        Create an adaptive Meta^N pipeline with GlobalDynamicsOptimizer.
        
        The global optimizer automatically gates layers based on task complexity:
        - Simple tasks use fewer layers (faster, less interference)
        - Complex tasks recruit higher layers when needed
        - Layers that hurt performance get gated down
        
        Args:
            n_neurons: Number of neurons in Meta^0
            n_inputs: Input dimension
            n_outputs: Output dimension
            global_config: Configuration for the GlobalDynamicsOptimizer
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
        
        # Wrap with GlobalDynamicsOptimizer
        global_optimizer = GlobalDynamicsOptimizer(
            stack, 
            config=global_config or GlobalConfig()
        )
        
        return cls(stack, global_optimizer=global_optimizer)
    
    def step(self, input_data: np.ndarray, 
             target: Optional[np.ndarray] = None,
             reward: float = 0.0,
             episode_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Run one full pipeline step.
        
        Args:
            input_data: Input to the neural network
            target: Optional target output (for supervised learning)
            reward: External reward signal
            episode_return: Optional average episode return (for RL loss calculation)
            
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
            'episode_return': episode_return,  # Add episode return to context
            'output': output,
            'weights': weights,
            'activations': activations,
            'spikes': spikes,
            'outputs': output,
            'step': self._step_count,
            'n_outputs': neuron_layer.n_outputs if neuron_layer else 4,  # For Q-learner
        }
        
        # If we have Meta^1 results from last step, include them
        if self._history:
            last = self._history[-1]
            if 'layer_results' in last:
                meta1_result = last['layer_results'].get(1, {})
                context['meta1_performance'] = meta1_result
        
        # Run the full stack step (through global optimizer if present)
        if self.global_optimizer:
            result = self.global_optimizer.step(context)
            global_metrics = result.get('global', {})
        else:
            result = self.stack.step(context)
            global_metrics = {}
        
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
            'global': global_metrics,  # Include global optimizer metrics
        }
        
        self._history.append(step_result)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        
        # Collect experience for MAML meta-updates
        self._experience_buffer.append({
            'state': input_data.copy(),
            'output': final_output.copy(),
            'reward': reward,
            'weights': {k: v.copy() for k, v in weights.items()},
        })
        
        self._step_count += 1
        return step_result
    
    def step_rl(self, state: np.ndarray, 
                reward: float = 0.0,
                done: bool = False,
                prev_action: Optional[int] = None) -> Dict[str, Any]:
        """
        Single-step RL API: get Q-values AND learn from previous transition.
        
        Unlike step() which requires two calls per env step, step_rl() handles
        both inference and learning in one call:
        - Forward pass through Meta^0 to get Q-values
        - Pass (prev_action, reward, done) to Meta^1 Q-learner for TD update
        
        Args:
            state: Current environment observation
            reward: Reward received from the previous action
            done: Whether the episode is over
            prev_action: The action taken in the previous step (for Q-learning)
            
        Returns:
            Dict with output (Q-values), metrics, and system state
        """
        neuron_layer = self.stack.get_layer(0)
        
        if neuron_layer:
            output = neuron_layer.forward(state)
            weights = neuron_layer.get_weights()
            activations = neuron_layer.activations.copy()
            spikes = neuron_layer.spikes.copy()
        else:
            output = np.zeros(1)
            weights = {}
            activations = np.array([])
            spikes = np.array([])
        
        # Build context with RL-specific fields
        context = {
            'input': state,
            'raw_observation': state,  # Q-learner uses this for state
            'target': None,
            'reward': reward,
            'output': output,
            'weights': weights,
            'activations': activations,
            'spikes': spikes,
            'outputs': output,
            'step': self._step_count,
            'n_outputs': neuron_layer.n_outputs if neuron_layer else 4,
            # RL-specific: pass action and done to Q-learner
            'action': prev_action if prev_action is not None else 0,
            'done': done,
        }
        
        # Include previous meta1 results
        if self._history:
            last = self._history[-1]
            if 'layer_results' in last:
                meta1_result = last['layer_results'].get(1, {})
                context['meta1_performance'] = meta1_result
        
        # Run the full stack 
        if self.global_optimizer:
            result = self.global_optimizer.step(context)
            global_metrics = result.get('global', {})
        else:
            result = self.stack.step(context)
            global_metrics = {}
        
        # Extract output
        meta0_result = result.get('layer_results', {}).get(0, {})
        final_output = meta0_result.get('output', output)
        
        # Use Q-values from Meta^1 if available (better than raw neuron output)
        meta1_result = result.get('layer_results', {}).get(1, {})
        q_values = meta1_result.get('q_values', None)
        if q_values is not None:
            final_output = q_values  # Use Q-table values for action selection
        
        step_result = {
            'output': final_output,
            'loss': meta0_result.get('loss', float('inf')),
            'reward': reward,
            'q_values': q_values,
            'td_error': meta1_result.get('td_error', 0.0),
            'system_state': self.stack.get_system_state(),
            'step': self._step_count,
            'layer_results': result.get('layer_results', {}),
            'global': global_metrics,
        }
        
        self._history.append(step_result)
        if len(self._history) > 100:
            self._history = self._history[-50:]
        
        # Collect experience for MAML meta-updates
        self._experience_buffer.append({
            'state': state.copy(),
            'output': final_output.copy(),
            'reward': reward,
            'weights': {k: v.copy() for k, v in weights.items()},
        })
        
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
        
        # Clear experience buffer for new task
        self._experience_buffer = []

    
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
    
    def consolidate_task(self):
        """
        Consolidate the current task's knowledge using EWC.
        
        Call this after a task completes to:
        1. Compute Fisher information (weight importance)
        2. Store optimal weights
        3. Protect important weights from future changes
        """
        meta0 = self.stack.get_layer(0)
        if not meta0:
            return
        
        weights = meta0.get_weights()
        
        meta1 = self.stack.get_layer(1)
        if not meta1 or not hasattr(meta1, 'gradient'):
            return
        
        fisher = meta1.gradient.get_fisher_information()
        
        if not fisher:
            return
        
        meta3 = self.stack.get_layer(3)
        if meta3 and hasattr(meta3, 'consolidate_task'):
            meta3.consolidate_task(weights, fisher)
    
    def consolidate_maml_task(self, gamma: float = 0.95):
        """
        Consolidate MAML meta-learning from collected RL experience.
        
        Converts experience buffer to support/query sets using TD targets
        and triggers MAML meta-update.
        
        Args:
            gamma: Discount factor for TD targets
        """
        if not self._experience_buffer:
            return
        
        maml_layer = self.stack.get_layer(3)
        if not maml_layer or not hasattr(maml_layer, 'meta_update'):
            return
        
        # Convert RL experience to MAML-compatible (state, target) pairs
        # Target = reward + gamma * max(Q(next_state))
        support_set = []
        query_set = []
        
        # Get task type from MAML layer
        task_type = getattr(maml_layer, '_detected_task_type', 'rl')
        
        # Use first experience's weights as initial params
        if not self._experience_buffer:
            return
        raw_weights = self._experience_buffer[0]['weights']
        
        # Create combined weight matrix: W_combined = W_out @ W_in
        # This maps state (n_inputs) → output (n_outputs) directly
        # so MAML's _forward can do W_combined @ state
        if 'W_out' in raw_weights and 'W_in' in raw_weights:
            W_combined = raw_weights['W_out'] @ raw_weights['W_in']
            weights = {'W_out': W_combined}
        else:
            # Fallback if weights don't have expected structure
            return
        
        # Convert each transition to (state, TD-target)
        for i, exp in enumerate(self._experience_buffer[:-1]):
            state = exp['state']
            output = exp['output']  # Q-values
            reward = exp['reward']
            next_exp = self._experience_buffer[i + 1]
            next_output = next_exp['output']
            
            # TD target: r + gamma * max(Q(s'))
            # For each action, target is current Q-value except for the chosen action
            # which gets the TD update
            action = np.argmax(output)  # Greedy action
            td_target = output.copy()
            td_target[action] = reward + gamma * np.max(next_output)
            
            # Split 60/40 into support/query
            if i < len(self._experience_buffer) * 0.6:
                support_set.append((state, td_target))
            else:
                query_set.append((state, td_target))
        
        # Create task dict for MAML
        task = {
            'task_type': task_type,
            'support_set': support_set,
            'query_set': query_set,
            'weights': weights,
        }
        
        # Trigger meta-update
        if support_set and query_set:
            maml_layer.meta_update([task])
        
        # Clear buffer
        self._experience_buffer = []

    
    def reset_task_state(self):
        """Reset task-specific state (for task boundaries)."""
        meta0 = self.stack.get_layer(0)
        if meta0:
            meta0.membrane_potential[:] = 0
            meta0.activations[:] = 0

