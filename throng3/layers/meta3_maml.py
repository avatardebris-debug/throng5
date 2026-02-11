"""
Task-Conditioned MAML (Model-Agnostic Meta-Learning)

Meta^3 layer that learns task-type-specific optimization strategies.

Key insight from Fisher boosting experiments:
- Supervised tasks need to boost shared features
- RL tasks need to preserve policy stability
- MAML should learn these strategies automatically
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import Signal, SignalType
from throng3.core.task_detector import TaskDetector
from throng3.config.maml_config import MAMLConfig


class TaskConditionedMAML(MetaLayer):
    """
    Task-conditioned MAML: Learn different optimization strategies per task type.
    
    Architecture:
        Meta^2 classifies task type → Meta^3 selects learned params → Meta^1 applies
    
    Learned parameters per task type:
        - Learning rate multipliers (per weight)
        - Adaptation steps (inner loop)
        - Meta-initialization (starting weights)
    """
    
    def __init__(self, config: Optional[MAMLConfig] = None):
        """
        Initialize task-conditioned MAML.
        
        Args:
            config: MAML configuration
        """
        maml_config = config or MAMLConfig()
        
        # Pass dict to parent (MetaLayer expects a dict with .get())
        config_dict = {
            'meta_lr': maml_config.meta_lr,
            'inner_steps': maml_config.inner_steps,
            'inner_lr': maml_config.inner_lr,
            'meta_batch_size': maml_config.meta_batch_size,
            'first_order': maml_config.first_order,
            'use_task_conditioning': maml_config.use_task_conditioning,
        }
        super().__init__(level=3, name="TaskConditionedMAML", config=config_dict)
        
        # Store typed config separately (like consolidation does)
        self.maml_config = maml_config
        
        # Learned meta-parameters per task type
        # These are the "initializations" that MAML learns
        self.meta_params = {
            'supervised': {
                'lr_multipliers': {},  # Will be initialized on first use
                'inner_lr': self.maml_config.inner_lr,
                'inner_steps': self.maml_config.inner_steps,
            },
            'rl': {
                'lr_multipliers': {},
                'inner_lr': self.maml_config.inner_lr,
                'inner_steps': self.maml_config.inner_steps,
            },
        }
        
        # Meta-optimizer state (Adam)
        self.meta_optimizer_state = {
            'supervised': {'m': {}, 'v': {}, 't': 0},
            'rl': {'m': {}, 'v': {}, 't': 0},
        }
        
        # Task history for meta-learning
        self.task_history = []
        self.meta_batch = []
        
        # Task conditioning: detected task type from Meta^2 signals
        self._detected_task_type = 'supervised'  # Default
        self._task_confidence = 0.0
        
        # Fallback task detector (for first steps before signals arrive)
        self._fallback_detector = TaskDetector(window=50)
        
        # Stats
        self.meta_updates = 0
        self.tasks_seen = 0
    
    def _handle_performance_update(self, signal: Signal):
        """
        Handle performance signals from Meta^2 — extract task type.
        
        Meta^2 sends: {task_type, task_confidence, target_freq, reward_freq}
        """
        payload = signal.payload or {}
        if 'task_type' in payload:
            raw_type = payload['task_type']
            confidence = payload.get('task_confidence', 0.5)
            
            # Map TaskDetector types to MAML types
            if raw_type == 'rl':
                self._detected_task_type = 'rl'
            else:
                # 'supervised', 'hybrid', 'unknown' all map to supervised
                self._detected_task_type = 'supervised'
            
            self._task_confidence = confidence
    
    def _resolve_task_type(self, context: Dict[str, Any]) -> str:
        """
        Determine the current task type using the best available source.
        
        Priority:
            1. Explicit context override (for direct API use / tests)
            2. Signal from Meta^2 (if confident enough)
            3. Fallback TaskDetector (for first steps)
            4. Default: 'supervised'
        """
        # 1. Explicit override in context
        explicit = context.get('task_type')
        if explicit and explicit in self.meta_params:
            return explicit
        
        # 2. Signal from Meta^2 (updated via _handle_performance_update)
        if self._task_confidence > 0.5:
            return self._detected_task_type
        
        # 3. Fallback detector
        self._fallback_detector.update(context)
        chars = self._fallback_detector.get_characteristics()
        if chars and chars.confidence > 0.5:
            if chars.signal_type == 'rl':
                return 'rl'
            return 'supervised'
        
        # 4. Default
        return 'supervised'
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply task-conditioned MAML.
        
        Returns learned learning rate multipliers for current task type.
        Uses task type from Meta^2 signals, fallback detector, or context.
        
        Args:
            context: Pipeline context
            
        Returns:
            Dict with lr_multipliers and other MAML params
        """
        self.process_inbox()
        
        # Resolve task type from best available source
        task_type = self._resolve_task_type(context)
        
        # Ensure task type is valid
        if task_type not in self.meta_params:
            task_type = 'supervised'  # Default fallback
        
        # Get learned parameters for this task type
        params = self.meta_params[task_type]
        
        # Initialize lr_multipliers if needed
        weights = context.get('weights', {})
        if not params['lr_multipliers'] and weights:
            # Initialize to uniform (1.0 everywhere)
            for name, W in weights.items():
                params['lr_multipliers'][name] = np.ones_like(W)
        
        return {
            'lr_multipliers': params['lr_multipliers'],
            'inner_lr': params['inner_lr'],
            'inner_steps': params['inner_steps'],
            'task_type': task_type,
            'meta_updates': self.meta_updates,
        }
    
    def inner_loop(self, 
                   initial_params: Dict[str, np.ndarray],
                   support_set: List[Tuple[np.ndarray, np.ndarray]],
                   inner_lr: float,
                   inner_steps: int) -> Dict[str, np.ndarray]:
        """
        Inner loop: Adapt parameters to support set.
        
        This is the "fast adaptation" part of MAML.
        
        Args:
            initial_params: Starting parameters (meta-initialization)
            support_set: List of (input, target) pairs for adaptation
            inner_lr: Learning rate for inner loop
            inner_steps: Number of gradient steps
            
        Returns:
            Adapted parameters
        """
        params = {k: v.copy() for k, v in initial_params.items()}
        
        for step in range(inner_steps):
            # Compute gradient on support set
            total_grad = {k: np.zeros_like(v) for k, v in params.items()}
            
            for x, y in support_set:
                # Forward pass
                output = self._forward(params, x)
                
                # Compute loss gradient
                grad = self._compute_gradient(params, x, y, output)
                
                # Accumulate
                for k in total_grad:
                    if k in grad:
                        total_grad[k] += grad[k]
            
            # Average gradient
            for k in total_grad:
                total_grad[k] /= len(support_set)
            
            # Gradient descent step
            for k in params:
                if k in total_grad:
                    params[k] -= inner_lr * total_grad[k]
        
        return params
    
    def meta_update(self, task_batch: List[Dict[str, Any]]):
        """
        Meta-learning update (outer loop).
        
        This updates the meta-parameters (initializations) based on
        performance on query sets after adaptation.
        
        Args:
            task_batch: List of task dicts with:
                - task_type: 'supervised' or 'rl'
                - support_set: Training data for adaptation
                - query_set: Test data for meta-gradient
                - weights: Current network weights
        """
        if not task_batch:
            return
        
        # Group tasks by type
        tasks_by_type = {'supervised': [], 'rl': []}
        for task in task_batch:
            task_type = task.get('task_type', 'supervised')
            if task_type in tasks_by_type:
                tasks_by_type[task_type].append(task)
        
        # Meta-update for each task type
        for task_type, tasks in tasks_by_type.items():
            if not tasks:
                continue
            
            # Compute meta-gradient
            meta_grad = self._compute_meta_gradient(task_type, tasks)
            
            # Apply meta-gradient with Adam
            self._apply_meta_gradient(task_type, meta_grad)
            
            self.meta_updates += 1
    
    def _compute_meta_gradient(self, 
                               task_type: str,
                               tasks: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Compute meta-gradient for a task type.
        
        Meta-gradient = gradient of query loss w.r.t. meta-parameters
        
        Args:
            task_type: 'supervised' or 'rl'
            tasks: List of tasks of this type
            
        Returns:
            Meta-gradient dict
        """
        meta_params = self.meta_params[task_type]
        meta_grad = {}
        
        # Get weight shapes from first task
        if tasks and 'weights' in tasks[0]:
            task_weights = tasks[0]['weights']
            # Reinitialize lr_multipliers if shapes don't match
            for name, W in task_weights.items():
                if name not in meta_params['lr_multipliers'] or \
                   meta_params['lr_multipliers'][name].shape != W.shape:
                    meta_params['lr_multipliers'][name] = np.ones_like(W)
        
        # Initialize meta-gradient
        for name in meta_params['lr_multipliers']:
            meta_grad[name] = np.zeros_like(meta_params['lr_multipliers'][name])
        
        # Accumulate meta-gradient across tasks
        for task in tasks:
            support_set = task.get('support_set', [])
            query_set = task.get('query_set', [])
            weights = task.get('weights', {})
            
            if not support_set or not query_set or not weights:
                continue
            
            # Inner loop: adapt to support set
            adapted_params = self.inner_loop(
                weights,
                support_set,
                meta_params['inner_lr'],
                meta_params['inner_steps']
            )
            
            # Compute query loss gradient
            query_grad = {}
            for x, y in query_set:
                output = self._forward(adapted_params, x)
                grad = self._compute_gradient(adapted_params, x, y, output)
                
                for k, v in grad.items():
                    if k not in query_grad:
                        query_grad[k] = np.zeros_like(v)
                    query_grad[k] += v
            
            # Average query gradient
            for k in query_grad:
                query_grad[k] /= len(query_set)
            
            # Meta-gradient (simplified first-order MAML)
            # In full MAML, we'd compute Hessian-vector product
            # For now, use first-order approximation
            for name in meta_grad:
                if name in query_grad:
                    meta_grad[name] += query_grad[name]
        
        # Average meta-gradient across tasks
        if tasks:
            for name in meta_grad:
                meta_grad[name] /= len(tasks)
        
        return meta_grad
    
    def _apply_meta_gradient(self, task_type: str, meta_grad: Dict[str, np.ndarray]):
        """
        Apply meta-gradient using Adam optimizer.
        
        Args:
            task_type: 'supervised' or 'rl'
            meta_grad: Meta-gradient to apply
        """
        meta_params = self.meta_params[task_type]
        opt_state = self.meta_optimizer_state[task_type]
        
        # Adam hyperparameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        opt_state['t'] += 1
        t = opt_state['t']
        
        for name, grad in meta_grad.items():
            if name not in meta_params['lr_multipliers']:
                continue
            
            # Initialize Adam state if needed
            if name not in opt_state['m']:
                opt_state['m'][name] = np.zeros_like(grad)
                opt_state['v'][name] = np.zeros_like(grad)
            
            # Adam update
            opt_state['m'][name] = beta1 * opt_state['m'][name] + (1 - beta1) * grad
            opt_state['v'][name] = beta2 * opt_state['v'][name] + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = opt_state['m'][name] / (1 - beta1 ** t)
            v_hat = opt_state['v'][name] / (1 - beta2 ** t)
            
            # Update meta-parameters
            meta_params['lr_multipliers'][name] -= self.maml_config.meta_lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Clip to reasonable range [0.1, 10.0]
            meta_params['lr_multipliers'][name] = np.clip(
                meta_params['lr_multipliers'][name],
                0.1, 10.0
            )
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Forward pass (no-op for Meta^3, just passes through)."""
        return input_data
    
    def _forward(self, params: Dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
        """
        Forward pass through network.
        
        Simplified for now - assumes linear output layer.
        
        Args:
            params: Network parameters
            x: Input
            
        Returns:
            Output
        """
        # This is a placeholder - in practice, would use actual network forward pass
        if 'W_out' in params:
            return params['W_out'] @ x
        return x
    
    def _compute_gradient(self,
                         params: Dict[str, np.ndarray],
                         x: np.ndarray,
                         y: np.ndarray,
                         output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute gradient of loss w.r.t. parameters.
        
        Args:
            params: Network parameters
            x: Input
            y: Target
            output: Network output
            
        Returns:
            Gradient dict
        """
        # MSE loss gradient
        error = output - y
        
        grad = {}
        if 'W_out' in params:
            grad['W_out'] = np.outer(error, x)
        
        return grad
    
    def add_task_to_batch(self, task: Dict[str, Any]):
        """
        Add task to meta-batch for later meta-update.
        
        Args:
            task: Task dict with support_set, query_set, etc.
        """
        self.meta_batch.append(task)
        self.tasks_seen += 1
        
        # Trigger meta-update when batch is full
        if len(self.meta_batch) >= self.maml_config.meta_batch_size:
            self.meta_update(self.meta_batch)
            self.meta_batch = []
    
    def reset(self):
        """Reset MAML state."""
        # Reset meta-parameters to uniform
        for task_type in self.meta_params:
            self.meta_params[task_type]['lr_multipliers'] = {}
            self.meta_params[task_type]['inner_lr'] = self.maml_config.inner_lr
            self.meta_params[task_type]['inner_steps'] = self.maml_config.inner_steps
        
        # Reset optimizer state
        for task_type in self.meta_optimizer_state:
            self.meta_optimizer_state[task_type] = {'m': {}, 'v': {}, 't': 0}
        
        # Reset task history
        self.task_history = []
        self.meta_batch = []
        self.meta_updates = 0
        self.tasks_seen = 0
    
    # Abstract method implementations (required by MetaLayer)
    def _compute_state_vector(self) -> np.ndarray:
        """Compute state vector for holographic encoding."""
        # Encode meta-update count and task stats
        values = [float(self.meta_updates), float(self.tasks_seen)]
        # Pad to 64 dimensions
        return np.array(values + [0.0] * (64 - len(values)))
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply suggestion (e.g., change meta_lr)."""
        if 'meta_lr' in suggestion:
            self.maml_config.meta_lr = suggestion['meta_lr']
            return True
        return False
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> tuple:
        """Evaluate suggestion."""
        return (0.8, "Meta^3 MAML accepts most suggestions")
    
    def _self_optimize_weights(self):
        """Optimize meta-lr based on performance."""
        pass
    
    def _self_optimize_synapses(self):
        """No synapse-level optimization for Meta^3."""
        pass
    
    def _self_optimize_neurons(self):
        """No neuron-level optimization for Meta^3."""
        pass
