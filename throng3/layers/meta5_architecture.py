"""
Meta^5: ArchitectureSearch — NAS-lite for the Fractal Stack

Searches over architectural configurations:
- Number of neurons per layer
- Connection topology (sparse, dense, modular)
- Which meta-layers to activate
- Layer width/depth tradeoffs
- Module composition

Uses evolutionary strategies + performance-based pruning.
Not a full NAS (too expensive) — more like structured hyperparameter
optimization over architectural choices.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import copy

from throng3.core.meta_layer import MetaLayer
from throng3.core.signal import SignalDirection, SignalType, SignalPriority


@dataclass
class ArchitectureConfig:
    """Configuration for architecture search."""
    population_size: int = 8         # Number of candidate architectures
    mutation_rate: float = 0.1       # Probability of mutating each parameter
    mutation_scale: float = 0.2      # Scale of mutations
    tournament_size: int = 3         # Tournament selection size
    evaluation_steps: int = 100      # Steps to evaluate each architecture
    elite_fraction: float = 0.25     # Top fraction to keep
    search_interval: int = 200       # Steps between architecture proposals


@dataclass
class ArchitectureCandidate:
    """A candidate architecture configuration."""
    id: int
    params: Dict[str, Any]
    fitness: float = 0.0
    evaluations: int = 0
    generation: int = 0
    parent_id: Optional[int] = None


class ArchitectureSearch(MetaLayer):
    """
    Meta^5: Architecture search over the fractal stack.
    
    Proposes architectural changes and evaluates their impact
    on system-wide performance.
    """
    
    def __init__(self, config: Optional[ArchitectureConfig] = None, **kwargs):
        cfg = config or ArchitectureConfig()
        super().__init__(level=5, name="ArchitectureSearch", config=vars(cfg))
        self.arch_config = cfg
        
        # Search space definition
        self.search_space = self._define_search_space()
        
        # Population of candidates
        self.population: List[ArchitectureCandidate] = []
        self._next_id = 0
        self._generation = 0
        self._init_population()
        
        # Current active architecture
        self.current_arch: Optional[ArchitectureCandidate] = None
        if self.population:
            self.current_arch = self.population[0]
        
        # Performance tracking
        self._fitness_history: deque = deque(maxlen=500)
        self._best_fitness = -float('inf')
        self._best_architecture: Optional[Dict] = None
        self._stagnation_counter = 0
        
        # Architecture evaluation state
        self._evaluating: Optional[ArchitectureCandidate] = None
        self._evaluation_step = 0
        self._evaluation_rewards: List[float] = []
    
    def _define_search_space(self) -> Dict[str, Dict]:
        """Define the architectural search space."""
        return {
            'n_neurons': {
                'type': 'int',
                'range': [100, 10000],
                'default': 1000,
                'description': 'Number of neurons in Meta^0',
            },
            'sparsity': {
                'type': 'float',
                'range': [0.5, 0.99],
                'default': 0.9,
                'description': 'Weight matrix sparsity',
            },
            'exc_ratio': {
                'type': 'float',
                'range': [0.5, 0.95],
                'default': 0.8,
                'description': 'Excitatory neuron ratio',
            },
            'activation': {
                'type': 'categorical',
                'options': ['relu', 'tanh', 'sigmoid'],
                'default': 'relu',
                'description': 'Activation function',
            },
            'learning_rule': {
                'type': 'categorical',
                'options': ['stdp', 'hebbian', 'both'],
                'default': 'stdp',
                'description': 'Default learning rule',
            },
            'prune_interval': {
                'type': 'int',
                'range': [50, 500],
                'default': 100,
                'description': 'Steps between pruning',
            },
            'encoding_scheme': {
                'type': 'categorical',
                'options': ['rate', 'sparse', 'predictive', 'population'],
                'default': 'rate',
                'description': 'Neural encoding scheme',
            },
            'exploration_rate': {
                'type': 'float',
                'range': [0.01, 0.5],
                'default': 0.1,
                'description': 'Initial exploration rate',
            },
            'holographic_dim': {
                'type': 'int',
                'range': [32, 256],
                'default': 128,
                'description': 'Holographic state dimensionality',
            },
            'meta_layers_active': {
                'type': 'bitmask',
                'n_bits': 6,
                'default': [True] * 6,
                'description': 'Which meta-layers are active',
            },
        }
    
    def _init_population(self):
        """Initialize population with random candidates."""
        # First candidate: defaults
        default = self._create_candidate(self._get_defaults())
        self.population.append(default)
        
        # Random variations
        for _ in range(self.arch_config.population_size - 1):
            params = self._random_architecture()
            candidate = self._create_candidate(params)
            self.population.append(candidate)
    
    def _create_candidate(self, params: Dict) -> ArchitectureCandidate:
        """Create a new architecture candidate."""
        candidate = ArchitectureCandidate(
            id=self._next_id,
            params=params,
            generation=self._generation,
        )
        self._next_id += 1
        return candidate
    
    def _get_defaults(self) -> Dict:
        """Get default architecture parameters."""
        return {
            name: spec['default']
            for name, spec in self.search_space.items()
        }
    
    def _random_architecture(self) -> Dict:
        """Generate a random architecture."""
        params = {}
        for name, spec in self.search_space.items():
            if spec['type'] == 'float':
                low, high = spec['range']
                params[name] = np.random.uniform(low, high)
            elif spec['type'] == 'int':
                low, high = spec['range']
                params[name] = int(np.random.randint(low, high + 1))
            elif spec['type'] == 'categorical':
                params[name] = np.random.choice(spec['options'])
            elif spec['type'] == 'bitmask':
                params[name] = [np.random.random() > 0.3 for _ in range(spec['n_bits'])]
        return params
    
    def _mutate(self, params: Dict) -> Dict:
        """Mutate an architecture."""
        mutated = copy.deepcopy(params)
        
        for name, spec in self.search_space.items():
            if np.random.random() > self.arch_config.mutation_rate:
                continue
            
            if spec['type'] == 'float':
                low, high = spec['range']
                delta = np.random.randn() * self.arch_config.mutation_scale * (high - low)
                mutated[name] = np.clip(mutated[name] + delta, low, high)
            
            elif spec['type'] == 'int':
                low, high = spec['range']
                delta = int(np.random.randn() * self.arch_config.mutation_scale * (high - low))
                mutated[name] = int(np.clip(mutated[name] + delta, low, high))
            
            elif spec['type'] == 'categorical':
                mutated[name] = np.random.choice(spec['options'])
            
            elif spec['type'] == 'bitmask':
                idx = np.random.randint(spec['n_bits'])
                mutated[name][idx] = not mutated[name][idx]
        
        return mutated
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Uniform crossover between two architectures."""
        child = {}
        for name in self.search_space:
            if np.random.random() < 0.5:
                child[name] = copy.deepcopy(parent1.get(name, self.search_space[name]['default']))
            else:
                child[name] = copy.deepcopy(parent2.get(name, self.search_space[name]['default']))
        return child
    
    def _tournament_select(self) -> ArchitectureCandidate:
        """Tournament selection."""
        candidates = np.random.choice(
            len(self.population),
            size=min(self.arch_config.tournament_size, len(self.population)),
            replace=False
        )
        best = max(candidates, key=lambda i: self.population[i].fitness)
        return self.population[best]
    
    def optimize(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run one architecture search step.
        
        1. If evaluating: collect performance data
        2. If evaluation complete: update fitness, possibly evolve
        3. Periodically: propose new architecture to try
        """
        self.process_inbox()
        
        # Get system performance
        system_summary = context.get('system_summary', {})
        layer_results = context.get('layer_results', {})
        
        # Compute current fitness
        current_fitness = self._compute_fitness(system_summary, layer_results)
        self._fitness_history.append(current_fitness)
        
        # Update evaluation if in progress
        if self._evaluating is not None:
            self._evaluation_rewards.append(current_fitness)
            self._evaluation_step += 1
            
            if self._evaluation_step >= self.arch_config.evaluation_steps:
                # Evaluation complete
                self._evaluating.fitness = np.mean(self._evaluation_rewards)
                self._evaluating.evaluations += 1
                
                # Check if new best
                if self._evaluating.fitness > self._best_fitness:
                    self._best_fitness = self._evaluating.fitness
                    self._best_architecture = copy.deepcopy(self._evaluating.params)
                    self._stagnation_counter = 0
                else:
                    self._stagnation_counter += 1
                
                self._evaluating = None
                self._evaluation_rewards = []
        
        # Periodic evolution
        if (self._optimization_step % self.arch_config.search_interval == 0 
            and self._evaluating is None):
            self._evolve()
            
            # Select next architecture to evaluate
            candidate = self._select_next_to_evaluate()
            if candidate:
                self._evaluating = candidate
                self._evaluation_step = 0
                
                # Propose architecture change DOWN
                self._propose_architecture(candidate.params)
        
        # Metrics
        best_fitness = max(c.fitness for c in self.population) if self.population else 0
        self.metrics.update(1.0 - best_fitness, best_fitness)
        
        # Signal UP
        self.signal(
            direction=SignalDirection.UP,
            signal_type=SignalType.PERFORMANCE,
            payload={
                'best_fitness': self._best_fitness,
                'generation': self._generation,
                'population_size': len(self.population),
                'stagnation': self._stagnation_counter,
                'best_architecture': self._best_architecture,
            },
        )
        
        return {
            'best_fitness': self._best_fitness,
            'current_fitness': current_fitness,
            'generation': self._generation,
            'evaluating': self._evaluating is not None,
            'metrics': self.metrics,
        }
    
    def _compute_fitness(self, summary: Dict, results: Dict) -> float:
        """Compute fitness from system performance."""
        scores = []
        
        # Coherence
        coherence = summary.get('coherence', 0.5)
        scores.append(coherence)
        
        # Average layer performance
        for level, result in (results.items() if isinstance(results, dict) else []):
            if isinstance(result, dict):
                loss = result.get('loss', 1.0)
                scores.append(1.0 - min(loss, 1.0))
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _evolve(self):
        """Run one generation of evolution."""
        if len(self.population) < 3:
            return
        
        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)
        
        # Keep elite
        n_elite = max(1, int(len(self.population) * self.arch_config.elite_fraction))
        elite = self.population[:n_elite]
        
        # Generate offspring
        offspring = []
        while len(offspring) < self.arch_config.population_size - n_elite:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            child_params = self._crossover(parent1.params, parent2.params)
            child_params = self._mutate(child_params)
            
            child = self._create_candidate(child_params)
            child.generation = self._generation + 1
            child.parent_id = parent1.id
            offspring.append(child)
        
        self.population = elite + offspring
        self._generation += 1
    
    def _select_next_to_evaluate(self) -> Optional[ArchitectureCandidate]:
        """Select the next candidate to evaluate."""
        # Prefer unevaluated candidates
        unevaluated = [c for c in self.population if c.evaluations == 0]
        if unevaluated:
            return unevaluated[0]
        
        # Otherwise, least-evaluated
        return min(self.population, key=lambda c: c.evaluations)
    
    def _propose_architecture(self, params: Dict):
        """Send architecture proposal DOWN to relevant layers."""
        # Meta^0 parameters
        meta0_params = {}
        for key in ('n_neurons', 'sparsity', 'exc_ratio', 'activation'):
            if key in params:
                meta0_params[key] = params[key]
        
        if meta0_params:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.SUGGESTION,
                payload=meta0_params,
                target_level=0,
                requires_response=True,
            )
        
        # Meta^1 parameters
        if 'learning_rule' in params:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.SUGGESTION,
                payload={'active_rule': params['learning_rule']},
                target_level=1,
                requires_response=True,
            )
        
        if 'prune_interval' in params:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.SUGGESTION,
                payload={'prune_interval': params['prune_interval']},
                target_level=1,
                requires_response=True,
            )
        
        # Meta^3 parameters
        if 'encoding_scheme' in params:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.SUGGESTION,
                payload={'encoding_scheme': params['encoding_scheme']},
                target_level=3,
                requires_response=True,
            )
        
        # Meta^4 parameters
        if 'exploration_rate' in params:
            self.signal(
                direction=SignalDirection.DOWN,
                signal_type=SignalType.SUGGESTION,
                payload={'exploration_rate': params['exploration_rate']},
                target_level=4,
                requires_response=True,
            )
    
    def _compute_state_vector(self) -> np.ndarray:
        """Holographic state for Meta^5."""
        best_params = self._best_architecture or self._get_defaults()
        
        # Encode architecture parameters numerically
        arch_vector = []
        for name, spec in self.search_space.items():
            val = best_params.get(name, spec['default'])
            if spec['type'] == 'float':
                arch_vector.append(val)
            elif spec['type'] == 'int':
                low, high = spec['range']
                arch_vector.append((val - low) / max(high - low, 1))
            elif spec['type'] == 'categorical':
                idx = spec['options'].index(val) if val in spec['options'] else 0
                arch_vector.append(idx / max(len(spec['options']) - 1, 1))
            elif spec['type'] == 'bitmask':
                arch_vector.extend([float(b) for b in val])
        
        arch_vector.extend([
            self._best_fitness,
            float(self._generation),
            float(self._stagnation_counter),
            self.metrics.loss,
            self.metrics.accuracy,
        ])
        
        return np.array(arch_vector)
    
    def _apply_suggestion(self, suggestion: Dict[str, Any]) -> bool:
        """Apply suggestion from LLM/Meta^N."""
        applied = False
        
        if 'mutation_rate' in suggestion:
            self.arch_config.mutation_rate = np.clip(suggestion['mutation_rate'], 0, 1)
            applied = True
        
        if 'population_size' in suggestion:
            self.arch_config.population_size = max(4, int(suggestion['population_size']))
            applied = True
        
        if 'force_architecture' in suggestion:
            # Force a specific architecture (from LLM insight)
            params = suggestion['force_architecture']
            candidate = self._create_candidate(params)
            candidate.generation = self._generation
            self.population.append(candidate)
            self._evaluating = candidate
            self._evaluation_step = 0
            self._evaluation_rewards = []
            applied = True
        
        return applied
    
    def _evaluate_suggestion(self, suggestion: Dict[str, Any]) -> Tuple[float, str]:
        """Evaluate suggestions."""
        if 'force_architecture' in suggestion:
            return 0.9, "LLM-suggested architecture: high priority"
        return 0.5, "Generic suggestion"
    
    def _self_optimize_weights(self):
        """Increase mutation when stagnating."""
        if self._stagnation_counter > 10:
            self.arch_config.mutation_rate = min(
                self.arch_config.mutation_rate * 1.1, 0.5
            )
            self.arch_config.mutation_scale = min(
                self.arch_config.mutation_scale * 1.05, 0.5
            )
    
    def _self_optimize_synapses(self):
        """Trim population of consistently poor performers."""
        if len(self.population) > self.arch_config.population_size * 2:
            self.population.sort(key=lambda c: c.fitness, reverse=True)
            self.population = self.population[:self.arch_config.population_size]
    
    def _self_optimize_neurons(self):
        """Expand search space if stagnating heavily."""
        pass
