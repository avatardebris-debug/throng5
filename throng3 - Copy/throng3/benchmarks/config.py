"""Configuration dataclasses for benchmark experiments."""

from dataclasses import dataclass, field
from typing import List, Type, Optional
from throng3.environments.adapter import EnvironmentAdapter


@dataclass
class TaskConfig:
    """Configuration for a single task in the benchmark."""
    
    name: str
    """Human-readable task name (e.g., 'gridworld', 'cartpole')."""
    
    env_class: Type[EnvironmentAdapter]
    """Environment adapter class to instantiate."""
    
    max_steps: int = 5000
    """Maximum training steps before timeout."""
    
    convergence_threshold: float = 0.1
    """Loss threshold for convergence detection."""
    
    convergence_window: int = 10
    """Number of steps to average for convergence check."""
    
    env_kwargs: dict = field(default_factory=dict)
    """Additional kwargs to pass to environment constructor."""


@dataclass
class ExperimentConfig:
    """Configuration for a full transfer learning experiment."""
    
    tasks: List[TaskConfig]
    """List of tasks to benchmark. First task is source, last is target."""
    
    n_seeds: int = 30
    """Number of random seeds to run (for statistical significance)."""
    
    pretrain_steps: int = 1000
    """Steps to train on source task before transfer."""
    
    checkpoint_dir: str = "./checkpoints"
    """Directory to save/load model checkpoints."""
    
    results_dir: str = "./results"
    """Directory to save experiment results."""
    
    verbose: bool = True
    """Whether to print progress during experiments."""
