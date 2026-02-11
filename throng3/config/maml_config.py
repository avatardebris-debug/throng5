"""
MAML Configuration

Configuration for Model-Agnostic Meta-Learning (MAML).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MAMLConfig:
    """Configuration for MAML meta-learning."""
    
    # Meta-learning rate (outer loop)
    meta_lr: float = 0.001
    
    # Inner loop adaptation steps
    inner_steps: int = 5
    
    # Inner loop learning rate
    inner_lr: float = 0.01
    
    # Task conditioning
    use_task_conditioning: bool = True
    
    # Meta-batch size (number of tasks per meta-update)
    meta_batch_size: int = 4
    
    # First-order MAML (faster, less accurate)
    first_order: bool = False
