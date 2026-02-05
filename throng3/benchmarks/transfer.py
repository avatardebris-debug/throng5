"""Transfer learning benchmark orchestrator."""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from throng3.pipeline import MetaNPipeline
from throng3.benchmarks.config import TaskConfig, ExperimentConfig
from throng3.benchmarks.runner import BenchmarkRunner, TrainResult
from throng3.benchmarks.stats import StatisticalAnalyzer


@dataclass
class ExperimentResults:
    """Results from a full transfer learning experiment."""
    
    config: ExperimentConfig
    pretrained_steps: List[int]
    fresh_steps: List[int]
    statistics: Dict[str, Any]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        stats = self.statistics
        
        lines = [
            "=" * 60,
            "Transfer Learning Experiment Results",
            "=" * 60,
            f"Tasks: {' → '.join([t.name for t in self.config.tasks])}",
            f"Seeds: {self.config.n_seeds}",
            "",
            "Results:",
            f"  Fresh (no transfer):     {stats['fresh_mean']:.1f} steps (95% CI: {stats['fresh_ci'][0]:.1f}-{stats['fresh_ci'][1]:.1f})",
            f"  Pretrained (transfer):   {stats['pretrained_mean']:.1f} steps (95% CI: {stats['pretrained_ci'][0]:.1f}-{stats['pretrained_ci'][1]:.1f})",
            f"  Speedup:                 {stats['speedup']:.2f}x",
            "",
            "Statistical Significance:",
            f"  t-statistic:             {stats['t_statistic']:.3f}",
            f"  p-value:                 {stats['p_value']:.4f}",
            f"  Effect size (Cohen's d): {stats['effect_size']:.3f}",
            f"  Significant (p<0.05):    {'✓ YES' if stats['significant'] else '✗ NO'}",
            "=" * 60,
        ]
        
        return "\n".join(lines)
    
    def save(self, path: str) -> None:
        """Save results to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'pretrained_steps': self.pretrained_steps,
            'fresh_steps': self.fresh_steps,
            'statistics': self.statistics,
            'config': {
                'n_seeds': self.config.n_seeds,
                'pretrain_steps': self.config.pretrain_steps,
                'tasks': [t.name for t in self.config.tasks],
            }
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class TransferBenchmark:
    """
    Orchestrates full transfer learning experiments.
    
    Runs N-seed experiments comparing:
    - Fresh training (no transfer)
    - Pretrained training (with transfer from source task)
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize transfer benchmark.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.analyzer = StatisticalAnalyzer()
        
    def run_experiment(self) -> ExperimentResults:
        """
        Run full N-seed transfer learning experiment.
        
        Returns:
            ExperimentResults with statistical analysis
        """
        pretrained_steps = []
        fresh_steps = []
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Running Transfer Learning Experiment")
            print(f"{'='*60}")
            print(f"Tasks: {' → '.join([t.name for t in self.config.tasks])}")
            print(f"Seeds: {self.config.n_seeds}")
            print(f"{'='*60}\n")
        
        for seed in range(self.config.n_seeds):
            if self.config.verbose:
                print(f"\n--- Seed {seed + 1}/{self.config.n_seeds} ---")
            
            # Train fresh (no transfer)
            fresh_result = self._train_fresh(seed)
            fresh_steps.append(fresh_result)
            
            if self.config.verbose:
                print(f"  Fresh: {fresh_result} steps")
            
            # Train with transfer
            pretrained_result = self._train_with_transfer(seed)
            pretrained_steps.append(pretrained_result)
            
            if self.config.verbose:
                print(f"  Pretrained: {pretrained_result} steps")
                speedup = fresh_result / max(pretrained_result, 1)
                print(f"  Speedup: {speedup:.2f}x")
        
        # Analyze results
        statistics = self.analyzer.summarize_results(pretrained_steps, fresh_steps)
        
        results = ExperimentResults(
            config=self.config,
            pretrained_steps=pretrained_steps,
            fresh_steps=fresh_steps,
            statistics=statistics
        )
        
        if self.config.verbose:
            print(f"\n{results.summary()}")
        
        return results
    
    def _train_fresh(self, seed: int) -> int:
        """
        Train from scratch on target task.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Steps to convergence
        """
        np.random.seed(seed)
        
        # Get target task (last in list)
        target_task = self.config.tasks[-1]
        
        # Create environment to get input/output sizes
        env = target_task.env_class(**target_task.env_kwargs)
        obs = env.reset()
        n_inputs = len(obs)
        
        # Infer output size from environment
        # For now, assume discrete action space with reasonable default
        n_outputs = 4  # Default for GridWorld
        if hasattr(env, 'env') and hasattr(env.env, 'action_space'):
            n_outputs = env.env.action_space.n
        
        # Create fresh pipeline with correct sizes
        pipeline = MetaNPipeline.create_adaptive(
            n_inputs=n_inputs,
            n_outputs=n_outputs
        )
        
        # Create runner and train
        runner = BenchmarkRunner(pipeline, env)
        result = runner.train_until_convergence(
            max_steps=target_task.max_steps,
            convergence_threshold=target_task.convergence_threshold,
            convergence_window=target_task.convergence_window,
            verbose=False
        )
        
        return result.steps_to_convergence or target_task.max_steps
    
    def _train_with_transfer(self, seed: int) -> int:
        """
        Pretrain on source tasks, then transfer to target.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Steps to convergence on target task
        """
        np.random.seed(seed)
        
        # Get environment info from first task to create pipeline
        first_task = self.config.tasks[0]
        first_env = first_task.env_class(**first_task.env_kwargs)
        obs = first_env.reset()
        n_inputs = len(obs)
        
        # Infer output size
        n_outputs = 4  # Default
        if hasattr(first_env, 'env') and hasattr(first_env.env, 'action_space'):
            n_outputs = first_env.env.action_space.n
        
        # Create pipeline (will be reused across tasks)
        pipeline = MetaNPipeline.create_adaptive(
            n_inputs=n_inputs,
            n_outputs=n_outputs
        )
        
        # Pretrain on source tasks (all but last)
        for task_config in self.config.tasks[:-1]:
            env = task_config.env_class(**task_config.env_kwargs)
            runner = BenchmarkRunner(pipeline, env)
            
            # Train for fixed number of steps
            runner.train_until_convergence(
                max_steps=self.config.pretrain_steps,
                convergence_threshold=task_config.convergence_threshold,
                convergence_window=task_config.convergence_window,
                verbose=False
            )
        
        # Now train on target task (last in list)
        target_task = self.config.tasks[-1]
        target_env = target_task.env_class(**target_task.env_kwargs)
        target_runner = BenchmarkRunner(pipeline, target_env)
        
        result = target_runner.train_until_convergence(
            max_steps=target_task.max_steps,
            convergence_threshold=target_task.convergence_threshold,
            convergence_window=target_task.convergence_window,
            verbose=False
        )
        
        return result.steps_to_convergence or target_task.max_steps
