#!/usr/bin/env python
"""
Phase 3 Statistical Validation Experiment Runner

Runs N=30 seed transfer learning experiments on GridWorld and CartPole
to validate Meta^N transfer learning capabilities.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

from throng3.benchmarks.config import TaskConfig, ExperimentConfig
from throng3.benchmarks.transfer import TransferBenchmark
from throng3.environments import GridWorldAdapter, CartPoleAdapter


def create_gridworld_config(n_seeds=30, pretrain_steps=500):
    """Create GridWorld experiment configuration."""
    task = TaskConfig(
        name="gridworld",
        env_class=GridWorldAdapter,
        max_steps=2000,
        convergence_threshold=0.3,
        convergence_window=20
    )
    
    return ExperimentConfig(
        tasks=[task],
        n_seeds=n_seeds,
        pretrain_steps=pretrain_steps,
        results_dir="./results/phase3",
        verbose=True
    )


def create_cartpole_config(n_seeds=30, pretrain_steps=500):
    """Create CartPole experiment configuration."""
    task = TaskConfig(
        name="cartpole",
        env_class=CartPoleAdapter,
        max_steps=2000,
        convergence_threshold=0.3,
        convergence_window=20
    )
    
    return ExperimentConfig(
        tasks=[task],
        n_seeds=n_seeds,
        pretrain_steps=pretrain_steps,
        results_dir="./results/phase3",
        verbose=True
    )


def run_experiment(task_name, n_seeds=30, pretrain_steps=500):
    """
    Run a full N-seed experiment.
    
    Args:
        task_name: 'gridworld' or 'cartpole'
        n_seeds: Number of random seeds
        pretrain_steps: Steps to pretrain on source task
    
    Returns:
        ExperimentResults object
    """
    print(f"\n{'='*70}")
    print(f"Phase 3 Statistical Validation — {task_name.upper()}")
    print(f"{'='*70}")
    print(f"N seeds: {n_seeds}")
    print(f"Pretrain steps: {pretrain_steps}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Create config
    if task_name == "gridworld":
        config = create_gridworld_config(n_seeds, pretrain_steps)
    elif task_name == "cartpole":
        config = create_cartpole_config(n_seeds, pretrain_steps)
    else:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Run experiment
    start_time = time.time()
    benchmark = TransferBenchmark(config)
    results = benchmark.run_experiment()
    elapsed = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(
        config.results_dir,
        f"{task_name}_n{n_seeds}_{timestamp}.json"
    )
    results.save(results_path)
    
    print(f"\n{'='*70}")
    print(f"Experiment Complete!")
    print(f"{'='*70}")
    print(f"Duration: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}\n")
    
    # Print summary
    print(results.summary())
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 3 statistical validation experiments"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="gridworld",
        choices=["gridworld", "cartpole", "both"],
        help="Which task to run (default: gridworld)"
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=30,
        help="Number of random seeds (default: 30)"
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=500,
        help="Pretraining steps (default: 500)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot with N=5 seeds"
    )
    
    args = parser.parse_args()
    
    # Override n_seeds for pilot
    if args.pilot:
        args.n_seeds = 5
        print("\n🧪 PILOT MODE: Running with N=5 seeds\n")
    
    # Run experiments
    if args.task == "both":
        print("\n📊 Running both GridWorld and CartPole experiments\n")
        run_experiment("gridworld", args.n_seeds, args.pretrain_steps)
        run_experiment("cartpole", args.n_seeds, args.pretrain_steps)
    else:
        run_experiment(args.task, args.n_seeds, args.pretrain_steps)
    
    print("\n✅ Phase 3 experiments complete!\n")


if __name__ == "__main__":
    main()
