"""Tests for benchmark framework."""

import os
import tempfile
import numpy as np

from throng3.benchmarks.config import TaskConfig, ExperimentConfig
from throng3.benchmarks.runner import BenchmarkRunner, TrainResult
from throng3.benchmarks.stats import StatisticalAnalyzer
from throng3.benchmarks.transfer import TransferBenchmark
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter, CartPoleAdapter


def test_config_creation():
    """Test that config dataclasses can be created."""
    task = TaskConfig(
        name="test_task",
        env_class=GridWorldAdapter,
        max_steps=100,
        convergence_threshold=0.1
    )
    
    assert task.name == "test_task"
    assert task.max_steps == 100
    
    config = ExperimentConfig(
        tasks=[task],
        n_seeds=5,
        pretrain_steps=50
    )
    
    assert config.n_seeds == 5
    assert len(config.tasks) == 1


def test_runner_basic():
    """Test that runner can train without crashing."""
    # GridWorld: 2D input (x, y), 4 actions
    pipeline = MetaNPipeline.create_adaptive(n_inputs=2, n_outputs=4)
    env = GridWorldAdapter()
    
    runner = BenchmarkRunner(pipeline, env)
    
    # Train for just a few steps
    result = runner.train_until_convergence(
        max_steps=50,
        convergence_threshold=0.1,
        convergence_window=5,
        verbose=False
    )
    
    assert isinstance(result, TrainResult)
    assert len(result.loss_history) > 0
    assert result.steps_to_convergence is not None



def test_checkpoint_roundtrip():
    """Test that checkpoints can be saved and loaded."""
    pipeline = MetaNPipeline.create_adaptive(n_inputs=2, n_outputs=4)
    env = GridWorldAdapter()
    
    runner = BenchmarkRunner(pipeline, env)
    
    # Create temporary checkpoint file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        checkpoint_path = f.name
    
    try:
        # Validate checkpoint works
        is_valid = runner.validate_checkpoint(checkpoint_path)
        assert is_valid, "Checkpoint validation failed"
        
    finally:
        # Clean up
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def test_convergence_detection():
    """Test that convergence is detected correctly."""
    pipeline = MetaNPipeline.create_adaptive(n_inputs=2, n_outputs=4)
    env = GridWorldAdapter()
    
    runner = BenchmarkRunner(pipeline, env)
    
    # Train with generous threshold to ensure convergence
    result = runner.train_until_convergence(
        max_steps=500,
        convergence_threshold=1.0,  # Very generous
        convergence_window=5,
        verbose=False
    )
    
    # Should converge with such a high threshold
    assert result.steps_to_convergence is not None
    assert result.steps_to_convergence <= 500


def test_stats_known_values():
    """Test statistical functions with known inputs."""
    analyzer = StatisticalAnalyzer()
    
    # Test speedup
    pretrained = [100, 110, 90]
    fresh = [200, 210, 190]
    
    speedup = analyzer.compute_speedup(pretrained, fresh)
    assert speedup > 1.5  # Should be ~2x
    
    # Test t-test
    t_stat, p_value = analyzer.t_test(pretrained, fresh)
    assert p_value < 0.05  # Should be significant
    
    # Test effect size
    effect = analyzer.effect_size(pretrained, fresh)
    assert effect > 0.8  # Should be large effect


def test_stats_edge_cases():
    """Test that stats handle edge cases gracefully."""
    analyzer = StatisticalAnalyzer()
    
    # Empty lists
    speedup = analyzer.compute_speedup([], [])
    assert speedup == 0.0
    
    # Single sample
    t_stat, p_value = analyzer.t_test([100], [200])
    assert p_value == 1.0  # Not enough samples
    
    # Zero variance
    pretrained = [100, 100, 100]
    fresh = [100, 100, 100]
    effect = analyzer.effect_size(pretrained, fresh)
    assert effect == 0.0


def test_mini_experiment():
    """Test full experiment with N=2 seeds (sanity check)."""
    # Create minimal config
    gridworld_task = TaskConfig(
        name="gridworld",
        env_class=GridWorldAdapter,
        max_steps=100,
        convergence_threshold=0.5
    )
    
    config = ExperimentConfig(
        tasks=[gridworld_task],
        n_seeds=2,
        pretrain_steps=50,
        verbose=False
    )
    
    # Run experiment
    benchmark = TransferBenchmark(config)
    results = benchmark.run_experiment()
    
    # Verify results structure
    assert len(results.pretrained_steps) == 2
    assert len(results.fresh_steps) == 2
    assert 'speedup' in results.statistics
    assert 'p_value' in results.statistics
    
    # Print summary for manual inspection
    print("\n" + results.summary())


def test_results_serialization():
    """Test that results can be saved and loaded."""
    gridworld_task = TaskConfig(
        name="gridworld",
        env_class=GridWorldAdapter,
        max_steps=50,
        convergence_threshold=0.5
    )
    
    config = ExperimentConfig(
        tasks=[gridworld_task],
        n_seeds=2,
        pretrain_steps=25,
        verbose=False
    )
    
    benchmark = TransferBenchmark(config)
    results = benchmark.run_experiment()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as f:
        results_path = f.name
    
    try:
        results.save(results_path)
        assert os.path.exists(results_path)
        
        # Verify file is valid JSON
        import json
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        assert 'pretrained_steps' in data
        assert 'fresh_steps' in data
        assert 'statistics' in data
        
    finally:
        if os.path.exists(results_path):
            os.remove(results_path)


if __name__ == '__main__':
    # Run tests
    print("Running benchmark tests...\n")
    
    test_config_creation()
    print("✓ test_config_creation")
    
    test_runner_basic()
    print("✓ test_runner_basic")
    
    test_checkpoint_roundtrip()
    print("✓ test_checkpoint_roundtrip")
    
    test_convergence_detection()
    print("✓ test_convergence_detection")
    
    test_stats_known_values()
    print("✓ test_stats_known_values")
    
    test_stats_edge_cases()
    print("✓ test_stats_edge_cases")
    
    test_mini_experiment()
    print("✓ test_mini_experiment")
    
    test_results_serialization()
    print("✓ test_results_serialization")
    
    print("\n✓ All tests passed!")
