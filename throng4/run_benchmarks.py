"""
CLI Runner for Benchmark Suite.

Usage:
    python -m throng4.run_benchmarks --test generalization --train-level 1 --test-level 5
    python -m throng4.run_benchmarks --test progressive --episodes-per 100
    python -m throng4.run_benchmarks --test all
"""

import argparse
import sys
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.benchmarks.benchmark import BenchmarkSuite


def main():
    parser = argparse.ArgumentParser(description='Run Tetris benchmark tests')
    
    parser.add_argument(
        '--test',
        choices=['generalization', 'progressive', 'all'],
        required=True,
        help='Which test to run'
    )
    
    # Generalization test args
    parser.add_argument('--train-level', type=int, default=1,
                       help='Training level (1-7)')
    parser.add_argument('--test-level', type=int, default=5,
                       help='Test level (1-7)')
    parser.add_argument('--train-episodes', type=int, default=100,
                       help='Training episodes')
    parser.add_argument('--test-episodes', type=int, default=20,
                       help='Test episodes')
    
    # Progressive test args
    parser.add_argument('--episodes-per', type=int, default=100,
                       help='Episodes per level in progressive test')
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6, 7],
                       help='Levels for progressive test')
    
    # Agent config
    parser.add_argument('--hidden', type=int, default=48,
                       help='Hidden layer size')
    parser.add_argument('--epsilon', type=float, default=0.20,
                       help='Initial epsilon')
    parser.add_argument('--lr', type=float, default=0.005,
                       help='Learning rate')
    
    # Output
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output JSON file')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Create agent config
    config = AgentConfig(
        n_hidden=args.hidden,
        epsilon=args.epsilon,
        learning_rate=args.lr
    )
    
    # Create benchmark suite
    suite = BenchmarkSuite(verbose=not args.quiet)
    
    # Run tests
    if args.test == 'generalization' or args.test == 'all':
        print("\n" + "="*60)
        print("RUNNING GENERALIZATION TEST")
        print("="*60)
        
        result = suite.run_generalization(
            agent_class=PortableNNAgent,
            adapter_class=TetrisAdapter,
            train_level=args.train_level,
            test_level=args.test_level,
            train_episodes=args.train_episodes,
            test_episodes=args.test_episodes,
            config=config
        )
        
        print(f"\nGeneralization Gap: {result.mean_score:.2f} lines")
        print(f"  (trained on level {args.train_level}, tested on level {args.test_level})")
    
    if args.test == 'progressive' or args.test == 'all':
        print("\n" + "="*60)
        print("RUNNING PROGRESSIVE CURRICULUM TEST")
        print("="*60)
        
        results = suite.run_progressive(
            agent_class=PortableNNAgent,
            adapter_class=TetrisAdapter,
            levels=args.levels,
            episodes_per_level=args.episodes_per,
            config=config
        )
        
        print("\nProgressive Learning Summary:")
        for i, (level, res) in enumerate(zip(args.levels, results)):
            improvement = ""
            if i > 0:
                prev_mean = results[i-1].mean_score
                delta = res.mean_score - prev_mean
                improvement = f" ({delta:+.1f} from prev)"
            print(f"  Level {level}: {res.mean_score:.2f} ± {res.std_score:.2f}{improvement}")
    
    # Export results
    suite.export_results(args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    summary = suite.get_summary()
    for test_type, stats in summary.items():
        print(f"\n{test_type}:")
        print(f"  Tests run: {stats['count']}")
        print(f"  Mean score: {stats['mean_score']:.2f}")
        print(f"  Best overall: {stats['best_overall']:.0f}")
    
    print(f"\nResults saved to: {args.output}\n")


if __name__ == '__main__':
    main()
