"""
Benchmark Suite — Generalization, Progressive, and Transfer Tests.

Provides automated testing protocols for:
1. Generalization: Train on level N → test on level M
2. Progressive: Auto-advance through curriculum levels
3. Cross-environment transfer: GridWorld → Tetris, etc.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import json


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    test_type: str
    mean_score: float
    std_score: float
    best_score: float
    worst_score: float
    learning_curve: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            'test_type': self.test_type,
            'mean_score': float(self.mean_score),
            'std_score': float(self.std_score),
            'best_score': float(self.best_score),
            'worst_score': float(self.worst_score),
            'learning_curve': [float(x) for x in self.learning_curve],
            'metadata': self.metadata
        }


class BenchmarkSuite:
    """
    Automated benchmark suite for RL agents.
    
    Tests:
    - Generalization: Train level A → evaluate level B
    - Progressive: Curriculum learning through levels 1→7
    - Transfer: Cross-environment weight transfer
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize benchmark suite.
        
        Args:
            verbose: Whether to print progress
        """
        self.verbose = verbose
        self.results = []
    
    def run_generalization(
        self,
        agent_class,
        adapter_class,
        train_level: int,
        test_level: int,
        train_episodes: int = 100,
        test_episodes: int = 20,
        **agent_kwargs
    ) -> BenchmarkResult:
        """
        Test generalization: train on one level, test on another.
        
        Args:
            agent_class: PortableNNAgent class
            adapter_class: Environment adapter class (e.g., TetrisAdapter)
            train_level: Level to train on
            test_level: Level to test on
            train_episodes: Number of training episodes
            test_episodes: Number of test episodes
            **agent_kwargs: Additional args for agent (n_features, config, etc.)
            
        Returns:
            BenchmarkResult with test performance
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GENERALIZATION TEST")
            print(f"Train: Level {train_level} ({train_episodes} episodes)")
            print(f"Test:  Level {test_level} ({test_episodes} episodes)")
            print(f"{'='*60}\n")
        
        # Training phase
        train_adapter = adapter_class(level=train_level)
        if 'n_features' not in agent_kwargs:
            agent_kwargs['n_features'] = train_adapter.n_features
        
        agent = agent_class(**agent_kwargs)
        
        train_scores = []
        for ep in range(train_episodes):
            score = self._run_episode(agent, train_adapter, explore=True)
            train_scores.append(score)
            
            if self.verbose and (ep + 1) % 20 == 0:
                recent = train_scores[-20:]
                print(f"  Train {ep+1:3d}/{train_episodes}: "
                      f"avg={np.mean(recent):.1f} best={max(recent):.0f}")
        
        # Testing phase (frozen weights, no exploration)
        test_adapter = adapter_class(level=test_level)
        agent.epsilon = 0.0  # Greedy only
        
        test_scores = []
        for ep in range(test_episodes):
            score = self._run_episode(agent, test_adapter, explore=False)
            test_scores.append(score)
        
        result = BenchmarkResult(
            test_type='generalization',
            mean_score=np.mean(test_scores),
            std_score=np.std(test_scores),
            best_score=max(test_scores),
            worst_score=min(test_scores),
            learning_curve=train_scores,
            metadata={
                'train_level': train_level,
                'test_level': test_level,
                'train_episodes': train_episodes,
                'test_episodes': test_episodes,
                'test_scores': test_scores
            }
        )
        
        if self.verbose:
            print(f"\n  Test Results (Level {test_level}):")
            print(f"    Mean:  {result.mean_score:.2f} ± {result.std_score:.2f}")
            print(f"    Best:  {result.best_score:.0f}")
            print(f"    Worst: {result.worst_score:.0f}\n")
        
        self.results.append(result)
        return result
    
    def run_progressive(
        self,
        agent_class,
        adapter_class,
        levels: List[int] = [1, 2, 3, 4, 5, 6, 7],
        episodes_per_level: int = 100,
        **agent_kwargs
    ) -> List[BenchmarkResult]:
        """
        Progressive curriculum: train through levels sequentially.
        
        Args:
            agent_class: PortableNNAgent class
            adapter_class: Environment adapter class
            levels: Sequence of levels to train on
            episodes_per_level: Episodes per level
            **agent_kwargs: Additional args for agent
            
        Returns:
            List of BenchmarkResults (one per level)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"PROGRESSIVE CURRICULUM TEST")
            print(f"Levels: {levels}")
            print(f"Episodes per level: {episodes_per_level}")
            print(f"{'='*60}\n")
        
        # Initialize agent on first level
        first_adapter = adapter_class(level=levels[0])
        if 'n_features' not in agent_kwargs:
            agent_kwargs['n_features'] = first_adapter.n_features
        
        agent = agent_class(**agent_kwargs)
        
        level_results = []
        
        for level in levels:
            if self.verbose:
                print(f"\n--- Level {level} ---")
            
            adapter = adapter_class(level=level)
            scores = []
            
            for ep in range(episodes_per_level):
                score = self._run_episode(agent, adapter, explore=True)
                scores.append(score)
                
                if self.verbose and (ep + 1) % 20 == 0:
                    recent = scores[-20:]
                    print(f"  {ep+1:3d}/{episodes_per_level}: "
                          f"avg={np.mean(recent):.1f} best={max(recent):.0f} "
                          f"ε={agent.epsilon:.3f}")
            
            result = BenchmarkResult(
                test_type='progressive',
                mean_score=np.mean(scores),
                std_score=np.std(scores),
                best_score=max(scores),
                worst_score=min(scores),
                learning_curve=scores,
                metadata={
                    'level': level,
                    'episodes': episodes_per_level,
                    'final_epsilon': agent.epsilon
                }
            )
            
            level_results.append(result)
            self.results.append(result)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("PROGRESSIVE SUMMARY")
            print(f"{'='*60}")
            for i, (level, res) in enumerate(zip(levels, level_results)):
                print(f"  Level {level}: {res.mean_score:.2f} ± {res.std_score:.2f} "
                      f"(best={res.best_score:.0f})")
            print()
        
        return level_results
    
    def run_transfer(
        self,
        source_agent,
        source_adapter,
        target_adapter_class,
        target_level: int,
        source_episodes: int = 100,
        target_episodes: int = 50,
        freeze_source: bool = False
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Cross-environment transfer test.
        
        Args:
            source_agent: Pre-trained agent
            source_adapter: Source environment adapter (already trained)
            target_adapter_class: Target environment adapter class
            target_level: Level in target environment
            source_episodes: Episodes to train in source (if not frozen)
            target_episodes: Episodes to evaluate in target
            freeze_source: If True, don't train in source, just transfer
            
        Returns:
            (source_result, target_result)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TRANSFER TEST")
            print(f"Source → Target (Level {target_level})")
            print(f"{'='*60}\n")
        
        # Source training (if not frozen)
        source_scores = []
        if not freeze_source:
            if self.verbose:
                print("Training in source environment...")
            
            for ep in range(source_episodes):
                score = self._run_episode(source_agent, source_adapter, explore=True)
                source_scores.append(score)
                
                if self.verbose and (ep + 1) % 20 == 0:
                    recent = source_scores[-20:]
                    print(f"  {ep+1:3d}/{source_episodes}: avg={np.mean(recent):.1f}")
        
        source_result = BenchmarkResult(
            test_type='transfer_source',
            mean_score=np.mean(source_scores) if source_scores else 0,
            std_score=np.std(source_scores) if source_scores else 0,
            best_score=max(source_scores) if source_scores else 0,
            worst_score=min(source_scores) if source_scores else 0,
            learning_curve=source_scores,
            metadata={'frozen': freeze_source}
        )
        
        # Target evaluation (frozen weights)
        if self.verbose:
            print(f"\nTransferring to target environment (Level {target_level})...")
        
        target_adapter = target_adapter_class(level=target_level)
        
        # Check feature compatibility
        if target_adapter.n_features != source_agent.n_features:
            if self.verbose:
                print(f"  WARNING: Feature mismatch! "
                      f"Source={source_agent.n_features}, "
                      f"Target={target_adapter.n_features}")
                print(f"  Transfer may fail or perform poorly.")
        
        # Freeze agent for evaluation
        original_epsilon = source_agent.epsilon
        source_agent.epsilon = 0.0
        
        target_scores = []
        for ep in range(target_episodes):
            score = self._run_episode(source_agent, target_adapter, explore=False)
            target_scores.append(score)
        
        source_agent.epsilon = original_epsilon
        
        target_result = BenchmarkResult(
            test_type='transfer_target',
            mean_score=np.mean(target_scores),
            std_score=np.std(target_scores),
            best_score=max(target_scores),
            worst_score=min(target_scores),
            learning_curve=target_scores,
            metadata={
                'target_level': target_level,
                'episodes': target_episodes
            }
        )
        
        if self.verbose:
            print(f"\n  Target Results:")
            print(f"    Mean:  {target_result.mean_score:.2f} ± {target_result.std_score:.2f}")
            print(f"    Best:  {target_result.best_score:.0f}\n")
        
        self.results.extend([source_result, target_result])
        return source_result, target_result
    
    def _run_episode(self, agent, adapter, explore: bool = True) -> float:
        """
        Run a single episode.
        
        Args:
            agent: PortableNNAgent instance
            adapter: Environment adapter
            explore: Whether to use epsilon-greedy
            
        Returns:
            Episode score (lines cleared or similar metric)
        """
        agent.reset_episode()
        adapter.reset()
        
        total_reward = 0.0
        done = False
        
        while not done:
            valid_actions = adapter.get_valid_actions()
            if not valid_actions:
                break
            
            # Select action
            action = agent.select_action(
                valid_actions=valid_actions,
                feature_fn=adapter.make_features,
                explore=explore
            )
            
            # Record features for this action
            features = adapter.make_features(action)
            
            # Take step
            _, reward, done, info = adapter.step(action)
            total_reward += reward
            
            # Record for training
            agent.record_step(features, reward)
        
        # End episode (triggers training if explore=True)
        final_score = info.get('lines_cleared', adapter.env.lines_cleared)
        if explore:
            agent.end_episode(final_score)
        
        return final_score
    
    def export_results(self, path: str):
        """
        Export all results to JSON file.
        
        Args:
            path: Output file path
        """
        data = {
            'timestamp': time.time(),
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"Results exported to {path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all tests."""
        if not self.results:
            return {}
        
        by_type = {}
        for result in self.results:
            test_type = result.test_type
            if test_type not in by_type:
                by_type[test_type] = []
            by_type[test_type].append(result)
        
        summary = {}
        for test_type, results in by_type.items():
            summary[test_type] = {
                'count': len(results),
                'mean_score': np.mean([r.mean_score for r in results]),
                'best_overall': max(r.best_score for r in results)
            }
        
        return summary
