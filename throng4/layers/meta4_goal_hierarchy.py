"""
Meta^4: GoalHierarchy — Coordinates all learning systems.

Manages short → medium → long term reward horizons and decides
when to use implicit learning (Meta^0-3) vs explicit learning
(Meta^N concepts/LLM).

Architecture:
    Meta^N (LLM/Tetra) ←→ Meta^4 (GoalHierarchy) → Meta^3 (MAML)
                                                   → Meta^1 (Synapse)
                                                   → Meta^0 (ANN)

Design Philosophy:
    - Short-term: Maximize episode reward (Meta^0-1 handle this)
    - Medium-term: Build concept library (Meta^N handles this)
    - Long-term: Accelerate learning on NEW games (Meta^3-4 handle this)
    
    Meta^4 decides the MIX between these horizons at each step.
"""

import numpy as np
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


@dataclass
class GoalHierarchyConfig:
    """Configuration for Meta^4 GoalHierarchy."""
    # Exploration
    exploration_decay: float = 0.999     # Per-episode exploration decay
    min_exploration: float = 0.05        # Never go below this
    initial_exploration: float = 1.0     # Start fully exploring
    
    # Mode selection thresholds
    implicit_only_episodes: int = 50     # Use only MAML for first N episodes
    llm_query_cooldown: int = 25         # Min episodes between LLM queries
    concept_confidence_threshold: float = 0.7  # Min confidence to use concept
    
    # Performance tracking
    plateau_window: int = 30             # Window for detecting plateaus
    plateau_threshold: float = 0.05      # Max improvement to count as plateau
    
    # Concept library
    library_update_interval: int = 100   # Episodes between library updates
    max_failed_concepts: int = 3         # Archive after N failures
    
    # Goal horizon weights (sum to 1.0)
    short_term_weight: float = 0.5       # Episode reward
    medium_term_weight: float = 0.3      # Concept quality
    long_term_weight: float = 0.2        # Transfer speedup


class LearningMode:
    """Enum-like for learning modes."""
    IMPLICIT = 'implicit'           # Meta^0-3 only (MAML/Synapse/ANN)
    EXPLICIT_LIBRARY = 'library'    # Concept library bias
    EXPLICIT_LLM = 'llm'           # Query LLM for guidance
    HYBRID = 'hybrid'              # Both implicit + explicit


@dataclass
class GoalState:
    """Tracks goals across time horizons."""
    # Short-term (within episode)
    episode_reward: float = 0.0
    episode_steps: int = 0
    best_episode_reward: float = float('-inf')
    
    # Medium-term (across episodes in one game)
    game_episodes: int = 0
    running_avg_reward: float = 0.0
    reward_history: list = field(default_factory=list)
    concepts_applied: list = field(default_factory=list)
    concept_hit_count: int = 0
    concept_miss_count: int = 0
    
    # Long-term (across games)
    games_completed: int = 0
    episodes_to_converge: dict = field(default_factory=dict)
    concept_library_size: int = 0
    meta_concept_count: int = 0
    transfer_speedups: list = field(default_factory=list)


class GoalHierarchy:
    """
    Meta^4: Manages learning goals across time horizons.
    
    Decides:
    1. Which learning mode to use (implicit vs explicit)
    2. When to explore vs exploit
    3. When to query LLM vs use library
    4. When to update concept library
    5. How to balance short/medium/long term goals
    """
    
    def __init__(self, config: Optional[GoalHierarchyConfig] = None):
        self.config = config or GoalHierarchyConfig()
        
        # State
        self.state = GoalState()
        self.current_game: str = ''
        self.current_mode: str = LearningMode.IMPLICIT
        
        # Exploration
        self.exploration_rate = self.config.initial_exploration
        
        # Performance tracking
        self.performance_window = deque(maxlen=self.config.plateau_window)
        self.mode_history = deque(maxlen=1000)
        
        # LLM query tracking
        self.last_llm_query_episode = -self.config.llm_query_cooldown
        self.llm_query_results: List[Dict] = []
        
        # Concept performance tracking
        self.concept_performance: Dict[str, Dict] = {}
        
        # Stats
        self.total_mode_switches = 0
        self.total_llm_queries = 0
    
    def start_game(self, game_name: str):
        """Begin a new game. Reset medium-term goals."""
        self.current_game = game_name
        self.state.game_episodes = 0
        self.state.running_avg_reward = 0.0
        self.state.reward_history = []
        self.state.concepts_applied = []
        self.state.concept_hit_count = 0
        self.state.concept_miss_count = 0
        self.exploration_rate = self.config.initial_exploration
        self.performance_window.clear()
        
        print(f"\n[Meta^4] Starting game: {game_name}")
        print(f"  Exploration: {self.exploration_rate:.2f}")
        print(f"  Games completed: {self.state.games_completed}")
        print(f"  Library size: {self.state.concept_library_size}")
    
    def end_game(self, convergence_episode: Optional[int] = None):
        """End current game. Update long-term goals."""
        if convergence_episode is not None:
            self.state.episodes_to_converge[self.current_game] = convergence_episode
        
        self.state.games_completed += 1
        
        # Calculate transfer speedup (only if we have reference data)
        if self.state.games_completed > 1 and self.state.episodes_to_converge:
            first_game_eps = list(self.state.episodes_to_converge.values())[0]
            current = convergence_episode or len(self.state.reward_history)
            if first_game_eps > 0:
                speedup = 1.0 - (current / max(first_game_eps, 1))
                self.state.transfer_speedups.append(speedup)
                print(f"[Meta^4] Transfer speedup: {speedup:.1%}")
        
        print(f"[Meta^4] Game {self.current_game} complete. "
              f"Episodes: {self.state.game_episodes}, "
              f"Best reward: {self.state.best_episode_reward:.1f}")

    
    def start_episode(self):
        """Begin a new episode. Update mode selection."""
        self.state.game_episodes += 1
        self.state.episode_reward = 0.0
        self.state.episode_steps = 0
        
        # Decay exploration
        self.exploration_rate = max(
            self.config.min_exploration,
            self.exploration_rate * self.config.exploration_decay
        )
        
        # Select learning mode for this episode
        self.current_mode = self._select_learning_mode()
    
    def end_episode(self, total_reward: float):
        """End episode. Update all goal horizons."""
        self.state.episode_reward = total_reward
        self.state.reward_history.append(total_reward)
        self.performance_window.append(total_reward)
        
        # Update best
        if total_reward > self.state.best_episode_reward:
            self.state.best_episode_reward = total_reward
        
        # Update running average
        alpha = 0.1
        self.state.running_avg_reward = (
            alpha * total_reward + 
            (1 - alpha) * self.state.running_avg_reward
        )
        
        # Track mode usage
        self.mode_history.append(self.current_mode)
    
    def _select_learning_mode(self) -> str:
        """
        Core decision: which learning mode to use this episode.
        
        Decision tree:
        1. Early episodes → IMPLICIT (let MAML/ANN explore freely)
        2. Plateau detected → query LLM (get unstuck)
        3. High-confidence concepts → LIBRARY (exploit knowledge)
        4. Default → HYBRID (use everything)
        """
        ep = self.state.game_episodes
        
        # 1. Early episodes: implicit only (let MAML explore)
        if ep <= self.config.implicit_only_episodes:
            return LearningMode.IMPLICIT
        
        # 2. Check for plateau → query LLM
        if self._is_plateauing():
            if self._can_query_llm():
                self.total_llm_queries += 1
                self.last_llm_query_episode = ep
                self._switch_mode(LearningMode.EXPLICIT_LLM)
                return LearningMode.EXPLICIT_LLM
        
        # 3. Check concept library for high-confidence matches
        if self.state.concept_library_size > 0:
            hit_rate = self._get_concept_hit_rate()
            if hit_rate > self.config.concept_confidence_threshold:
                return LearningMode.EXPLICIT_LIBRARY
        
        # 4. Default: hybrid (use everything available)
        if self.state.concept_library_size > 0:
            return LearningMode.HYBRID
        
        return LearningMode.IMPLICIT
    
    def _is_plateauing(self) -> bool:
        """Detect performance plateau."""
        if len(self.performance_window) < self.config.plateau_window:
            return False
        
        perf = list(self.performance_window)
        first_half = np.mean(perf[:len(perf)//2])
        second_half = np.mean(perf[len(perf)//2:])
        
        if first_half == 0:
            return False
        
        improvement = abs(second_half - first_half) / max(abs(first_half), 1e-8)
        return improvement < self.config.plateau_threshold
    
    def _can_query_llm(self) -> bool:
        """Check if LLM query cooldown has expired."""
        return (self.state.game_episodes - self.last_llm_query_episode 
                >= self.config.llm_query_cooldown)
    
    def _get_concept_hit_rate(self) -> float:
        """Get success rate of applied concepts."""
        total = self.state.concept_hit_count + self.state.concept_miss_count
        if total == 0:
            return 0.0
        return self.state.concept_hit_count / total
    
    def _switch_mode(self, new_mode: str):
        """Record mode switch."""
        if new_mode != self.current_mode:
            self.total_mode_switches += 1
    
    def record_concept_result(self, concept_id: str, helped: bool):
        """Track whether a concept helped or hurt."""
        if concept_id not in self.concept_performance:
            self.concept_performance[concept_id] = {
                'hits': 0, 'misses': 0, 'games': set()
            }
        
        tracking = self.concept_performance[concept_id]
        tracking['games'].add(self.current_game)
        
        if helped:
            tracking['hits'] += 1
            self.state.concept_hit_count += 1
        else:
            tracking['misses'] += 1
            self.state.concept_miss_count += 1
    
    def should_update_library(self) -> bool:
        """Check if concept library should be updated."""
        return (self.state.game_episodes > 0 and 
                self.state.game_episodes % self.config.library_update_interval == 0)
    
    def get_concepts_to_archive(self) -> List[str]:
        """Get concepts that should be archived (too many failures)."""
        to_archive = []
        for concept_id, perf in self.concept_performance.items():
            if perf['misses'] >= self.config.max_failed_concepts:
                if perf['hits'] < perf['misses']:
                    to_archive.append(concept_id)
        return to_archive
    
    def get_exploration_rate(self) -> float:
        """Get current exploration rate (for epsilon-greedy)."""
        # Mode-dependent exploration
        if self.current_mode == LearningMode.IMPLICIT:
            return self.exploration_rate  # Standard decay
        elif self.current_mode == LearningMode.EXPLICIT_LIBRARY:
            return self.exploration_rate * 0.5  # Less exploration with concepts
        elif self.current_mode == LearningMode.EXPLICIT_LLM:
            return self.exploration_rate * 0.3  # Much less with LLM guidance
        else:  # HYBRID
            return self.exploration_rate * 0.7
    
    def get_goal_weights(self) -> Dict[str, float]:
        """
        Get current goal horizon weights.
        
        Adapts weights based on:
        - Early → high short-term (learn basics first)
        - Mid → high medium-term (build concepts)
        - Late → high long-term (optimize transfer)
        """
        ep = self.state.game_episodes
        
        if ep < 100:
            # Early: focus on episode performance
            return {
                'short': 0.7,
                'medium': 0.2,
                'long': 0.1
            }
        elif ep < 300:
            # Mid: balance episode + concept building
            return {
                'short': 0.4,
                'medium': 0.4,
                'long': 0.2
            }
        else:
            # Late: focus on transfer value
            return {
                'short': 0.3,
                'medium': 0.3,
                'long': 0.4
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats."""
        return {
            'current_game': self.current_game,
            'current_mode': self.current_mode,
            'episode': self.state.game_episodes,
            'exploration_rate': self.exploration_rate,
            'running_avg_reward': self.state.running_avg_reward,
            'best_reward': self.state.best_episode_reward,
            'games_completed': self.state.games_completed,
            'concept_hit_rate': self._get_concept_hit_rate(),
            'library_size': self.state.concept_library_size,
            'total_llm_queries': self.total_llm_queries,
            'total_mode_switches': self.total_mode_switches,
            'transfer_speedups': self.state.transfer_speedups,
            'goal_weights': self.get_goal_weights(),
            'is_plateauing': self._is_plateauing(),
        }
    
    def get_cumulative_report(self) -> str:
        """Generate human-readable cumulative learning report."""
        report = []
        report.append("=" * 60)
        report.append("META^4 CUMULATIVE LEARNING REPORT")
        report.append("=" * 60)
        report.append(f"Games completed: {self.state.games_completed}")
        report.append(f"Concept library: {self.state.concept_library_size} concepts")
        report.append(f"Concept hit rate: {self._get_concept_hit_rate():.1%}")
        report.append(f"LLM queries used: {self.total_llm_queries}")
        report.append(f"Mode switches: {self.total_mode_switches}")
        
        if self.state.episodes_to_converge:
            report.append("\nConvergence by game:")
            for game, eps in self.state.episodes_to_converge.items():
                report.append(f"  {game}: {eps} episodes")
        
        if self.state.transfer_speedups:
            report.append(f"\nTransfer speedups: {self.state.transfer_speedups}")
            report.append(f"Mean speedup: {np.mean(self.state.transfer_speedups):.1%}")
            
            # Is speedup accelerating? (the META meta-learning test)
            if len(self.state.transfer_speedups) >= 3:
                diffs = np.diff(self.state.transfer_speedups)
                if np.mean(diffs) > 0:
                    report.append("✅ SPEEDUP IS ACCELERATING! True meta-learning detected.")
                else:
                    report.append("⚠️  Speedup is plateauing. May need architectural changes.")
        
        report.append("=" * 60)
        return "\n".join(report)


if __name__ == "__main__":
    # Test GoalHierarchy
    print("Testing Meta^4 GoalHierarchy...")
    
    gh = GoalHierarchy()
    
    # Simulate Game 1: Tetris
    gh.start_game('Tetris')
    
    for ep in range(200):
        gh.start_episode()
        
        # Simulate reward
        reward = np.random.exponential(2.0) + (ep * 0.05)
        gh.end_episode(reward)
        
        if (ep + 1) % 50 == 0:
            stats = gh.get_stats()
            print(f"  Ep {ep+1}: mode={stats['current_mode']}, "
                  f"avg_reward={stats['running_avg_reward']:.1f}, "
                  f"exploration={stats['exploration_rate']:.3f}")
    
    gh.end_game(convergence_episode=150)
    
    # Simulate Game 2: Breakout (with concepts)
    gh.state.concept_library_size = 5
    gh.start_game('Breakout')
    
    for ep in range(150):
        gh.start_episode()
        
        # Simulate better reward (concepts help)
        reward = np.random.exponential(3.0) + (ep * 0.08)
        gh.end_episode(reward)
        
        # Simulate concept hits
        if ep > 50:
            gh.record_concept_result('avoid_danger_spatial', helped=True)
        
        if (ep + 1) % 50 == 0:
            stats = gh.get_stats()
            print(f"  Ep {ep+1}: mode={stats['current_mode']}, "
                  f"avg_reward={stats['running_avg_reward']:.1f}, "
                  f"concept_hit_rate={stats['concept_hit_rate']:.1%}")
    
    gh.end_game(convergence_episode=100)
    
    # Print cumulative report
    print(gh.get_cumulative_report())
