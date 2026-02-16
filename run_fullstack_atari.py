"""
Full-Stack Atari Runner — Uses REAL Meta^N Architecture

Integrates:
- Meta^0 (ANNLayer): Dual-head network
- Meta^1 (SynapseOptimizer): Per-head LR optimization
- Meta^3 (DualHeadMAML): Weight initialization + LR multipliers
- Meta^4 (GoalHierarchy): Coordinates implicit/explicit learning
- Meta^N (ConceptLibrary + Tetra): Linguistic concept layer

This is the REAL test of "learning to learn to learn".
Unlike the simple PortableNNAgent baselines, this uses the full stack.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from throng4.metastack_pipeline import MetaStackPipeline
from throng4.layers.meta4_goal_hierarchy import GoalHierarchy, GoalHierarchyConfig, LearningMode
from throng4.llm_policy.concept_library import ConceptLibrary, ConceptTransfer
from throng4.environments.atari_adapter import AtariAdapter


class FullStackAtariRunner:
    """
    Runs Atari games through the full Meta^N stack.
    
    This is the "ceiling" test — the best our architecture can do.
    Compare against flat PortableNNAgent baselines for the "floor".
    """
    
    def __init__(self, 
                 game: str = 'Breakout',
                 n_hidden: int = 128,
                 goal_config: Optional[GoalHierarchyConfig] = None):
        
        self.game = game
        
        # Environment
        self.env = AtariAdapter(game)
        info = self.env.get_info()
        n_inputs = self.env.n_features
        n_outputs = info['n_actions']
        
        # Full Meta^N Stack
        self.pipeline = MetaStackPipeline(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=n_hidden
        )
        
        # Meta^4: Goal Hierarchy
        self.goals = GoalHierarchy(goal_config or GoalHierarchyConfig())
        
        # Meta^N: Concept Library
        self.concept_library = ConceptLibrary()
        self.concept_transfer = ConceptTransfer(self.concept_library)
        
        # Performance tracking
        self.episode_rewards = []
        self.concept_events = []
    
    def apply_concept_bias(self, concepts):
        """Apply concept-based weight biases to the pipeline's ANN."""
        weights = self.pipeline.ann.get_weights()
        
        for concept in concepts:
            name = concept.get('name', '')
            confidence = concept.get('transfer_potential', 0.5)
            bias_strength = confidence * 0.1  # Subtle bias
            
            if 'danger' in name or 'avoid' in name:
                # Bias toward cautious play
                weights['W_q'] *= (1.0 + bias_strength * 0.5)
            elif 'target' in name or 'completion' in name:
                # Bias toward goal-directed behavior
                weights['W_q'][:, 0] += bias_strength  # Bias action 0
            elif 'patience' in name:
                # Reduce impulsive actions
                weights['b_q'] *= (1.0 - bias_strength * 0.3)
            
            self.concept_events.append({
                'episode': self.goals.state.game_episodes,
                'concept': name,
                'confidence': confidence,
                'action': 'applied'
            })
        
        self.pipeline.ann.set_weights(weights)
        self.pipeline._sync_target_network()
    
    def run_game(self, 
                 n_episodes: int = 500,
                 source_pipeline: Optional[MetaStackPipeline] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        """
        Run a complete game through the full stack.
        
        Args:
            n_episodes: Number of episodes
            source_pipeline: Previous game's pipeline (for weight transfer)
            verbose: Print progress
        """
        # Start game in Meta^4
        self.goals.start_game(self.game)
        
        # Transfer weights from previous game (if available)
        if source_pipeline is not None:
            self.pipeline.transfer_weights(source_pipeline)
            if verbose:
                print(f"  [Transfer] Weights transferred from previous game")
        
        # Get applicable concepts
        concept_ids = self.concept_transfer.get_applicable_concepts_heuristic(
            'tetris', self.game.lower()
        )
        concepts = []
        for cid in concept_ids:
            c = self.concept_library.get_concept(cid)
            if c:
                c['name'] = cid
                concepts.append(c)
        
        self.goals.state.concept_library_size = len(
            self.concept_library.get_all_concepts()
        )
        
        if verbose:
            print(f"  [Concepts] {len(concepts)} applicable concepts found")
        
        for ep in range(n_episodes):
            # Meta^4 decides learning mode for this episode
            self.goals.start_episode()
            mode = self.goals.current_mode
            
            # Apply concepts based on mode
            if mode in [LearningMode.EXPLICIT_LIBRARY, LearningMode.HYBRID]:
                if ep == self.goals.config.implicit_only_episodes:
                    # First time using concepts
                    self.apply_concept_bias(concepts)
                    if verbose:
                        print(f"  [Meta^4] Ep {ep}: Applying {len(concepts)} concepts")
            
            # Get exploration rate from Meta^4
            epsilon = self.goals.get_exploration_rate()
            self.pipeline.epsilon = epsilon
            
            # Run episode
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Meta^0: Select action
                action = self.pipeline.select_action(state, explore=True)
                
                # Environment step
                next_state, reward, done, info = self.env.step(action)
                
                # Meta^0 → Meta^1 → Meta^3: Update
                self.pipeline.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
            
            # Meta^4: End episode, update goals
            self.goals.end_episode(episode_reward)
            self.episode_rewards.append(episode_reward)
            
            # Check if Meta^4 says to update library
            if self.goals.should_update_library():
                # Track which concepts helped
                avg_before = np.mean(self.episode_rewards[max(0, len(self.episode_rewards)-200):max(0, len(self.episode_rewards)-100)])
                avg_after = np.mean(self.episode_rewards[max(0, len(self.episode_rewards)-100):])
                
                for concept in concepts:
                    helped = avg_after > avg_before * 1.05  # 5% improvement
                    self.goals.record_concept_result(concept.get('name', ''), helped)
            
            # Verbose logging
            if verbose and (ep + 1) % 50 == 0:
                stats = self.goals.get_stats()
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"  Ep {ep+1}/{n_episodes}: "
                      f"mode={stats['current_mode']}, "
                      f"avg_reward={avg_reward:.1f}, "
                      f"ε={stats['exploration_rate']:.3f}, "
                      f"plateau={stats['is_plateauing']}")
        
        # End game in Meta^4
        # Detect convergence (when improvement < 5% over 50 episodes)
        convergence_ep = self._detect_convergence()
        self.goals.end_game(convergence_episode=convergence_ep)
        
        # Results
        return {
            'game': self.game,
            'episodes': n_episodes,
            'rewards': self.episode_rewards,
            'final_avg': np.mean(self.episode_rewards[-50:]),
            'best_reward': max(self.episode_rewards),
            'convergence_episode': convergence_ep,
            'concept_events': self.concept_events,
            'goal_stats': self.goals.get_stats(),
            'pipeline_stats': self.pipeline.get_stats(),
            'cumulative_report': self.goals.get_cumulative_report(),
        }
    
    def _detect_convergence(self, window: int = 50, threshold: float = 0.05) -> Optional[int]:
        """Detect when performance converged."""
        if len(self.episode_rewards) < window * 2:
            return None
        
        for i in range(window, len(self.episode_rewards) - window):
            before = np.mean(self.episode_rewards[i-window:i])
            after = np.mean(self.episode_rewards[i:i+window])
            if before > 0 and abs(after - before) / before < threshold:
                return i
        
        return None


def run_full_stack_experiment(games: list = None, episodes_per_game: int = 500):
    """
    Run cumulative learning experiment across multiple games.
    
    This is the REAL test: does the system accumulate value over time?
    """
    if games is None:
        games = ['Breakout', 'Pong']  # Start with 2 games
    
    results = {}
    prev_pipeline = None
    goal_config = GoalHierarchyConfig()
    
    print("=" * 70)
    print("FULL META^N STACK EXPERIMENT")
    print("=" * 70)
    print(f"Games: {games}")
    print(f"Episodes per game: {episodes_per_game}")
    print(f"Stack: Meta^0 → Meta^1 → Meta^3 → Meta^4 → Meta^N")
    print("=" * 70)
    
    for i, game in enumerate(games):
        print(f"\n{'='*70}")
        print(f"GAME {i+1}/{len(games)}: {game}")
        print(f"{'='*70}")
        
        runner = FullStackAtariRunner(
            game=game,
            goal_config=goal_config
        )
        
        result = runner.run_game(
            n_episodes=episodes_per_game,
            source_pipeline=prev_pipeline,
            verbose=True
        )
        
        results[game] = result
        prev_pipeline = runner.pipeline
        
        print(f"\n{result['cumulative_report']}")
    
    # Save results
    output_dir = Path('experiments/full_stack')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'games': games,
        'episodes_per_game': episodes_per_game,
        'results': {
            game: {
                'final_avg': r['final_avg'],
                'best_reward': r['best_reward'],
                'convergence_episode': r['convergence_episode'],
            }
            for game, r in results.items()
        }
    }
    
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == "__main__":
    # Quick test: single game with full stack
    print("Quick test: Full stack on Breakout (100 episodes)")
    
    runner = FullStackAtariRunner('Breakout')
    result = runner.run_game(n_episodes=100, verbose=True)
    
    print(f"\nFinal avg reward: {result['final_avg']:.1f}")
    print(f"Best reward: {result['best_reward']:.1f}")
    print(f"Convergence: {result['convergence_episode']}")
