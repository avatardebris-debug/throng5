"""
Wide Multi-Game Cumulative Test

Tests the full Meta^N stack across many games with fewer episodes each.
The goal: does the system accumulate value over time?

Strategy: Go wide (many games, 100 episodes each) to detect cumulative patterns.
Can go deep later (more episodes per game) if needed.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from throng4.metastack_pipeline import MetaStackPipeline
from throng4.layers.meta4_goal_hierarchy import GoalHierarchy, GoalHierarchyConfig
from throng4.llm_policy.concept_library import ConceptLibrary, ConceptTransfer
from throng4.environments.atari_adapter import AtariAdapter


# Games to test (ordered by expected difficulty)
GAMES = [
    'Breakout',         # Paddle + ball
    'Pong',             # Paddle vs opponent
    'SpaceInvaders',    # Shooting + dodging
    'Freeway',          # Cross traffic
    'Skiing',           # Dodge obstacles
    'Boxing',           # Combat
]

EPISODES_PER_GAME = 100  # Wide, not deep


class CumulativeExperiment:
    """
    Run the full Meta^N stack across multiple games sequentially.
    Tracks cumulative learning metrics.
    """
    
    def __init__(self, games: List[str] = None, episodes: int = EPISODES_PER_GAME):
        self.games = games or GAMES
        self.episodes = episodes
        self.goal_hierarchy = GoalHierarchy(GoalHierarchyConfig(
            implicit_only_episodes=20,  # Shorter for wide test
            plateau_window=15,
            library_update_interval=50,
        ))
        self.concept_library = ConceptLibrary()
        self.concept_transfer = ConceptTransfer(self.concept_library)
        
        # Track across all games
        self.all_results = {}
        self.prev_pipeline = None
        self.cumulative_metrics = {
            'games': [],
            'avg_rewards': [],
            'best_rewards': [],
            'convergence_episodes': [],
            'concepts_used': [],
            'modes_used': [],
        }
    
    def run_single_game(self, game: str) -> Dict[str, Any]:
        """Run one game with the full stack, sharing state from previous."""
        print(f"\n{'='*60}")
        print(f"GAME: {game} ({self.episodes} episodes)")
        print(f"{'='*60}")
        
        # Create environment
        try:
            env = AtariAdapter(game)
            info = env.get_info()
            n_inputs = env.n_features
            n_outputs = info['n_actions']
        except Exception as e:
            print(f"  ❌ Environment failed: {e}")
            return {'game': game, 'status': 'env_error', 'error': str(e)}
        
        # Create pipeline (transfer weights from previous game if available)
        pipeline = MetaStackPipeline(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_hidden=128
        )
        
        if self.prev_pipeline is not None:
            try:
                pipeline.transfer_weights(self.prev_pipeline)
                print(f"  ✅ Weights transferred from previous game")
            except Exception as e:
                print(f"  ⚠️  Weight transfer failed (dimension mismatch): {e}")
        
        # Meta^4: Start game
        self.goal_hierarchy.start_game(game)
        self.goal_hierarchy.state.concept_library_size = len(
            self.concept_library.get_all_concepts()
        )
        
        # Get applicable concepts
        concept_ids = self.concept_transfer.get_applicable_concepts_heuristic(
            'tetris', game.lower()
        )
        concepts = []
        for cid in concept_ids:
            c = self.concept_library.get_concept(cid)
            if c:
                c['name'] = cid
                concepts.append(c)
        
        print(f"  Concepts: {len(concepts)} applicable")
        
        # Run episodes
        episode_rewards = []
        modes_used = set()
        
        for ep in range(self.episodes):
            self.goal_hierarchy.start_episode()
            mode = self.goal_hierarchy.current_mode
            modes_used.add(mode)
            
            # Set exploration from Meta^4
            pipeline.epsilon = self.goal_hierarchy.get_exploration_rate()
            
            # Apply concept bias once
            if mode in ['library', 'hybrid'] and ep == 20 and concepts:
                self._apply_concept_bias(pipeline, concepts)
                print(f"  [Ep {ep}] Applied {len(concepts)} concepts (mode={mode})")
            
            # Run episode
            state = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < 1000:
                action = pipeline.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)
                pipeline.update(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                steps += 1
            
            self.goal_hierarchy.end_episode(episode_reward)
            episode_rewards.append(episode_reward)
            
            # Progress every 25 episodes
            if (ep + 1) % 25 == 0:
                avg = np.mean(episode_rewards[-25:])
                print(f"  Ep {ep+1}/{self.episodes}: "
                      f"avg={avg:.1f}, mode={mode}, "
                      f"ε={pipeline.epsilon:.3f}")
        
        # Cleanup
        env.close()
        
        # End game in Meta^4
        avg_reward = np.mean(episode_rewards[-25:]) if episode_rewards else 0
        best_reward = max(episode_rewards) if episode_rewards else 0
        self.goal_hierarchy.end_game()
        
        # Save pipeline for next game
        self.prev_pipeline = pipeline
        
        result = {
            'game': game,
            'status': 'completed',
            'episodes': self.episodes,
            'avg_reward': float(avg_reward),
            'best_reward': float(best_reward),
            'all_rewards': [float(r) for r in episode_rewards],
            'modes_used': list(modes_used),
            'concepts_applied': len(concepts),
        }
        
        print(f"\n  Result: avg={avg_reward:.1f}, best={best_reward:.1f}")
        return result
    
    def _apply_concept_bias(self, pipeline, concepts):
        """Apply concept biases to pipeline weights."""
        weights = pipeline.ann.get_weights()
        for concept in concepts:
            confidence = concept.get('transfer_potential', 0.5)
            bias = confidence * 0.05  # Subtle bias
            weights['W1'] += np.random.randn(*weights['W1'].shape) * bias * 0.01
        pipeline.ann.set_weights(weights)
    
    def run_all(self):
        """Run all games sequentially, tracking cumulative learning."""
        print("=" * 70)
        print("WIDE MULTI-GAME CUMULATIVE TEST")
        print("=" * 70)
        print(f"Games: {len(self.games)}")
        print(f"Episodes per game: {self.episodes}")
        print(f"Stack: Meta^0 → Meta^1 → Meta^3 → Meta^4 → Meta^N")
        print("=" * 70)
        
        start_time = time.time()
        
        for i, game in enumerate(self.games):
            print(f"\n{'#'*60}")
            print(f"# GAME {i+1}/{len(self.games)}: {game}")
            print(f"{'#'*60}")
            
            result = self.run_single_game(game)
            self.all_results[game] = result
            
            if result['status'] == 'completed':
                self.cumulative_metrics['games'].append(game)
                self.cumulative_metrics['avg_rewards'].append(result['avg_reward'])
                self.cumulative_metrics['best_rewards'].append(result['best_reward'])
                self.cumulative_metrics['concepts_used'].append(result['concepts_applied'])
                self.cumulative_metrics['modes_used'].append(result['modes_used'])
        
        elapsed = time.time() - start_time
        
        # Print cumulative report
        self._print_report(elapsed)
        
        # Save results
        self._save_results()
        
        return self.all_results
    
    def _print_report(self, elapsed: float):
        """Print the cumulative learning report."""
        print("\n" + "=" * 70)
        print("CUMULATIVE LEARNING REPORT")
        print("=" * 70)
        print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Games tested: {len(self.cumulative_metrics['games'])}")
        
        print("\n--- Per-Game Results ---")
        print(f"{'Game':<20} {'Avg Reward':>10} {'Best':>8} {'Concepts':>8}")
        print("-" * 50)
        
        for i, game in enumerate(self.cumulative_metrics['games']):
            avg = self.cumulative_metrics['avg_rewards'][i]
            best = self.cumulative_metrics['best_rewards'][i]
            concepts = self.cumulative_metrics['concepts_used'][i]
            print(f"{game:<20} {avg:>10.1f} {best:>8.1f} {concepts:>8}")
        
        # Cumulative trend
        if len(self.cumulative_metrics['avg_rewards']) >= 2:
            print("\n--- Cumulative Trend ---")
            rewards = self.cumulative_metrics['avg_rewards']
            
            # Is performance improving across games?
            first_half = np.mean(rewards[:len(rewards)//2])
            second_half = np.mean(rewards[len(rewards)//2:])
            
            if first_half > 0:
                change = (second_half - first_half) / first_half * 100
                trend = "📈 IMPROVING" if change > 5 else "📉 DECLINING" if change < -5 else "➡️ STABLE"
                print(f"  First half avg: {first_half:.1f}")
                print(f"  Second half avg: {second_half:.1f}")
                print(f"  Change: {change:+.1f}%")
                print(f"  Trend: {trend}")
        
        # Meta^4 report  
        print(f"\n{self.goal_hierarchy.get_cumulative_report()}")
    
    def _save_results(self):
        """Save all results to JSON."""
        output_dir = Path('experiments/wide_multigame')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'games': self.cumulative_metrics['games'],
            'avg_rewards': self.cumulative_metrics['avg_rewards'],
            'best_rewards': self.cumulative_metrics['best_rewards'],
            'per_game': {
                game: {
                    'avg_reward': r.get('avg_reward', 0),
                    'best_reward': r.get('best_reward', 0),
                    'status': r.get('status', 'unknown'),
                }
                for game, r in self.all_results.items()
            }
        }
        
        with open(output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    experiment = CumulativeExperiment()
    experiment.run_all()
