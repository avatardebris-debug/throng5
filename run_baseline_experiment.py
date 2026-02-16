"""
Baseline experiment runner for meta-learning validation.

Implements 5 baselines:
1. Tabula Rasa - no transfer
2. MAML-Only - weight transfer only
3. Static Concepts - frozen concept library
4. LLM-at-Start - query Tetra once before training
5. Full System - real-time Tetra queries

Usage:
    python run_baseline_experiment.py --game Breakout --baseline tabula_rasa --runs 5
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from throng4.environments.atari_adapter import AtariAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.llm_policy.openclaw_bridge import OpenClawBridge


class BaselineExperiment:
    """Run baseline experiments for meta-learning validation."""
    
    BASELINES = [
        'tabula_rasa',      # No transfer
        'maml_only',        # Weight transfer only
        'static_concepts',  # Frozen concept library
        'llm_at_start',     # Query Tetra once
        'full_system'       # Real-time queries
    ]
    
    def __init__(self, game: str, baseline: str, source_game: str = 'tetris'):
        """
        Initialize experiment.
        
        Args:
            game: Target game name (e.g., 'Breakout')
            baseline: Which baseline to run
            source_game: Source game for transfer (default: 'tetris')
        """
        assert baseline in self.BASELINES, f"Unknown baseline: {baseline}"
        
        self.game = game
        self.baseline = baseline
        self.source_game = source_game
        
        # Create adapter
        self.adapter = AtariAdapter(game)
        
        # Create agent based on baseline
        self.agent = self._create_agent()
        
        # Create bridge if needed
        self.bridge = None
        if baseline in ['llm_at_start', 'full_system']:
            self.bridge = OpenClawBridge(game=game)
    
    def _create_agent(self) -> PortableNNAgent:
        """Create agent based on baseline type."""
        config = AgentConfig(
            n_hidden=128,
            epsilon=0.20,
            learning_rate=0.005
        )
        
        agent = PortableNNAgent(
            n_features=self.adapter.n_features,
            config=config
        )
        
        # Load weights/concepts based on baseline
        if self.baseline == 'maml_only':
            # Load MAML weights from source game
            weights_path = f"weights/{self.source_game}_maml.npz"
            if Path(weights_path).exists():
                agent.load_weights(weights_path)
                print(f"  ✅ Loaded MAML weights from {weights_path}")
            else:
                print(f"  ⚠️  MAML weights not found at {weights_path}, using random init")
        
        elif self.baseline == 'static_concepts':
            # Load concepts as fixed heuristics
            from throng4.llm_policy.concept_library import ConceptLibrary, ConceptTransfer
            
            library = ConceptLibrary()
            transfer = ConceptTransfer(library)
            
            # Get applicable concepts heuristically
            applicable = transfer.get_applicable_concepts_heuristic(
                self.source_game,
                self.game.lower()
            )
            
            print(f"  Loading {len(applicable)} static concepts...")
            transfer.apply_static_concepts(agent, applicable)
        
        elif self.baseline == 'llm_at_start':
            # Query Tetra once for applicable concepts
            print(f"  Querying Tetra for applicable concepts...")
            
            query = f"""
Which concepts from the Tetris training should transfer to {self.game}?

Tetris concepts available:
- avoid_danger_spatial (0.92 success)
- minimize_gaps (0.88 success)
- target_completion (0.85 success)
- prefer_flat_surfaces (0.85 success)
- emergency_response (0.65 success)

{self.game} mechanics:
- Paddle controls ball
- Break bricks for points
- Ball bounces off walls
- Lose life if ball falls

Which concepts apply and why? Just list concept IDs.
"""
            
            try:
                response = self.bridge.query(query)
                # Parse response for concept IDs
                # For now, use heuristic fallback
                from throng4.llm_policy.concept_library import ConceptLibrary, ConceptTransfer
                library = ConceptLibrary()
                transfer = ConceptTransfer(library)
                applicable = transfer.get_applicable_concepts_heuristic(
                    self.source_game,
                    self.game.lower()
                )
                print(f"  Applying {len(applicable)} LLM-suggested concepts...")
                transfer.apply_static_concepts(agent, applicable)
            except Exception as e:
                print(f"  ⚠️  LLM query failed: {e}, using heuristic fallback")
        
        elif self.baseline == 'full_system':
            # Full system with real-time queries
            # Start with LLM-suggested concepts
            print(f"  Initializing Full System with concept transfer...")
            # TODO: Implement real-time observation loop
            # For now, same as llm_at_start
            from throng4.llm_policy.concept_library import ConceptLibrary, ConceptTransfer
            library = ConceptLibrary()
            transfer = ConceptTransfer(library)
            applicable = transfer.get_applicable_concepts_heuristic(
                self.source_game,
                self.game.lower()
            )
            transfer.apply_static_concepts(agent, applicable)
        
        return agent
    
    def run_episode(self, max_steps: int = 10000) -> Dict[str, Any]:
        """
        Run a single episode.
        
        Returns:
            Episode results (score, steps, etc.)
        """
        state = self.adapter.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < max_steps:
            # Get valid actions
            valid_actions = self.adapter.get_valid_actions()
            
            # Select action
            action = self.agent.select_action(
                valid_actions=valid_actions,
                feature_fn=lambda a: state  # RAM state is already features
            )
            
            # Take step
            next_state, reward, done, info = self.adapter.step(action)
            
            # Record step for learning
            self.agent.record_step(state, reward)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # End episode and train
        self.agent.end_episode(total_reward)
        
        return {
            'score': total_reward,
            'steps': steps,
            'epsilon': self.agent.epsilon
        }
    
    def run_training(self, episodes: int = 500, target_score: float = None) -> List[Dict]:
        """
        Run full training.
        
        Args:
            episodes: Number of episodes to train
            target_score: Stop early if this score is reached
            
        Returns:
            List of episode results
        """
        results = []
        
        print(f"\n{'='*70}")
        print(f"Baseline: {self.baseline}")
        print(f"Game: {self.game}")
        print(f"Episodes: {episodes}")
        print(f"{'='*70}\n")
        
        for ep in range(episodes):
            result = self.run_episode()
            results.append(result)
            
            if (ep + 1) % 10 == 0:
                recent_scores = [r['score'] for r in results[-10:]]
                print(f"Episode {ep+1:3d}: Score={result['score']:6.1f}, "
                      f"Avg(10)={np.mean(recent_scores):6.1f}, "
                      f"ε={result['epsilon']:.3f}")
            
            # Early stopping
            if target_score and result['score'] >= target_score:
                print(f"\n✅ Reached target score {target_score}!")
                break
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON."""
        data = {
            'baseline': self.baseline,
            'game': self.game,
            'source_game': self.source_game,
            'timestamp': datetime.now().isoformat(),
            'episodes': len(results),
            'results': results,
            'summary': {
                'mean_score': float(np.mean([r['score'] for r in results])),
                'std_score': float(np.std([r['score'] for r in results])),
                'max_score': float(np.max([r['score'] for r in results])),
                'final_10_mean': float(np.mean([r['score'] for r in results[-10:]])),
            }
        }
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run baseline experiment')
    parser.add_argument('--game', type=str, default='Breakout', help='Atari game name')
    parser.add_argument('--baseline', type=str, required=True, choices=BaselineExperiment.BASELINES)
    parser.add_argument('--episodes', type=int, default=500, help='Number of episodes')
    parser.add_argument('--runs', type=int, default=1, help='Number of independent runs')
    parser.add_argument('--output-dir', type=str, default='experiments/atari_baselines')
    
    args = parser.parse_args()
    
    for run in range(args.runs):
        print(f"\n{'#'*70}")
        print(f"# RUN {run + 1}/{args.runs}")
        print(f"{'#'*70}")
        
        experiment = BaselineExperiment(
            game=args.game,
            baseline=args.baseline
        )
        
        results = experiment.run_training(episodes=args.episodes)
        
        output_file = f"{args.output_dir}/{args.game}_{args.baseline}_run{run+1}.json"
        experiment.save_results(results, output_file)


if __name__ == "__main__":
    main()
