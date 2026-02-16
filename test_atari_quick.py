"""
Quick test: Train Breakout with Tabula Rasa baseline (10 episodes).

This validates the setup before running full experiments.
"""
import sys
import numpy as np
from throng4.environments.atari_adapter import AtariAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig


def main():
    print("="*70)
    print("QUICK TEST: Breakout Tabula Rasa (10 episodes)")
    print("="*70)
    
    # Create adapter
    adapter = AtariAdapter('Breakout', max_steps=1000)
    print(f"\nGame: {adapter.game_name}")
    print(f"Features: {adapter.n_features}")
    print(f"Actions: {len(adapter.get_valid_actions())}")
    
    # Create agent
    config = AgentConfig(
        n_hidden=128,
        epsilon=0.3,
        learning_rate=0.01
    )
    agent = PortableNNAgent(n_features=adapter.n_features, config=config)
    
    # Train for 10 episodes
    scores = []
    
    for ep in range(10):
        state = adapter.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < 1000:
            # Select action
            valid_actions = adapter.get_valid_actions()
            action = agent.select_action(
                valid_actions=valid_actions,
                feature_fn=lambda a: state
            )
            
            # Take step
            next_state, reward, done, info = adapter.step(action)
            
            # Record for learning
            agent.record_step(state, reward)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # End episode
        agent.end_episode(total_reward)
        scores.append(total_reward)
        
        print(f"Episode {ep+1:2d}: Score={total_reward:6.1f}, Steps={steps:4d}, ε={agent.epsilon:.3f}")
    
    adapter.close()
    
    print(f"\n{'='*70}")
    print(f"Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Max Score: {np.max(scores):.1f}")
    print(f"{'='*70}")
    print("\n✅ Quick test complete! Setup validated.")


if __name__ == "__main__":
    main()
