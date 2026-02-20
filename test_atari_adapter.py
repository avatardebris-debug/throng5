import numpy as np
from throng4.environments.atari_adapter import AtariAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig


def test_atari_adapter():
    """Test AtariAdapter and PortableNNAgent together."""
    print("Testing Atari Adapter with ALE/Breakout-v5 (RAM state)")
    adapter = AtariAdapter(game_id="ALE/Breakout-v5")
    
    # 128 RAM features + N action features (one-hot)
    num_actions = adapter.env.action_space.n
    n_features = 128 + num_actions
    print(f"Features: 128 RAM + {num_actions} Actions = {n_features}")
    
    config = AgentConfig(
        n_hidden=128,
        n_hidden2=64,
        epsilon=1.0,         # Pure exploration for test
        train_freq=4,
        batch_size=32,
        replay_buffer_size=1000
    )
    
    agent = PortableNNAgent(n_features=n_features, config=config)
    
    state = adapter.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    trajectory = []
    
    while not done and steps < 1000:
        valid_actions = adapter.get_valid_actions()
        action = agent.select_action(
            valid_actions=valid_actions,
            feature_fn=adapter.make_features,
            explore=True,
            lookahead_fn=None
        )
        
        # Make features BEFORE step, as step mutates env state!
        features = adapter.make_features(action)
        
        # Take step
        next_obs, reward, done, info = adapter.step(action)
        
        # Log semantic string
        semantic_string = adapter.get_semantic_obs(action, reward)
        trajectory.append({
            "step": steps,
            "obs": semantic_string,
        })
        
        next_valid_actions = adapter.get_valid_actions()
        if not done and next_valid_actions:
            next_features = [adapter.make_features(a) for a in next_valid_actions]
        else:
            next_features = []
            
        agent.record_step(features, reward, next_features, done)
        
        episode_reward += reward
        steps += 1
        
        if steps % 50 == 0:
            print(semantic_string)
            
    agent.end_episode(final_score=episode_reward)
    
    # Save offline log dataset
    import json
    with open("atari_offline_log.json", "w") as f:
        json.dump({
            "game_id": "Breakout",
            "episodes": 1,
            "trajectory": trajectory
        }, f, indent=2)
    
    stats = agent.get_stats()
    print("\nEpisode finished!")
    print(f"Total Steps: {steps}")
    print(f"Total Reward: {episode_reward}")
    print(f"Agent Updates: {stats['total_updates']}")
    print(f"Replay Buffer Size: {stats['buffer_size']}")
    print(f"Avg Loss: {stats['avg_loss']:.4f}")

if __name__ == "__main__":
    test_atari_adapter()
