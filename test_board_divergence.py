import sys
sys.path.insert(0, '.')
from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
import random
import numpy as np

adapter = TetrisAdapter(level=2, seed=42)
np.random.seed(42)
random.seed(42)
adapter.reset()

config = AgentConfig(n_hidden=48, epsilon=0.0)
agent = PortableNNAgent(n_features=adapter.n_features, config=config, seed=42)

episode_reward = 0.0
done = False
step = 0
history = []

while not done and step < 10:
    valid_actions = adapter.get_valid_actions()
    if not valid_actions:
        break
        
    piece = adapter.current_piece
    action = agent.select_action(
        valid_actions=valid_actions,
        feature_fn=adapter.make_features,
        explore=False
    )
    
    state, reward, done, info = adapter.step(action)
    history.append((piece, action, reward))
    step += 1

print("\n--- Original Sequence ---")
for p, a, r in history:
    print(f"Piece: {p}, Action: {a}, Reward: {r:.2f}")

print("\n--- Testing Replay ---")
np.random.seed(42)
random.seed(42)

replay_adapter = TetrisAdapter(level=2, seed=42)
replay_adapter.reset()

for i, (orig_piece, action, _) in enumerate(history):
    replay_piece = replay_adapter.current_piece
    
    print(f"Step {i} Piece Mismatch? {orig_piece != replay_piece} (Orig: {orig_piece}, Replay: {replay_piece})")
    
    state, reward, done, info = replay_adapter.step(action)
    print(f"Step {i} Reward Mismatch? {history[i][2] != reward} (Orig: {history[i][2]:.2f}, Replay: {reward:.2f})")
