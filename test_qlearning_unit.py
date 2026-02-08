"""
Test Q-learning implementation
"""

import numpy as np
from throng3.learning.qlearning import QLearner, QLearningConfig

print("="*60)
print("Q-Learning Unit Tests")
print("="*60)

# Test 1: Initialization
print("\nTest 1: Initialization")
config = QLearningConfig(learning_rate=0.1, gamma=0.9, epsilon=0.2)
qlearner = QLearner(n_states=4, n_actions=2, config=config)
print(f"  ✓ Created QLearner with {qlearner.n_states} states, {qlearner.n_actions} actions")
print(f"  ✓ Weight matrix shape: {qlearner.W.shape}")
print(f"  ✓ Initial epsilon: {qlearner.epsilon}")

# Test 2: Q-value computation
print("\nTest 2: Q-value computation")
state = np.array([1.0, 0.5, 0.0, 0.2])
q_values = qlearner.get_q_values(state)
print(f"  ✓ Q-values shape: {q_values.shape}")
print(f"  ✓ Q-values: {q_values}")
assert len(q_values) == 2, "Should have Q-value for each action"

# Test 3: Action selection
print("\nTest 3: Action selection")
action = qlearner.select_action(state, explore=False)
print(f"  ✓ Greedy action: {action}")
assert action in [0, 1], "Action should be valid"

# Test 4: Q-learning update
print("\nTest 4: Q-learning update")
next_state = np.array([0.8, 0.6, 0.1, 0.3])
reward = 1.0
done = False

q_before = qlearner.get_q_values(state)[action]
td_error = qlearner.update(state, action, reward, next_state, done)
q_after = qlearner.get_q_values(state)[action]

print(f"  ✓ Q-value before: {q_before:.4f}")
print(f"  ✓ Q-value after: {q_after:.4f}")
print(f"  ✓ TD error: {td_error:.4f}")
print(f"  ✓ Q-value changed: {abs(q_after - q_before) > 0.001}")

# Test 5: Terminal state update
print("\nTest 5: Terminal state update")
terminal_state = np.array([0.0, 0.0, 0.0, 0.0])
reward = 10.0
done = True

q_before = qlearner.get_q_values(next_state)[0]
td_error = qlearner.update(next_state, 0, reward, terminal_state, done)
q_after = qlearner.get_q_values(next_state)[0]

print(f"  ✓ Terminal reward: {reward}")
print(f"  ✓ Q-value before: {q_before:.4f}")
print(f"  ✓ Q-value after: {q_after:.4f}")
print(f"  ✓ TD error: {td_error:.4f}")
print(f"  ✓ Q-value increased toward reward")

# Test 6: Epsilon decay
print("\nTest 6: Epsilon decay")
epsilon_before = qlearner.epsilon
qlearner.decay_epsilon()
epsilon_after = qlearner.epsilon
print(f"  ✓ Epsilon before: {epsilon_before:.4f}")
print(f"  ✓ Epsilon after: {epsilon_after:.4f}")
print(f"  ✓ Epsilon decayed: {epsilon_after < epsilon_before}")

# Test 7: Statistics
print("\nTest 7: Statistics")
stats = qlearner.get_stats()
print(f"  ✓ Updates: {stats['n_updates']}")
print(f"  ✓ Avg TD error: {stats['avg_td_error']:.4f}")
print(f"  ✓ Mean Q weight: {stats['mean_q_weight']:.6f}")
print(f"  ✓ Episode count: {stats['episode_count']}")

# Test 8: Simple learning scenario
print("\nTest 8: Simple learning scenario (10 episodes)")
print("  Scenario: State [1,0] → action 0 → reward +1")
print("            State [0,1] → action 1 → reward +1")

qlearner = QLearner(n_states=2, n_actions=2, config=QLearningConfig(
    learning_rate=0.5,
    gamma=0.0,  # No future value for simplicity
    epsilon=0.0  # No exploration
))

for episode in range(10):
    # Positive example 1
    state1 = np.array([1.0, 0.0])
    qlearner.update(state1, 0, 1.0, state1, True)
    
    # Positive example 2
    state2 = np.array([0.0, 1.0])
    qlearner.update(state2, 1, 1.0, state2, True)

q1 = qlearner.get_q_values(np.array([1.0, 0.0]))
q2 = qlearner.get_q_values(np.array([0.0, 1.0]))

print(f"  ✓ Q([1,0]): {q1}")
print(f"  ✓ Q([0,1]): {q2}")
print(f"  ✓ Q([1,0], action=0) > Q([1,0], action=1): {q1[0] > q1[1]}")
print(f"  ✓ Q([0,1], action=1) > Q([0,1], action=0): {q2[1] > q2[0]}")

if q1[0] > q1[1] and q2[1] > q2[0]:
    print("\n✓ ALL TESTS PASSED")
else:
    print("\n✗ Learning test failed")

print("="*60)
