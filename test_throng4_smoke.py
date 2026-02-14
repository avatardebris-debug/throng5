"""
Throng4 Smoke Test — Quick validation (<30 seconds)

Runs minimal checks on all components:
- ANNLayer forward/backward
- DQNLearner update
- Meta^1 synapse optimizer
- Meta^3 MAML inner loop
- SimplePipeline on GridWorld (10 episodes)
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.layers.meta0_ann import ANNLayer
from throng4.layers.meta1_synapse import DualHeadSynapseOptimizer
from throng4.layers.meta3_maml import DualHeadMAML
from throng4.learning.dqn import DQNLearner, DQNConfig
from throng4.pipeline import SimplePipeline
from throng3.environments.gym_envs import GridWorldAdapter


def test_ann():
    """Quick ANN test."""
    ann = ANNLayer(n_inputs=4, n_hidden=16, n_outputs=2)
    state = np.random.randn(4)
    output = ann.forward(state)
    assert output['q_values'].shape == (2,)
    ann.backward_q(0.5, 1, lr=0.01)
    ann.backward_reward(0.3, lr=0.01)
    print("[OK] ANNLayer")


def test_dqn():
    """Quick DQN test."""
    ann = ANNLayer(n_inputs=4, n_hidden=16, n_outputs=2)
    learner = DQNLearner(ann, DQNConfig())
    
    state = np.random.randn(4)
    action = learner.select_action(state)
    next_state = np.random.randn(4)
    errors = learner.update(state, action, 0.5, next_state, False)
    
    assert 'td_error' in errors
    assert 'reward_error' in errors
    print("[OK] DQNLearner")


def test_meta1():
    """Quick Meta^1 test."""
    ann = ANNLayer(n_inputs=2, n_hidden=16, n_outputs=4)
    optimizer = DualHeadSynapseOptimizer(ann)
    
    result = optimizer.optimize({
        'state': np.array([0.2, 0.3]),
        'action': 1,
        'reward': 0.5,
        'next_state': np.array([0.4, 0.3]),
        'done': False,
    })
    
    assert 'td_error' in result
    print("[OK] Meta^1 Synapse")


def test_maml():
    """Quick MAML test."""
    ann = ANNLayer(n_inputs=2, n_hidden=16, n_outputs=4)
    maml = DualHeadMAML()
    
    transitions = [
        {'state': np.array([0.0, 0.0]), 'action': 3, 'reward': -0.01,
         'next_state': np.array([0.25, 0.0]), 'done': False},
    ]
    
    adapted = maml.inner_loop(ann, transitions)
    assert len(adapted) > 0
    print("[OK] Meta^3 MAML")


def test_gridworld():
    """Quick GridWorld test (10 episodes)."""
    env = GridWorldAdapter(size=5)
    pipeline = SimplePipeline(n_inputs=2, n_outputs=4, n_hidden=32)
    
    rewards = []
    for _ in range(10):
        state = env.reset()
        total_reward = 0
        for _ in range(50):
            action = pipeline.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            pipeline.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        rewards.append(total_reward)
    
    mean_reward = np.mean(rewards)
    print(f"[OK] GridWorld (mean reward: {mean_reward:.3f})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("THRONG4 SMOKE TEST")
    print("=" * 60 + "\n")
    
    try:
        test_ann()
        test_dqn()
        test_meta1()
        test_maml()
        test_gridworld()
        
        print("\n" + "=" * 60)
        print("ALL SMOKE TESTS PASSED [OK]")
        print("=" * 60)
        print("\nThrong4 foundation is validated!")
        
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
