"""
Test: Verify Dual-Head ANN Architecture

Simple test to ensure:
1. Forward pass produces both Q-values and reward predictions
2. Backward passes update weights correctly
3. DQN learner integrates properly
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.layers.meta0_ann import ANNLayer
from throng4.learning.dqn import DQNLearner, DQNConfig


def test_ann_forward():
    """Test dual-head forward pass."""
    print("=" * 60)
    print("TEST 1: Dual-Head Forward Pass")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=25, n_hidden=64, n_outputs=4)
    
    # Random state (5×5 GridWorld)
    state = np.random.randn(25)
    
    # Forward pass
    output = ann.forward(state)
    
    print(f"Input shape: {state.shape}")
    print(f"Q-values shape: {output['q_values'].shape}")
    print(f"Q-values: {output['q_values']}")
    print(f"Reward prediction: {output['reward_pred']:.4f}")
    print(f"Hidden activations shape: {output['hidden'].shape}")
    print(f"Number of parameters: {ann.get_num_parameters()}")
    
    assert output['q_values'].shape == (4,), "Q-values should have 4 actions"
    assert isinstance(output['reward_pred'], (float, np.floating)), "Reward pred should be scalar"
    assert output['hidden'].shape == (64,), "Hidden should match n_hidden"
    
    print("✅ Forward pass test PASSED\n")
    return ann


def test_ann_backward():
    """Test backward passes update weights."""
    print("=" * 60)
    print("TEST 2: Backward Pass Updates")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=25, n_hidden=64, n_outputs=4)
    
    # Get initial weights
    initial_weights = ann.get_weights()
    
    # Forward pass
    state = np.random.randn(25)
    output = ann.forward(state)
    
    # Backward pass for Q-learning
    td_error = 0.5
    action = 2
    ann.backward_q(td_error, action, lr=0.01)
    
    # Backward pass for reward prediction
    reward_error = 0.3
    ann.backward_reward(reward_error, lr=0.01, aux_weight=0.1)
    
    # Check weights changed
    updated_weights = ann.get_weights()
    
    q_weights_changed = not np.allclose(initial_weights['W_q'], updated_weights['W_q'])
    r_weights_changed = not np.allclose(initial_weights['W_r'], updated_weights['W_r'])
    backbone_changed = not np.allclose(initial_weights['W1'], updated_weights['W1'])
    
    print(f"Q-head weights changed: {q_weights_changed}")
    print(f"Reward-head weights changed: {r_weights_changed}")
    print(f"Backbone weights changed: {backbone_changed}")
    
    assert q_weights_changed, "Q-head weights should update"
    assert r_weights_changed, "Reward-head weights should update"
    assert backbone_changed, "Backbone should get gradients from both heads"
    
    print("✅ Backward pass test PASSED\n")


def test_dqn_learner():
    """Test DQN learner integration."""
    print("=" * 60)
    print("TEST 3: DQN Learner Integration")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=25, n_hidden=64, n_outputs=4)
    config = DQNConfig(
        learning_rate=0.01,
        epsilon=0.1,
        aux_loss_weight=0.1
    )
    learner = DQNLearner(ann, config)
    
    # Simulate a few transitions
    state = np.random.randn(25)
    
    for i in range(5):
        action = learner.select_action(state, explore=True)
        next_state = np.random.randn(25)
        reward = np.random.randn()
        done = (i == 4)
        
        errors = learner.update(state, action, reward, next_state, done)
        
        print(f"Step {i}: action={action}, reward={reward:.3f}, "
              f"TD_err={errors['td_error']:.3f}, "
              f"R_err={errors['reward_error']:.3f}")
        
        state = next_state
    
    stats = learner.get_stats()
    print(f"\nLearner stats:")
    print(f"  Updates: {stats['n_updates']}")
    print(f"  Epsilon: {stats['epsilon']:.3f}")
    print(f"  Buffer size: {stats['buffer_size']}")
    print(f"  Mean TD error: {stats['mean_td_error']:.3f}")
    print(f"  Mean reward error: {stats['mean_reward_error']:.3f}")
    
    assert stats['n_updates'] == 5, "Should have 5 updates"
    assert stats['buffer_size'] == 5, "Should have 5 transitions in buffer"
    
    print("✅ DQN learner test PASSED\n")


def test_exploration():
    """Test epsilon-greedy exploration."""
    print("=" * 60)
    print("TEST 4: Epsilon-Greedy Exploration")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=25, n_hidden=64, n_outputs=4)
    config = DQNConfig(epsilon=0.5)  # 50% exploration
    learner = DQNLearner(ann, config)
    
    state = np.random.randn(25)
    
    # Sample 100 actions
    actions = [learner.select_action(state, explore=True) for _ in range(100)]
    
    # Count unique actions (should see variety with 50% epsilon)
    unique_actions = len(set(actions))
    
    print(f"Unique actions sampled (out of 4): {unique_actions}")
    print(f"Action distribution: {np.bincount(actions, minlength=4)}")
    
    # With 50% epsilon, we should see at least 2 different actions
    assert unique_actions >= 2, "Should explore multiple actions"
    
    # Test greedy mode (no exploration)
    greedy_actions = [learner.select_action(state, explore=False) for _ in range(10)]
    assert len(set(greedy_actions)) == 1, "Greedy should always pick same action"
    
    print("✅ Exploration test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("THRONG4 DUAL-HEAD ARCHITECTURE VERIFICATION")
    print("=" * 60 + "\n")
    
    try:
        test_ann_forward()
        test_ann_backward()
        test_dqn_learner()
        test_exploration()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        print("\nThrong4 scaffolding is ready for integration!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
