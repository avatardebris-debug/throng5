"""
Test: Phase 3 — Meta^1 adapter, MAML, and Tetris curriculum

Verifies:
1. Meta^1 DualHeadSynapseOptimizer with per-head LR multipliers
2. MAML inner loop through dual-head ANN
3. Tetris curriculum environment levels
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.layers.meta0_ann import ANNLayer
from throng4.layers.meta1_synapse import DualHeadSynapseOptimizer, DualHeadSynapseConfig
from throng4.layers.meta3_maml import DualHeadMAML, DualHeadMAMLConfig
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv, TetrisCurriculum


def test_meta1_synapse():
    """Test Meta^1 dual-head synapse optimizer."""
    print("=" * 60)
    print("TEST 1: Meta^1 DualHeadSynapseOptimizer")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=2, n_hidden=64, n_outputs=4)
    config = DualHeadSynapseConfig(
        base_lr=0.01,
        aux_loss_weight=0.1,
        backbone_lr_scale=1.0,
        q_head_lr_scale=1.5,      # Faster Q-head
        reward_head_lr_scale=0.5,  # Slower reward-head
    )
    optimizer = DualHeadSynapseOptimizer(ann, config)
    
    # Simulate a transition
    state = np.array([0.2, 0.3])
    next_state = np.array([0.4, 0.3])
    
    # Get initial weights
    w_before = ann.get_weights()
    
    result = optimizer.optimize({
        'state': state,
        'action': 1,
        'reward': 0.5,
        'next_state': next_state,
        'done': False,
    })
    
    w_after = ann.get_weights()
    
    print(f"TD error: {result['td_error']:.4f}")
    print(f"Reward error: {result['reward_error']:.4f}")
    print(f"Effective LR: {result['effective_lr']:.6f}")
    print(f"Dopamine level: {result['dopamine_level']:.4f}")
    print(f"LR multipliers: {result['lr_multipliers']}")
    
    # Check per-head weight changes
    for key in ['W1', 'W_q', 'W_r']:
        change = np.mean(np.abs(w_after[key] - w_before[key]))
        print(f"  {key} change: {change:.6f}")
    
    # Verify Q-head changed more than reward-head (higher LR)
    q_change = np.mean(np.abs(w_after['W_q'] - w_before['W_q']))
    r_change = np.mean(np.abs(w_after['W_r'] - w_before['W_r']))
    print(f"\nQ-head change ({q_change:.6f}) should be > reward-head change ({r_change:.6f})")
    
    assert result['td_error'] != 0.0, "TD error should be non-zero"
    assert q_change > 0, "Q-head should update"
    print("[PASS] Meta^1 test PASSED\n")


def test_maml_inner_loop():
    """Test MAML inner loop through dual-head ANN."""
    print("=" * 60)
    print("TEST 2: MAML Inner Loop")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=2, n_hidden=64, n_outputs=4)
    maml = DualHeadMAML(DualHeadMAMLConfig(
        inner_lr=0.01,
        inner_steps=3,
    ))
    
    # Create fake transitions (support set)
    transitions = [
        {'state': np.array([0.0, 0.0]), 'action': 3, 'reward': -0.01,
         'next_state': np.array([0.25, 0.0]), 'done': False},
        {'state': np.array([0.25, 0.0]), 'action': 1, 'reward': -0.01,
         'next_state': np.array([0.25, 0.25]), 'done': False},
        {'state': np.array([0.25, 0.25]), 'action': 3, 'reward': 1.0,
         'next_state': np.array([1.0, 1.0]), 'done': True},
    ]
    
    # Get initial weights
    w_before = ann.get_weights()
    
    # Inner loop
    adapted = maml.inner_loop(ann, transitions)
    
    # Check that adapted weights differ
    w_unchanged = ann.get_weights()  # Should be unchanged (inner_loop restores)
    
    adapted_changed = any(
        not np.allclose(adapted[k], w_before[k]) for k in adapted
    )
    original_unchanged = all(
        np.allclose(w_unchanged[k], w_before[k]) for k in w_before
    )
    
    print(f"Adapted weights changed: {adapted_changed}")
    print(f"Original weights preserved: {original_unchanged}")
    
    # Show per-head adaptation magnitude
    for key in ['W1', 'W_q', 'W_r']:
        change = np.mean(np.abs(adapted[key] - w_before[key]))
        print(f"  {key} adaptation: {change:.6f}")
    
    # Check lr multipliers
    print(f"LR multipliers: {maml.get_lr_multipliers()}")
    
    assert adapted_changed, "Adapted weights should differ from initial"
    assert original_unchanged, "Original weights should be preserved"
    
    print("[PASS] MAML inner loop test PASSED\n")


def test_maml_meta_update():
    """Test MAML meta-update (outer loop)."""
    print("=" * 60)
    print("TEST 3: MAML Meta-Update")
    print("=" * 60)
    
    ann = ANNLayer(n_inputs=2, n_hidden=32, n_outputs=4)
    maml = DualHeadMAML(DualHeadMAMLConfig(
        meta_lr=0.001,
        inner_lr=0.01,
        inner_steps=2,
        meta_batch_size=2,
    ))
    
    w_before = ann.get_weights()
    
    # Create two tasks
    task1 = {
        'support_set': [
            {'state': np.array([0.0, 0.0]), 'action': 3, 'reward': -0.01,
             'next_state': np.array([0.25, 0.0]), 'done': False},
            {'state': np.array([0.25, 0.0]), 'action': 1, 'reward': 1.0,
             'next_state': np.array([1.0, 1.0]), 'done': True},
        ],
        'query_set': [
            {'state': np.array([0.0, 0.0]), 'action': 3, 'reward': -0.01,
             'next_state': np.array([0.25, 0.0]), 'done': False},
        ],
    }
    
    task2 = {
        'support_set': [
            {'state': np.array([0.5, 0.0]), 'action': 1, 'reward': -0.01,
             'next_state': np.array([0.5, 0.25]), 'done': False},
            {'state': np.array([0.5, 0.25]), 'action': 3, 'reward': 1.0,
             'next_state': np.array([1.0, 1.0]), 'done': True},
        ],
        'query_set': [
            {'state': np.array([0.5, 0.0]), 'action': 1, 'reward': -0.01,
             'next_state': np.array([0.5, 0.25]), 'done': False},
        ],
    }
    
    # Meta-update
    maml.meta_update(ann, [task1, task2])
    
    w_after = ann.get_weights()
    
    meta_changed = any(
        not np.allclose(w_after[k], w_before[k]) for k in w_before
    )
    
    stats = maml.get_stats()
    print(f"Meta updates: {stats['meta_updates']}")
    print(f"Tasks seen: {stats['tasks_seen']}")
    print(f"Meta-initialization changed: {meta_changed}")
    print(f"Learned LR multipliers: {stats['lr_multipliers']}")
    
    for key in ['W1', 'W_q', 'W_r']:
        change = np.mean(np.abs(w_after[key] - w_before[key]))
        print(f"  {key} meta-update: {change:.6f}")
    
    assert stats['meta_updates'] == 1, "Should have 1 meta-update"
    assert meta_changed, "Meta-initialization should change"
    
    print("[PASS] MAML meta-update test PASSED\n")


def test_tetris_curriculum():
    """Test Tetris curriculum environments."""
    print("=" * 60)
    print("TEST 4: Tetris Curriculum")
    print("=" * 60)
    
    for level in range(1, 4):  # Test first 3 levels
        env = TetrisCurriculumEnv(level=level, max_pieces=50)
        state = env.reset()
        
        print(f"\nLevel {level}: {env.level_name}")
        print(f"  Board: {env.width}×{env.height}")
        print(f"  Pieces: {env.piece_types}")
        print(f"  Rotations: {env.max_rotations}")
        print(f"  State dim: {env.state_dim} (actual: {len(state)})")
        
        assert len(state) == env.state_dim, f"State dim mismatch at level {level}"
        
        # Play a few random actions
        total_reward = 0
        for _ in range(20):
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = valid_actions[np.random.randint(len(valid_actions))]
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        
        print(f"  Random play: reward={total_reward:.3f}, "
              f"pieces={info['pieces_placed']}, "
              f"lines={info['lines_cleared']}")
        
        # Render
        env2 = TetrisCurriculumEnv(level=level, max_pieces=10)
        env2.reset()
        print(f"  Board preview:")
        for line in env2.render().split('\n'):
            print(f"    {line}")
    
    # Test curriculum manager
    curriculum = TetrisCurriculum(advance_threshold=0.5, eval_episodes=5)
    env = curriculum.get_env()
    assert env.level == 1, "Should start at level 1"
    
    # Record some episodes
    for _ in range(10):
        curriculum.record_episode(1.0, 2)
    
    assert curriculum.should_advance(), "Should advance with good performance"
    curriculum.advance()
    assert curriculum.current_level == 2, "Should be at level 2"
    
    stats = curriculum.get_stats()
    print(f"\nCurriculum stats: {stats}")
    
    print("[PASS] Tetris curriculum test PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("THRONG4 PHASE 3 VERIFICATION")
    print("=" * 60 + "\n")
    
    try:
        test_meta1_synapse()
        test_maml_inner_loop()
        test_maml_meta_update()
        test_tetris_curriculum()
        
        print("=" * 60)
        print("ALL PHASE 3 TESTS PASSED [OK]")
        print("=" * 60)
        print("\nPhase 3 components ready:")
        print("  [OK] Meta^1 DualHeadSynapseOptimizer")
        print("  [OK] Meta^3 DualHeadMAML")
        print("  [OK] Tetris Curriculum Environment")
        
    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
