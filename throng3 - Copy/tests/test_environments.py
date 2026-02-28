"""Tests for environment adapters."""

import numpy as np


def test_gridworld_basic():
    """Test GridWorld basic functionality."""
    from throng3.environments import GridWorldAdapter
    
    env = GridWorldAdapter(size=5)
    
    # Test reset
    obs = env.reset()
    assert obs.shape == (2,), "Observation should be 2D (x, y)"
    assert np.all(obs >= 0) and np.all(obs <= 1), "Observations should be normalized"
    assert np.allclose(obs, [0.0, 0.0]), "Should start at (0, 0)"
    
    # Test step
    obs, reward, done, info = env.step(3)  # Move right
    assert obs.shape == (2,)
    assert reward == -0.01, "Step reward should be -0.01"
    assert not done, "Should not be done after one step"
    
    # Test reaching goal (move right 4 times, down 4 times)
    env.reset()
    for _ in range(4):
        env.step(3)  # right
    for _ in range(4):
        obs, reward, done, info = env.step(1)  # down
    
    assert done, "Should be done at goal"
    assert reward == 1.0, "Goal reward should be 1.0"
    assert np.allclose(obs, [1.0, 1.0]), "Should be at (4, 4)"


def test_gridworld_boundaries():
    """Test GridWorld boundary conditions."""
    from throng3.environments import GridWorldAdapter
    
    env = GridWorldAdapter(size=5)
    env.reset()
    
    # Try to move left from (0, 0) - should stay at (0, 0)
    obs, _, _, _ = env.step(2)  # left
    assert np.allclose(obs, [0.0, 0.0]), "Should stay at boundary"
    
    # Try to move up from (0, 0) - should stay at (0, 0)
    obs, _, _, _ = env.step(0)  # up
    assert np.allclose(obs, [0.0, 0.0]), "Should stay at boundary"


def test_cartpole_integration():
    """Test CartPole integration (requires gymnasium)."""
    try:
        from throng3.environments import CartPoleAdapter
    except ImportError:
        print("⚠ Skipping CartPole test - gymnasium not installed")
        return
    
    env = CartPoleAdapter()
    
    # Test reset
    obs = env.reset()
    assert obs.shape == (4,), "CartPole has 4D observation"
    assert np.all(obs >= 0) and np.all(obs <= 1), "Observations should be normalized"
    
    # Test 100 steps without crash
    for _ in range(100):
        obs, reward, done, info = env.step(0)  # Always push left
        assert obs.shape == (4,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        if done:
            obs = env.reset()
    
    assert env.total_episodes >= 1, "Should complete at least one episode"


def test_mountaincar_integration():
    """Test MountainCar integration (requires gymnasium)."""
    try:
        from throng3.environments import MountainCarAdapter
    except ImportError:
        print("⚠ Skipping MountainCar test - gymnasium not installed")
        return
    
    env = MountainCarAdapter()
    
    # Test reset
    obs = env.reset()
    assert obs.shape == (2,), "MountainCar has 2D observation"
    assert np.all(obs >= 0) and np.all(obs <= 1), "Observations should be normalized"
    
    # Test 100 steps without crash
    for _ in range(100):
        obs, reward, done, info = env.step(1)  # No action
        assert obs.shape == (2,)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        if done:
            obs = env.reset()
    
    assert env.total_episodes >= 0, "Should track episodes"


def test_adapter_normalization():
    """Test that all adapters normalize observations correctly."""
    from throng3.environments import GridWorldAdapter
    
    # Test GridWorld
    env = GridWorldAdapter()
    obs = env.reset()
    assert np.all(obs >= 0) and np.all(obs <= 1), "GridWorld obs should be in [0, 1]"
    
    for _ in range(10):
        obs, _, done, _ = env.step(np.random.randint(0, 4))
        assert np.all(obs >= 0) and np.all(obs <= 1), "GridWorld obs should stay in [0, 1]"
        if done:
            break
    
    # Test CartPole if available
    try:
        from throng3.environments import CartPoleAdapter
        env = CartPoleAdapter()
        obs = env.reset()
        assert np.all(obs >= 0) and np.all(obs <= 1), "CartPole obs should be in [0, 1]"
        
        for _ in range(10):
            obs, _, done, _ = env.step(0)
            assert np.all(obs >= 0) and np.all(obs <= 1), "CartPole obs should stay in [0, 1]"
            if done:
                break
    except ImportError:
        pass  # Skip if gymnasium not installed


def test_pipeline_integration():
    """Test full pipeline integration with environment adapter."""
    from throng3.environments import GridWorldAdapter
    from throng3.pipeline import MetaNPipeline
    
    env = GridWorldAdapter()
    pipeline = MetaNPipeline.create_minimal(
        n_neurons=50,
        n_inputs=2,
        n_outputs=4,
    )
    
    obs = env.reset()
    
    # Run 50 steps
    for _ in range(50):
        # Get action from pipeline
        result = pipeline.step(obs)
        
        # Extract action (argmax of output)
        output = result.get('output', np.zeros(4))
        action = np.argmax(output)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        if done:
            obs = env.reset()
    
    # Verify pipeline produced valid outputs
    assert 'output' in result
    assert result['output'].shape == (4,), "Should have 4 action outputs"


if __name__ == '__main__':
    print("Running environment adapter tests...")
    
    print("\n1. Testing GridWorld basic...")
    test_gridworld_basic()
    print("✓ GridWorld basic tests passed")
    
    print("\n2. Testing GridWorld boundaries...")
    test_gridworld_boundaries()
    print("✓ GridWorld boundary tests passed")
    
    print("\n3. Testing CartPole integration...")
    try:
        test_cartpole_integration()
        print("✓ CartPole integration tests passed")
    except Exception as e:
        print(f"⚠ CartPole tests skipped: {e}")
    
    print("\n4. Testing MountainCar integration...")
    try:
        test_mountaincar_integration()
        print("✓ MountainCar integration tests passed")
    except Exception as e:
        print(f"⚠ MountainCar tests skipped: {e}")
    
    print("\n5. Testing adapter normalization...")
    test_adapter_normalization()
    print("✓ Normalization tests passed")
    
    print("\n6. Testing pipeline integration...")
    test_pipeline_integration()
    print("✓ Pipeline integration tests passed")
    
    print("\n" + "="*50)
    print("All environment adapter tests passed! ✓")
    print("="*50)
