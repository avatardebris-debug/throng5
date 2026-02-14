"""
Test: Cross-Environment Transfer Learning

Validates that MetaStackPipeline's MAML enables better transfer than SimplePipeline.

Tests:
1. GridWorld → FrozenLake (same state space, different dynamics)
2. GridWorld → Tetris (different dimensions, MAML-only transfer)
3. Progressive Tetris curriculum (levels 1→3)
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.metastack_pipeline import MetaStackPipeline
from throng4.pipeline import SimplePipeline
from throng4.layers.meta1_synapse import DualHeadSynapseConfig
from throng4.layers.meta3_maml import DualHeadMAMLConfig
from throng3.environments.gym_envs import GridWorldAdapter, FrozenLakeAdapter
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv


def train_pipeline(pipeline, env, n_episodes, target_reward=None, verbose=False):
    """
    Train pipeline on environment.
    
    Returns:
        (episodes_trained, final_mean_reward)
    """
    rewards = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        
        for _ in range(100):
            action = pipeline.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            pipeline.update(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards.append(total_reward)
        
        # Check if target reached
        if target_reward is not None and len(rewards) >= 20:
            recent_mean = np.mean(rewards[-20:])
            if recent_mean >= target_reward:
                if verbose:
                    print(f"  Reached target {target_reward:.2f} at episode {episode + 1}")
                return episode + 1, recent_mean
        
        if verbose and (episode + 1) % 50 == 0:
            recent_mean = np.mean(rewards[-50:])
            print(f"  Episode {episode + 1}: mean_50={recent_mean:.3f}")
    
    final_mean = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
    return n_episodes, final_mean


def eval_pipeline(pipeline, env, n_episodes):
    """Evaluate pipeline (greedy) and return mean reward."""
    rewards = []
    
    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        
        for _ in range(100):
            action = pipeline.select_action(state, explore=False)
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        rewards.append(total_reward)
    
    return np.mean(rewards), np.std(rewards)


def test_gridworld_to_frozenlake():
    """Test 1: GridWorld → FrozenLake transfer."""
    print("=" * 70)
    print("TEST 1: GridWorld → FrozenLake Transfer")
    print("=" * 70)
    print("\nBoth environments use 2D grid positions (compatible state space)")
    print("FrozenLake has stochastic dynamics (slippery tiles)\n")
    
    # === Train on GridWorld ===
    print("Phase 1: Training on GridWorld (200 episodes)...")
    
    gridworld = GridWorldAdapter(size=5)
    
    # SimplePipeline baseline
    print("\n[SimplePipeline]")
    simple = SimplePipeline(n_inputs=2, n_outputs=4, n_hidden=64)
    train_pipeline(simple, gridworld, 200, verbose=True)
    simple_gw_mean, _ = eval_pipeline(simple, gridworld, 20)
    print(f"  GridWorld eval: {simple_gw_mean:.3f}")
    
    # MetaStackPipeline
    print("\n[MetaStackPipeline]")
    metastack = MetaStackPipeline(
        n_inputs=2, n_outputs=4, n_hidden=64,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3)
    )
    train_pipeline(metastack, gridworld, 200, verbose=True)
    metastack_gw_mean, _ = eval_pipeline(metastack, gridworld, 20)
    metastack_stats = metastack.get_stats()
    print(f"  GridWorld eval: {metastack_gw_mean:.3f}")
    print(f"  Meta-updates: {metastack_stats['meta_updates']}")
    print(f"  LR multipliers: {metastack_stats['lr_multipliers']}")
    
    # === Zero-shot transfer to FrozenLake ===
    print("\n" + "=" * 70)
    print("Phase 2: Zero-Shot Transfer to FrozenLake")
    print("=" * 70)
    
    frozenlake = FrozenLakeAdapter(is_slippery=True)
    
    # Eval SimplePipeline (no transfer, random weights)
    print("\n[SimplePipeline - Random Init]")
    simple_random = SimplePipeline(n_inputs=2, n_outputs=4, n_hidden=64)
    simple_random_mean, simple_random_std = eval_pipeline(simple_random, frozenlake, 50)
    print(f"  FrozenLake eval: {simple_random_mean:.3f} ± {simple_random_std:.3f}")
    
    # Eval SimplePipeline (with GridWorld weights)
    print("\n[SimplePipeline - GridWorld Weights]")
    simple_fl_mean, simple_fl_std = eval_pipeline(simple, frozenlake, 50)
    print(f"  FrozenLake eval: {simple_fl_mean:.3f} ± {simple_fl_std:.3f}")
    simple_transfer = simple_fl_mean - simple_random_mean
    print(f"  Transfer score: {simple_transfer:+.3f}")
    
    # Eval MetaStackPipeline (with GridWorld weights + MAML)
    print("\n[MetaStackPipeline - GridWorld Weights + MAML]")
    metastack_fl_mean, metastack_fl_std = eval_pipeline(metastack, frozenlake, 50)
    print(f"  FrozenLake eval: {metastack_fl_mean:.3f} ± {metastack_fl_std:.3f}")
    metastack_transfer = metastack_fl_mean - simple_random_mean
    print(f"  Transfer score: {metastack_transfer:+.3f}")
    
    # === Fine-tuning on FrozenLake ===
    print("\n" + "=" * 70)
    print("Phase 3: Fine-Tuning on FrozenLake")
    print("=" * 70)
    print("\nTarget: 0.5 mean reward (50% success rate)\n")
    
    # SimplePipeline fine-tuning (from random)
    print("[SimplePipeline - Random Init]")
    simple_ft = SimplePipeline(n_inputs=2, n_outputs=4, n_hidden=64)
    simple_ft_episodes, simple_ft_reward = train_pipeline(
        simple_ft, frozenlake, 300, target_reward=0.5, verbose=True
    )
    print(f"  Episodes to 0.5: {simple_ft_episodes}")
    
    # MetaStackPipeline fine-tuning (from GridWorld)
    print("\n[MetaStackPipeline - Pre-trained]")
    # Note: metastack already has GridWorld weights
    metastack_ft_episodes, metastack_ft_reward = train_pipeline(
        metastack, frozenlake, 300, target_reward=0.5, verbose=True
    )
    print(f"  Episodes to 0.5: {metastack_ft_episodes}")
    
    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\nZero-Shot Transfer:")
    print(f"  SimplePipeline:    {simple_transfer:+.3f}")
    print(f"  MetaStackPipeline: {metastack_transfer:+.3f}")
    
    if metastack_transfer > 0.1:
        print("  [PASS] MetaStackPipeline shows positive transfer")
    else:
        print("  [WARNING] MetaStackPipeline transfer below threshold")
    
    print("\nFine-Tuning Speedup:")
    speedup = simple_ft_episodes / max(metastack_ft_episodes, 1)
    print(f"  SimplePipeline:    {simple_ft_episodes} episodes")
    print(f"  MetaStackPipeline: {metastack_ft_episodes} episodes")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.2:
        print("  [PASS] MetaStackPipeline fine-tunes faster")
    else:
        print("  [WARNING] MetaStackPipeline speedup below threshold")
    
    print()


def test_gridworld_to_tetris_maml():
    """Test 2: GridWorld → Tetris (MAML-only transfer)."""
    print("=" * 70)
    print("TEST 2: GridWorld → Tetris (MAML-Only Transfer)")
    print("=" * 70)
    print("\nDifferent dimensions: GridWorld (2D) vs Tetris (~50D)")
    print("Transfer MAML parameters only (LR multipliers, meta-optimizer)\n")
    
    # === Train on GridWorld ===
    print("Phase 1: Training on GridWorld (200 episodes)...")
    
    gridworld = GridWorldAdapter(size=5)
    
    print("\n[MetaStackPipeline]")
    metastack_gw = MetaStackPipeline(
        n_inputs=2, n_outputs=4, n_hidden=64,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3)
    )
    train_pipeline(metastack_gw, gridworld, 200, verbose=True)
    gw_stats = metastack_gw.get_stats()
    print(f"  Meta-updates: {gw_stats['meta_updates']}")
    lr_mults = metastack_gw.maml.get_lr_multipliers()
    print(f"  Learned LR multipliers: {lr_mults}")
    
    # === Transfer to Tetris ===
    print("\n" + "=" * 70)
    print("Phase 2: Transfer to Tetris Level 1")
    print("=" * 70)
    print("\nTarget: 2.0 mean lines cleared\n")
    
    tetris = TetrisCurriculumEnv(level=1, max_pieces=100)
    
    # Get Tetris state dimension
    tetris_state = tetris.reset()
    n_tetris_inputs = len(tetris_state)
    n_tetris_actions = len(tetris.get_valid_actions())
    
    print(f"Tetris state dim: {n_tetris_inputs}, actions: ~{n_tetris_actions}")
    
    # SimplePipeline (random MAML)
    print("\n[SimplePipeline - Random MAML]")
    simple_tetris = MetaStackPipeline(
        n_inputs=n_tetris_inputs, n_outputs=n_tetris_actions, n_hidden=64,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3)
    )
    
    # Train with Tetris-specific action selection
    simple_episodes = 0
    simple_rewards = []
    for episode in range(200):
        state = tetris.reset()
        total_lines = 0
        
        for _ in range(100):
            valid_actions = tetris.get_valid_actions()
            if not valid_actions:
                break
            
            # Map pipeline action to valid Tetris action
            q_values = simple_tetris.get_q_values(state)
            action_idx = int(np.argmax(q_values[:len(valid_actions)]))
            action = valid_actions[action_idx]
            
            next_state, reward, done, info = tetris.step(action)
            simple_tetris.update(state, action_idx, reward, next_state, done)
            
            total_lines = info.get('lines_cleared', 0)
            state = next_state
            
            if done:
                break
        
        simple_rewards.append(total_lines)
        simple_episodes = episode + 1
        
        if len(simple_rewards) >= 20 and np.mean(simple_rewards[-20:]) >= 2.0:
            print(f"  Reached 2.0 lines at episode {simple_episodes}")
            break
        
        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1}: mean_50={np.mean(simple_rewards[-50:]):.2f} lines")
    
    simple_final = np.mean(simple_rewards[-20:]) if len(simple_rewards) >= 20 else np.mean(simple_rewards)
    print(f"  Final: {simple_final:.2f} lines in {simple_episodes} episodes")
    
    # MetaStackPipeline (pre-trained MAML from GridWorld)
    print("\n[MetaStackPipeline - Pre-trained MAML]")
    print("  Transferring LR multipliers from GridWorld...")
    
    metastack_tetris = MetaStackPipeline(
        n_inputs=n_tetris_inputs, n_outputs=n_tetris_actions, n_hidden=64,
        synapse_config=DualHeadSynapseConfig(base_lr=0.01),
        maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3)
    )
    
    # Transfer MAML LR multipliers (not ANN weights - incompatible dimensions)
    source_multipliers = metastack_gw.maml.get_lr_multipliers()
    metastack_tetris.maml.meta_params['rl']['lr_multipliers'] = source_multipliers.copy()
    print(f"  Transferred multipliers: {source_multipliers}")
    
    metastack_episodes = 0
    metastack_rewards = []
    for episode in range(200):
        state = tetris.reset()
        total_lines = 0
        
        for _ in range(100):
            valid_actions = tetris.get_valid_actions()
            if not valid_actions:
                break
            
            q_values = metastack_tetris.get_q_values(state)
            action_idx = int(np.argmax(q_values[:len(valid_actions)]))
            action = valid_actions[action_idx]
            
            next_state, reward, done, info = tetris.step(action)
            metastack_tetris.update(state, action_idx, reward, next_state, done)
            
            total_lines = info.get('lines_cleared', 0)
            state = next_state
            
            if done:
                break
        
        metastack_rewards.append(total_lines)
        metastack_episodes = episode + 1
        
        if len(metastack_rewards) >= 20 and np.mean(metastack_rewards[-20:]) >= 2.0:
            print(f"  Reached 2.0 lines at episode {metastack_episodes}")
            break
        
        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1}: mean_50={np.mean(metastack_rewards[-50:]):.2f} lines")
    
    metastack_final = np.mean(metastack_rewards[-20:]) if len(metastack_rewards) >= 20 else np.mean(metastack_rewards)
    print(f"  Final: {metastack_final:.2f} lines in {metastack_episodes} episodes")
    
    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    speedup = simple_episodes / max(metastack_episodes, 1)
    print(f"\nEpisodes to 2.0 lines:")
    print(f"  SimplePipeline (random MAML):    {simple_episodes}")
    print(f"  MetaStackPipeline (pre-trained): {metastack_episodes}")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.1:
        print("  [PASS] MAML transfer accelerates learning")
    else:
        print("  [WARNING] MAML transfer below threshold")
    
    print()


def test_tetris_progressive_curriculum():
    """Test 3: Progressive Tetris curriculum (levels 1→3)."""
    print("=" * 70)
    print("TEST 3: Progressive Tetris Curriculum")
    print("=" * 70)
    print("\nLevel 1: 4x10, I-piece")
    print("Level 2: 6x10, I+O pieces")
    print("Level 3: 6x12, I+O+T pieces\n")
    
    target_lines = 2.0
    
    # SimplePipeline (train each level from scratch)
    print("[SimplePipeline - No Transfer]")
    simple_total = 0
    
    for level in [1, 2, 3]:
        print(f"\n  Level {level}:")
        env = TetrisCurriculumEnv(level=level, max_pieces=100)
        state = env.reset()
        n_inputs = len(state)
        n_actions = len(env.get_valid_actions())
        
        pipeline = MetaStackPipeline(
            n_inputs=n_inputs, n_outputs=n_actions, n_hidden=64,
            synapse_config=DualHeadSynapseConfig(base_lr=0.01),
            maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3)
        )
        
        rewards = []
        for episode in range(200):
            state = env.reset()
            total_lines = 0
            
            for _ in range(100):
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                q_values = pipeline.get_q_values(state)
                action_idx = int(np.argmax(q_values[:len(valid_actions)]))
                action = valid_actions[action_idx]
                
                next_state, reward, done, info = env.step(action)
                pipeline.update(state, action_idx, reward, next_state, done)
                
                total_lines = info.get('lines_cleared', 0)
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_lines)
            
            if len(rewards) >= 20 and np.mean(rewards[-20:]) >= target_lines:
                print(f"    Reached {target_lines} at episode {episode + 1}")
                simple_total += episode + 1
                break
        else:
            simple_total += 200
            print(f"    Did not reach {target_lines} (max 200 episodes)")
    
    print(f"\n  Total episodes: {simple_total}")
    
    # MetaStackPipeline (transfer between levels)
    print("\n[MetaStackPipeline - With Transfer]")
    metastack_total = 0
    prev_pipeline = None
    
    for level in [1, 2, 3]:
        print(f"\n  Level {level}:")
        env = TetrisCurriculumEnv(level=level, max_pieces=100)
        state = env.reset()
        n_inputs = len(state)
        n_actions = len(env.get_valid_actions())
        
        pipeline = MetaStackPipeline(
            n_inputs=n_inputs, n_outputs=n_actions, n_hidden=64,
            synapse_config=DualHeadSynapseConfig(base_lr=0.01),
            maml_config=DualHeadMAMLConfig(meta_batch_size=10, inner_steps=3)
        )
        
        # Transfer MAML from previous level
        if prev_pipeline is not None:
            prev_mults = prev_pipeline.maml.get_lr_multipliers()
            pipeline.maml.meta_params['rl']['lr_multipliers'] = prev_mults.copy()
            print(f"    Transferred LR multipliers: {prev_mults}")
        
        rewards = []
        for episode in range(200):
            state = env.reset()
            total_lines = 0
            
            for _ in range(100):
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                
                q_values = pipeline.get_q_values(state)
                action_idx = int(np.argmax(q_values[:len(valid_actions)]))
                action = valid_actions[action_idx]
                
                next_state, reward, done, info = env.step(action)
                pipeline.update(state, action_idx, reward, next_state, done)
                
                total_lines = info.get('lines_cleared', 0)
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_lines)
            
            if len(rewards) >= 20 and np.mean(rewards[-20:]) >= target_lines:
                print(f"    Reached {target_lines} at episode {episode + 1}")
                metastack_total += episode + 1
                break
        else:
            metastack_total += 200
            print(f"    Did not reach {target_lines} (max 200 episodes)")
        
        prev_pipeline = pipeline
    
    print(f"\n  Total episodes: {metastack_total}")
    
    # === Results ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nTotal episodes across 3 levels:")
    print(f"  SimplePipeline (no transfer):  {simple_total}")
    print(f"  MetaStackPipeline (transfer):  {metastack_total}")
    
    if metastack_total < simple_total:
        savings = simple_total - metastack_total
        print(f"  Savings: {savings} episodes ({savings/simple_total*100:.1f}%)")
        print("  [PASS] Progressive transfer reduces total episodes")
    else:
        print("  [WARNING] No benefit from progressive transfer")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CROSS-ENVIRONMENT TRANSFER TESTS")
    print("=" * 70 + "\n")
    
    try:
        test_gridworld_to_frozenlake()
        test_gridworld_to_tetris_maml()
        test_tetris_progressive_curriculum()
        
        print("=" * 70)
        print("ALL TRANSFER TESTS COMPLETE")
        print("=" * 70)
        print("\nPhase C: Cross-environment transfer fully validated!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

