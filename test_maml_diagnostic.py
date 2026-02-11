"""
MAML Diagnostic Audit — Find why MAML can't learn GridWorld

Tests:
1. Reward timing: Is reward arriving correctly? Are TD targets meaningful?
2. Meta-gradients: Are they non-zero? Are lr_multipliers changing?
3. Base Q-learning: Is Meta^0/1 actually learning the task at all?
4. Neuron capacity: Does scale matter (100 vs 500)?
5. Inner loop: Is MAML's inner loop adapting weights?
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from throng3.pipeline import MetaNPipeline
from throng3.envs.gridworld import GridWorld, create_gridworld_variants


def audit_1_reward_timing():
    """
    AUDIT 1: Is the reward signal reaching the right place at the right time?
    
    Checks:
    - Reward values during training
    - TD target quality (meaningful vs random)
    - Reward-action correlation
    """
    print("\n" + "="*70)
    print("AUDIT 1: Reward Timing & Signal Quality")
    print("="*70)
    
    variants = create_gridworld_variants()
    env = variants['empty_5x5']
    
    pipeline = MetaNPipeline.create_with_maml(
        n_inputs=25, n_outputs=4, meta_lr=0.001
    )
    
    rewards = []
    actions = []
    outputs_history = []
    
    for ep in range(50):
        state = env.reset()
        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 50:
            result = pipeline.step(state, reward=0.0)
            output = result['output']
            action = np.argmax(output)
            
            next_state, reward, done = env.step(action)
            pipeline.step(next_state, reward=reward)
            
            rewards.append(reward)
            actions.append(action)
            outputs_history.append(output.copy())
            
            ep_reward += reward
            state = next_state
            steps += 1
    
    rewards = np.array(rewards)
    actions = np.array(actions)
    outputs_arr = np.array(outputs_history)
    
    print(f"\n  Total steps: {len(rewards)}")
    print(f"  Reward stats: mean={rewards.mean():.4f}, std={rewards.std():.4f}")
    print(f"  Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
    print(f"  Non-zero rewards: {np.sum(rewards != 0)} / {len(rewards)} ({100*np.sum(rewards != 0)/len(rewards):.1f}%)")
    print(f"  Positive rewards: {np.sum(rewards > 0)} ({100*np.sum(rewards > 0)/len(rewards):.1f}%)")
    print(f"  Negative rewards: {np.sum(rewards < 0)} ({100*np.sum(rewards < 0)/len(rewards):.1f}%)")
    
    # Check output diversity
    print(f"\n  Output stats:")
    print(f"    Mean: {outputs_arr.mean():.6f}")
    print(f"    Std:  {outputs_arr.std():.6f}")
    print(f"    Range: [{outputs_arr.min():.6f}, {outputs_arr.max():.6f}]")
    
    # Check action distribution
    for a in range(4):
        pct = 100 * np.sum(actions == a) / len(actions)
        print(f"    Action {a}: {pct:.1f}%")
    
    # Check if outputs change over time (early vs late)
    early = outputs_arr[:100]
    late = outputs_arr[-100:]
    print(f"\n  Output evolution:")
    print(f"    Early mean: {early.mean():.6f}, std: {early.std():.6f}")
    print(f"    Late mean: {late.mean():.6f}, std: {late.std():.6f}")
    diff = np.mean(np.abs(late.mean(axis=0) - early.mean(axis=0)))
    print(f"    Mean abs change: {diff:.6f}")
    
    # TD target analysis
    print(f"\n  TD Target Quality:")
    td_targets = []
    for i in range(len(rewards) - 1):
        td = rewards[i] + 0.95 * np.max(outputs_arr[i + 1])
        td_targets.append(td)
    td_targets = np.array(td_targets)
    print(f"    TD target mean: {td_targets.mean():.6f}")
    print(f"    TD target std: {td_targets.std():.6f}")
    print(f"    TD target range: [{td_targets.min():.6f}, {td_targets.max():.6f}]")
    
    # Key diagnosis
    is_random = outputs_arr.std() < 1e-4
    no_learning = diff < 1e-5
    sparse_reward = np.sum(rewards != 0) / len(rewards) < 0.05
    
    if is_random:
        print("\n  ⚠ DIAGNOSIS: Outputs are near-constant → network not responding to input")
    if no_learning:
        print("  ⚠ DIAGNOSIS: Outputs don't change over time → no learning happening")
    if sparse_reward:
        print("  ⚠ DIAGNOSIS: Rewards are extremely sparse → hard to learn from")
    if not is_random and not no_learning:
        print("  ✓ Reward signal appears to be reaching the network")
    
    return {
        'is_random': is_random,
        'no_learning': no_learning,
        'sparse_reward': sparse_reward,
        'reward_mean': rewards.mean(),
        'output_std': outputs_arr.std(),
    }


def audit_2_meta_gradients():
    """
    AUDIT 2: Are meta-gradients non-zero? Are lr_multipliers changing?
    """
    print("\n" + "="*70)
    print("AUDIT 2: Meta-Gradient Analysis")
    print("="*70)
    
    variants = create_gridworld_variants()
    env = variants['empty_5x5']
    
    pipeline = MetaNPipeline.create_with_maml(
        n_inputs=25, n_outputs=4, meta_lr=0.001
    )
    maml = pipeline.stack.get_layer(3)
    
    # Check lr_multipliers before
    print("\n  Before training:")
    for task_type in ['supervised', 'rl']:
        mults = maml.meta_params[task_type]['lr_multipliers']
        print(f"    {task_type} lr_multipliers: {'empty' if not mults else f'{len(mults)} keys'}")
    print(f"    meta_updates: {maml.meta_updates}")
    
    # Train 100 episodes
    for ep in range(100):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            result = pipeline.step(state, reward=0.0)
            action = np.argmax(result['output'])
            next_state, reward, done = env.step(action)
            pipeline.step(next_state, reward=reward)
            state = next_state
            steps += 1
    
    # Check lr_multipliers after training (before meta-update)
    print("\n  After 100 episodes (before meta-update):")
    for task_type in ['supervised', 'rl']:
        mults = maml.meta_params[task_type]['lr_multipliers']
        if mults:
            for name, m in mults.items():
                print(f"    {task_type}/{name}: shape={m.shape}, mean={m.mean():.4f}, std={m.std():.6f}")
        else:
            print(f"    {task_type}: empty")
    
    # Now trigger consolidation
    print(f"\n  Experience buffer size: {len(pipeline._experience_buffer)}")
    
    # Snapshot lr_multipliers before meta-update
    pre_update = {}
    for task_type in ['supervised', 'rl']:
        mults = maml.meta_params[task_type]['lr_multipliers']
        pre_update[task_type] = {k: v.copy() for k, v in mults.items()}
    
    pipeline.consolidate_maml_task()
    
    print(f"\n  After meta-update:")
    print(f"    meta_updates: {maml.meta_updates}")
    
    for task_type in ['supervised', 'rl']:
        mults = maml.meta_params[task_type]['lr_multipliers']
        if mults:
            for name, m in mults.items():
                pre = pre_update[task_type].get(name)
                if pre is not None and pre.shape == m.shape:
                    change = np.linalg.norm(m - pre)
                    print(f"    {task_type}/{name}: mean={m.mean():.4f}, std={m.std():.6f}, L2 change={change:.6f}")
                else:
                    print(f"    {task_type}/{name}: mean={m.mean():.4f}, std={m.std():.6f}, (newly initialized)")
        else:
            print(f"    {task_type}: empty")
    
    updated = maml.meta_updates > 0
    if updated:
        print("\n  ✓ Meta-update executed")
    else:
        print("\n  ⚠ DIAGNOSIS: No meta-updates occurred!")
    
    return {'updated': updated, 'meta_updates': maml.meta_updates}


def audit_3_base_learning():
    """
    AUDIT 3: Is the base Q-learner (Meta^0 + Meta^1) even learning the task?
    Compare EWC pipeline vs MAML pipeline raw training quality.
    """
    print("\n" + "="*70)
    print("AUDIT 3: Base Q-Learning Quality")
    print("="*70)
    
    variants = create_gridworld_variants()
    env = variants['empty_5x5']
    np.random.seed(42)
    
    configs = {
        'EWC': MetaNPipeline.create_with_ewc(n_inputs=25, n_outputs=4, ewc_lambda=1000.0),
        'MAML': MetaNPipeline.create_with_maml(n_inputs=25, n_outputs=4, meta_lr=0.001),
    }
    
    for name, pipeline in configs.items():
        print(f"\n  [{name} Pipeline]")
        
        # Check synapse config
        synapse = pipeline.stack.get_layer(1)
        print(f"    Learning mode: {synapse.learning_mode}")
        print(f"    Active rule: {synapse.active_rule}")
        print(f"    Q-learning: {synapse.synapse_config.use_qlearning}")
        print(f"    Dopamine modulation: {synapse.synapse_config.dopamine_modulation}")
        
        epoch_rewards = []
        for ep in range(200):
            state = env.reset()
            ep_reward = 0
            done = False
            steps = 0
            epsilon = max(0.1, 1.0 - ep * 0.005)
            
            while not done and steps < 50:
                result = pipeline.step(state, reward=0.0)
                output = result['output']
                if np.random.random() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(output)
                
                next_state, reward, done = env.step(action)
                pipeline.step(next_state, reward=reward)
                
                ep_reward += reward
                state = next_state
                steps += 1
            
            epoch_rewards.append(ep_reward)
        
        # Report
        first_50 = np.mean(epoch_rewards[:50])
        last_50 = np.mean(epoch_rewards[-50:])
        best_50 = max(np.mean(epoch_rewards[i:i+50]) for i in range(0, 150, 10))
        
        print(f"    First 50 eps: {first_50:.3f}")
        print(f"    Last 50 eps: {last_50:.3f}")
        print(f"    Best 50-ep window: {best_50:.3f}")
        print(f"    Improvement: {last_50 - first_50:+.3f}")
        
        if last_50 > first_50 + 0.1:
            print(f"    ✓ Learning detected")
        else:
            print(f"    ⚠ No significant learning improvement")


def audit_4_neuron_capacity():
    """
    AUDIT 4: Does neuron scale matter? 100 vs 500 neurons.
    """
    print("\n" + "="*70)
    print("AUDIT 4: Neuron Capacity (100 vs 500)")
    print("="*70)
    
    variants = create_gridworld_variants()
    env = variants['empty_5x5']
    
    for n_neurons in [100, 500]:
        np.random.seed(42)
        print(f"\n  [{n_neurons} neurons]")
        
        pipeline = MetaNPipeline.create_with_ewc(
            n_neurons=n_neurons, n_inputs=25, n_outputs=4, ewc_lambda=1000.0
        )
        
        epoch_rewards = []
        for ep in range(200):
            state = env.reset()
            ep_reward = 0
            done = False
            steps = 0
            epsilon = max(0.1, 1.0 - ep * 0.005)
            
            while not done and steps < 50:
                result = pipeline.step(state, reward=0.0)
                output = result['output']
                if np.random.random() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(output)
                
                next_state, reward, done = env.step(action)
                pipeline.step(next_state, reward=reward)
                
                ep_reward += reward
                state = next_state
                steps += 1
            
            epoch_rewards.append(ep_reward)
        
        first_50 = np.mean(epoch_rewards[:50])
        last_50 = np.mean(epoch_rewards[-50:])
        
        print(f"    First 50 eps: {first_50:.3f}")
        print(f"    Last 50 eps: {last_50:.3f}")
        print(f"    Improvement: {last_50 - first_50:+.3f}")


def audit_5_inner_loop():
    """
    AUDIT 5: Is MAML's inner loop actually adapting parameters?
    """
    print("\n" + "="*70)
    print("AUDIT 5: Inner Loop Adaptation Quality")
    print("="*70)
    
    from throng3.layers.meta3_maml import TaskConditionedMAML
    from throng3.config.maml_config import MAMLConfig
    
    maml = TaskConditionedMAML(MAMLConfig(
        inner_lr=0.01, inner_steps=1, meta_lr=0.001
    ))
    
    # Create synthetic support set (known-good data)
    np.random.seed(42)
    W = np.random.randn(4, 25) * 0.1
    
    # Generate (state, target) pairs with structure
    support_set = []
    for _ in range(20):
        state = np.zeros(25)
        pos = np.random.randint(25)
        state[pos] = 1.0  # One-hot position
        target = np.zeros(4)
        target[pos % 4] = 1.0  # Target action based on position
        support_set.append((state, target))
    
    # Run inner loop
    initial_params = {'W_out': W.copy()}
    adapted = maml.inner_loop(initial_params, support_set, inner_lr=0.01, inner_steps=1)
    
    change_1step = np.linalg.norm(adapted['W_out'] - W)
    print(f"\n  1-step inner loop:")
    print(f"    Weight change (L2): {change_1step:.6f}")
    
    # Try more steps
    adapted_5 = maml.inner_loop(initial_params, support_set, inner_lr=0.01, inner_steps=5)
    change_5step = np.linalg.norm(adapted_5['W_out'] - W)
    print(f"  5-step inner loop:")
    print(f"    Weight change (L2): {change_5step:.6f}")
    
    adapted_10 = maml.inner_loop(initial_params, support_set, inner_lr=0.01, inner_steps=10)
    change_10step = np.linalg.norm(adapted_10['W_out'] - W)
    print(f"  10-step inner loop:")
    print(f"    Weight change (L2): {change_10step:.6f}")
    
    # Check if inner loop actually improves loss
    def compute_loss(params, data):
        total = 0
        for x, y in data:
            out = params['W_out'] @ x
            total += np.mean((out - y) ** 2)
        return total / len(data)
    
    loss_before = compute_loss(initial_params, support_set)
    loss_1step = compute_loss(adapted, support_set)
    loss_5step = compute_loss(adapted_5, support_set)
    loss_10step = compute_loss(adapted_10, support_set)
    
    print(f"\n  Loss reduction:")
    print(f"    Before: {loss_before:.6f}")
    print(f"    1-step: {loss_1step:.6f} ({loss_1step/loss_before*100:.1f}%)")
    print(f"    5-step: {loss_5step:.6f} ({loss_5step/loss_before*100:.1f}%)")
    print(f"    10-step: {loss_10step:.6f} ({loss_10step/loss_before*100:.1f}%)")
    
    if change_1step > 1e-6:
        print(f"\n  ✓ Inner loop adapts weights")
    else:
        print(f"\n  ⚠ Inner loop produces no weight change!")
    
    if loss_1step < loss_before:
        print(f"  ✓ Inner loop reduces loss")
    else:
        print(f"  ⚠ Inner loop doesn't reduce loss!")
    
    # Now test with RL-style data (TD targets)
    print(f"\n  RL-style data (TD targets):")
    rl_support = []
    for _ in range(20):
        state = np.random.rand(25)
        # TD target: mostly same as Q-values but with one action updated
        current_q = W @ state
        td_target = current_q.copy()
        action = np.argmax(current_q)
        td_target[action] = 0.5 + 0.95 * np.max(current_q)  # Simulated TD update
        rl_support.append((state, td_target))
    
    rl_adapted = maml.inner_loop({'W_out': W.copy()}, rl_support, inner_lr=0.01, inner_steps=5)
    rl_change = np.linalg.norm(rl_adapted['W_out'] - W)
    rl_loss_before = compute_loss({'W_out': W}, rl_support)
    rl_loss_after = compute_loss(rl_adapted, rl_support)
    
    print(f"    Weight change: {rl_change:.6f}")
    print(f"    Loss before: {rl_loss_before:.6f}")
    print(f"    Loss after: {rl_loss_after:.6f}")
    
    if rl_loss_after < rl_loss_before:
        print(f"    ✓ Inner loop reduces RL loss")
    else:
        print(f"    ⚠ Inner loop doesn't reduce RL loss!")


if __name__ == '__main__':
    print("="*70)
    print("MAML DIAGNOSTIC AUDIT")
    print("Why doesn't MAML learn GridWorld?")
    print("="*70)
    
    r1 = audit_1_reward_timing()
    r2 = audit_2_meta_gradients()
    audit_3_base_learning()
    audit_4_neuron_capacity()
    audit_5_inner_loop()
    
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\n  Reward signal:", "BROKEN" if r1['sparse_reward'] else "OK")
    print("  Network responsiveness:", "DEAD" if r1['is_random'] else "OK")
    print("  Learning detected:", "NO" if r1['no_learning'] else "YES")
    print("  Meta-updates executed:", "YES" if r2['updated'] else "NO")
    
    if r1['is_random']:
        print("\n  ROOT CAUSE: Network outputs are constant — not responding to input")
        print("  FIX: Check weight initialization, activation function, or neuron scale")
    elif r1['no_learning']:
        print("\n  ROOT CAUSE: Network doesn't learn even with reward")
        print("  FIX: Check synapse optimizer, dopamine modulation, learning rate")
    elif not r2['updated']:
        print("\n  ROOT CAUSE: Meta-updates aren't executing")
        print("  FIX: Check consolidate_maml_task flow")
    else:
        print("\n  ROOT CAUSE: Meta-updates execute but don't help")
        print("  FIX: Need better inner loop, more steps, or different architecture")
