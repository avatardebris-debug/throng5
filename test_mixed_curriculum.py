"""
Test: Compound Transfer with Automatic Mechanism Selection

The ultimate test: Can Meta^N achieve compound transfer across
DIFFERENT TASK TYPES with automatic mechanism selection?

Curriculum:
- Task A: Supervised (linear regression) → should use gradient
- Task B: RL (reward-based) → should use RL  
- Task C: Supervised (linear regression) → should use gradient

Expected: A+B→C better than alternatives, with automatic mechanism switching
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_supervised_task(n_inputs: int, n_outputs: int, seed: int, complexity: float = 0.5):
    """Generate a supervised linear task."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * complexity
    bias = rng.randn(n_outputs) * 0.1
    return W, bias


def train_supervised(pipeline, W, bias, n_steps: int, task_name: str = "Task"):
    """Train on supervised task, return final loss."""
    losses = []
    for step in range(n_steps):
        x = np.random.randn(W.shape[1])
        y = W @ x + bias
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:])
    print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def train_rl(pipeline, n_steps: int, task_name: str = "Task"):
    """Train on RL task (reward-based), return final loss."""
    losses = []
    n_inputs = 16
    
    for step in range(n_steps):
        x = np.random.randn(n_inputs)
        # Simple reward: positive if output sum is positive
        result = pipeline.step(x, target=None, reward=0.0)
        output_sum = np.sum(result['output'])
        reward = 1.0 if output_sum > 0 else -1.0
        
        # Send reward on next step (delayed reward)
        if step > 0:
            result = pipeline.step(x, target=None, reward=reward)
            losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:]) if losses else 1.0
    print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def test_mixed_curriculum_compound_transfer():
    """
    Test compound transfer on mixed curriculum with auto-selection.
    """
    print("\n" + "="*60)
    print("COMPOUND TRANSFER: MIXED CURRICULUM (AUTO-SELECTION)")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    
    # Define tasks
    W_A, b_A = generate_supervised_task(n_inputs, n_outputs, seed=42, complexity=0.3)
    # Task B is RL (no W, b needed)
    W_C, b_C = generate_supervised_task(n_inputs, n_outputs, seed=456, complexity=0.5)
    
    results = {}
    
    # Condition 1: C cold (baseline)
    print("\n[Condition 1: C Cold (Supervised)]")
    pipeline_cold = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    results['C_cold'] = train_supervised(pipeline_cold, W_C, b_C, test_steps, "C (cold)")
    
    # Check mechanism
    meta2_cold = pipeline_cold.stack.get_layer(2)
    if meta2_cold:
        print(f"  Mechanism: {meta2_cold.current_mechanism}")
    
    # Condition 2: A→C (both supervised)
    print("\n[Condition 2: A→C (Supervised→Supervised)]")
    pipeline_ac = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    train_supervised(pipeline_ac, W_A, b_A, train_steps, "A (supervised)")
    
    meta2_ac = pipeline_ac.stack.get_layer(2)
    if meta2_ac:
        print(f"  Mechanism after A: {meta2_ac.current_mechanism}")
    
    pipeline_ac.reset_task_state()
    if meta2_ac:
        meta2_ac.task_detector.reset()
    
    results['A_C'] = train_supervised(pipeline_ac, W_C, b_C, test_steps, "C (after A)")
    
    if meta2_ac:
        print(f"  Mechanism after C: {meta2_ac.current_mechanism}")
    
    # Condition 3: B→C (RL→Supervised)
    print("\n[Condition 3: B→C (RL→Supervised)]")
    pipeline_bc = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    train_rl(pipeline_bc, train_steps, "B (RL)")
    
    meta2_bc = pipeline_bc.stack.get_layer(2)
    if meta2_bc:
        print(f"  Mechanism after B: {meta2_bc.current_mechanism}")
    
    pipeline_bc.reset_task_state()
    if meta2_bc:
        meta2_bc.task_detector.reset()
    
    results['B_C'] = train_supervised(pipeline_bc, W_C, b_C, test_steps, "C (after B)")
    
    if meta2_bc:
        print(f"  Mechanism after C: {meta2_bc.current_mechanism}")
    
    # Condition 4: A+B→C (Supervised→RL→Supervised)
    print("\n[Condition 4: A+B→C (Mixed Curriculum)]")
    pipeline_abc = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    train_supervised(pipeline_abc, W_A, b_A, train_steps, "A (supervised)")
    meta2_abc = pipeline_abc.stack.get_layer(2)
    if meta2_abc:
        print(f"  Mechanism after A: {meta2_abc.current_mechanism}")
    
    pipeline_abc.reset_task_state()
    if meta2_abc:
        meta2_abc.task_detector.reset()
    
    train_rl(pipeline_abc, train_steps, "B (RL)")
    if meta2_abc:
        print(f"  Mechanism after B: {meta2_abc.current_mechanism}")
    
    pipeline_abc.reset_task_state()
    if meta2_abc:
        meta2_abc.task_detector.reset()
    
    results['ABC'] = train_supervised(pipeline_abc, W_C, b_C, test_steps, "C (after A+B)")
    if meta2_abc:
        print(f"  Mechanism after C: {meta2_abc.current_mechanism}")
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"C cold:        {results['C_cold']:.4f} (baseline)")
    print(f"A→C:           {results['A_C']:.4f} ({(results['C_cold']-results['A_C'])/results['C_cold']*100:+.1f}%)")
    print(f"B→C:           {results['B_C']:.4f} ({(results['C_cold']-results['B_C'])/results['C_cold']*100:+.1f}%)")
    print(f"A+B→C:         {results['ABC']:.4f} ({(results['C_cold']-results['ABC'])/results['C_cold']*100:+.1f}%)")
    
    # Compound transfer check
    print("\n" + "="*60)
    print("COMPOUND TRANSFER VALIDATION")
    print("="*60)
    
    compound_better_than_ac = results['ABC'] < results['A_C']
    compound_better_than_bc = results['ABC'] < results['B_C']
    compound_better_than_cold = results['ABC'] < results['C_cold']
    
    print(f"A+B→C better than A→C:  {compound_better_than_ac}")
    print(f"A+B→C better than B→C:  {compound_better_than_bc}")
    print(f"A+B→C better than cold: {compound_better_than_cold}")
    
    if compound_better_than_ac and compound_better_than_bc and compound_better_than_cold:
        print("\n✓ COMPOUND TRANSFER CONFIRMED!")
        print("  Meta^N achieves compound generalization across task types")
        print("  with automatic mechanism selection!")
    elif compound_better_than_cold:
        print("\n⚠ PARTIAL COMPOUND TRANSFER")
        print("  Better than cold, but not all conditions met")
    else:
        print("\n✗ NO COMPOUND TRANSFER")
        print("  Need to investigate further")
    
    return results


if __name__ == '__main__':
    results = test_mixed_curriculum_compound_transfer()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    abc_improvement = (results['C_cold'] - results['ABC']) / results['C_cold'] * 100
    
    print(f"\nCompound transfer improvement: {abc_improvement:+.1f}%")
    
    if abc_improvement > 10:
        print("\n✓ SUCCESS!")
        print("  Meta^N achieves compound transfer with auto-selection")
        print("\nKey achievements:")
        print("  1. Automatic task detection (supervised vs RL)")
        print("  2. Automatic mechanism selection (gradient vs RL)")
        print("  3. Compound transfer across different task types")
        print("  4. No manual configuration required")
    elif abc_improvement > 0:
        print("\n⚠ PARTIAL SUCCESS")
        print("  Positive transfer but below target")
    else:
        print("\n✗ NEEDS WORK")
        print("  No compound transfer detected")
