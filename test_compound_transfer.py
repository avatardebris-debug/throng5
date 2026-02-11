"""
Test: Compound Transfer Learning (A+B→C)

Tests the hypothesis that learning Task A, then Task B, then Task C
results in faster learning of C compared to:
- Learning C cold (no prior tasks)
- Learning A then C (skipping B)
- Learning B then C (skipping A)

This validates whether Meta^N achieves COMPOUND generalization,
not just basic transfer.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_task(n_inputs: int, n_outputs: int, seed: int, complexity: float = 0.5):
    """Generate a linear task with specific random seed and complexity."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * complexity
    bias = rng.randn(n_outputs) * 0.1
    return W, bias


def train_on_task(pipeline, W, bias, n_steps: int, task_name: str = "Task"):
    """Train pipeline on a task, return final loss."""
    losses = []
    for step in range(n_steps):
        x = np.random.randn(W.shape[1])
        y = W @ x + bias
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:])
    print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def test_compound_transfer():
    """Test A+B→C vs alternatives."""
    print("\n" + "="*60)
    print("COMPOUND TRANSFER TEST: A+B→C")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    
    # Define tasks with increasing complexity
    W_A, b_A = generate_task(n_inputs, n_outputs, seed=42, complexity=0.3)
    W_B, b_B = generate_task(n_inputs, n_outputs, seed=123, complexity=0.4)
    W_C, b_C = generate_task(n_inputs, n_outputs, seed=456, complexity=0.5)
    
    results = {}
    
    # Condition 1: C cold (baseline)
    print("\n[Condition 1: C Cold]")
    pipeline_cold = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    results['C_cold'] = train_on_task(pipeline_cold, W_C, b_C, test_steps, "C (cold)")
    
    # Condition 2: A→C
    print("\n[Condition 2: A→C]")
    pipeline_ac = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    train_on_task(pipeline_ac, W_A, b_A, train_steps, "A")
    pipeline_ac.reset_task_state()
    results['A_C'] = train_on_task(pipeline_ac, W_C, b_C, test_steps, "C (after A)")
    
    # Condition 3: B→C
    print("\n[Condition 3: B→C]")
    pipeline_bc = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    train_on_task(pipeline_bc, W_B, b_B, train_steps, "B")
    pipeline_bc.reset_task_state()
    results['B_C'] = train_on_task(pipeline_bc, W_C, b_C, test_steps, "C (after B)")
    
    # Condition 4: A+B→C (compound transfer)
    print("\n[Condition 4: A+B→C (Compound)]")
    pipeline_abc = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    train_on_task(pipeline_abc, W_A, b_A, train_steps, "A")
    pipeline_abc.reset_task_state()
    train_on_task(pipeline_abc, W_B, b_B, train_steps, "B")
    pipeline_abc.reset_task_state()
    results['ABC'] = train_on_task(pipeline_abc, W_C, b_C, test_steps, "C (after A+B)")
    
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
    
    print(f"A+B→C better than A→C:  {compound_better_than_ac} ({results['ABC']:.4f} vs {results['A_C']:.4f})")
    print(f"A+B→C better than B→C:  {compound_better_than_bc} ({results['ABC']:.4f} vs {results['B_C']:.4f})")
    print(f"A+B→C better than cold: {compound_better_than_cold} ({results['ABC']:.4f} vs {results['C_cold']:.4f})")
    
    if compound_better_than_ac and compound_better_than_bc and compound_better_than_cold:
        print("\n✓ COMPOUND TRANSFER CONFIRMED!")
        print("  Meta^N demonstrates compound generalization across curriculum.")
    else:
        print("\n✗ COMPOUND TRANSFER NOT CONFIRMED")
        print("  A+B→C does not outperform all alternatives.")
    
    return results


def test_superexponential_learning():
    """Test if learning accelerates across a 5-task curriculum."""
    print("\n" + "="*60)
    print("SUPEREXPONENTIAL LEARNING TEST")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    n_tasks = 5
    train_steps = 100
    
    pipeline = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    steps_to_converge = []
    
    for i in range(n_tasks):
        W, b = generate_task(n_inputs, n_outputs, seed=100+i, complexity=0.4)
        
        print(f"\n[Task {i+1}/{n_tasks}]")
        losses = []
        for step in range(train_steps):
            x = np.random.randn(n_inputs)
            y = W @ x + b
            result = pipeline.step(x, target=y, reward=0.0)
            losses.append(result['loss'])
            
            # Check convergence (loss < threshold)
            if len(losses) >= 20 and np.mean(losses[-20:]) < 2.0:
                steps_to_converge.append(step + 1)
                print(f"  Converged at step {step+1}, loss={np.mean(losses[-20:]):.4f}")
                break
        else:
            steps_to_converge.append(train_steps)
            print(f"  Did not converge, final loss={np.mean(losses[-20:]):.4f}")
        
        pipeline.reset_task_state()
    
    # Analysis
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS")
    print("="*60)
    for i, steps in enumerate(steps_to_converge):
        speedup = steps_to_converge[0] / steps if steps > 0 else 0
        print(f"Task {i+1}: {steps:3d} steps (speedup: {speedup:.2f}x)")
    
    # Check for acceleration
    if len(steps_to_converge) >= 3:
        improving = all(steps_to_converge[i] >= steps_to_converge[i+1] 
                       for i in range(len(steps_to_converge)-1))
        
        if improving:
            final_speedup = steps_to_converge[0] / steps_to_converge[-1]
            print(f"\n✓ LEARNING ACCELERATES! Final speedup: {final_speedup:.2f}x")
            
            if final_speedup > 3.0:
                print("  → SUPEREXPONENTIAL learning detected!")
        else:
            print("\n✗ Learning does not consistently accelerate")
    
    return steps_to_converge


if __name__ == '__main__':
    # Run compound transfer test
    results = test_compound_transfer()
    
    # Run superexponential learning test
    learning_curve = test_superexponential_learning()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
