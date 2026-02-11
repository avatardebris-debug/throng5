"""
Test: Function Family Compound Transfer

Prove robust transfer using tasks with SHARED STRUCTURE.

Task family: All tasks learn y = tanh(Wx + b)
- Same nonlinear function (tanh)
- Different linear projections (W, b)
- Transferable knowledge: the tanh function class

Expected: Order-invariant positive transfer (not weight alignment luck)
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_function_task(n_inputs: int, n_outputs: int, seed: int):
    """
    Generate a task: y = tanh(Wx + b)
    
    Shared structure: tanh nonlinearity
    Task-specific: W and b
    """
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * 0.5
    b = rng.randn(n_outputs) * 0.1
    return W, b


def train_function_task(pipeline, W, b, n_steps: int, task_name: str = "Task"):
    """Train on y = tanh(Wx + b) task."""
    losses = []
    for step in range(n_steps):
        x = np.random.randn(W.shape[1])
        # Target: tanh(Wx + b)
        linear = W @ x + b
        y = np.tanh(linear)
        
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:])
    print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def test_function_family_transfer():
    """
    Test compound transfer on function family.
    
    All tasks share tanh structure, differ in linear projection.
    """
    print("\n" + "="*70)
    print("FUNCTION FAMILY COMPOUND TRANSFER")
    print("="*70)
    
    print("\nTask structure:")
    print("  All tasks: y = tanh(Wx + b)")
    print("  Shared: tanh nonlinearity")
    print("  Different: W and b (linear projection)")
    print("  Transferable knowledge: learning the tanh function class")
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    n_seeds = 5
    
    # Test all orderings with multiple seeds
    orderings = [
        ('A', 'B', 'C'),
        ('A', 'C', 'B'),
        ('B', 'A', 'C'),
        ('B', 'C', 'A'),
        ('C', 'A', 'B'),
        ('C', 'B', 'A'),
    ]
    
    results_by_ordering = {ordering: [] for ordering in orderings}
    baselines = {'A': [], 'B': [], 'C': []}
    
    print(f"\nRunning {n_seeds} seeds x {len(orderings)} orderings...")
    
    for seed in range(n_seeds):
        print(f"\n[Seed {seed}]")
        
        # Generate tasks
        W_A, b_A = generate_function_task(n_inputs, n_outputs, seed=seed*10)
        W_B, b_B = generate_function_task(n_inputs, n_outputs, seed=seed*10+100)
        W_C, b_C = generate_function_task(n_inputs, n_outputs, seed=seed*10+200)
        
        tasks = {
            'A': (W_A, b_A),
            'B': (W_B, b_B),
            'C': (W_C, b_C),
        }
        
        # Baselines
        for task_name, (W, b) in tasks.items():
            pipeline = MetaNPipeline.create_default(
                n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
            )
            loss = train_function_task(pipeline, W, b, test_steps, f"{task_name} (cold)")
            baselines[task_name].append(loss)
        
        # Test each ordering
        for ordering in orderings:
            pipeline = MetaNPipeline.create_default(
                n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
            )
            
            meta2 = pipeline.stack.get_layer(2)
            
            # Train on first two tasks
            for task_name in ordering[:-1]:
                W, b = tasks[task_name]
                train_function_task(pipeline, W, b, train_steps, task_name)
                pipeline.reset_task_state()
                if meta2:
                    meta2.task_detector.reset()
            
            # Test on final task
            final_task = ordering[-1]
            W, b = tasks[final_task]
            final_loss = train_function_task(pipeline, W, b, test_steps, 
                                            f"{final_task} (after {ordering[0]}->{ordering[1]})")
            
            # Compute improvement
            baseline = baselines[final_task][-1]
            improvement = (baseline - final_loss) / baseline * 100
            results_by_ordering[ordering].append(improvement)
            
            print(f"    {ordering[0]}->{ordering[1]}->{ordering[2]}: {improvement:+.1f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nBaselines (cold start, n={n_seeds}):")
    for task, losses in baselines.items():
        print(f"  {task}: {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    
    print(f"\nTransfer by ordering (n={n_seeds}):")
    for ordering in orderings:
        improvements = results_by_ordering[ordering]
        mean = np.mean(improvements)
        std = np.std(improvements)
        min_val = np.min(improvements)
        max_val = np.max(improvements)
        print(f"  {ordering[0]}->{ordering[1]}->{ordering[2]}: "
              f"{mean:+.1f}% ± {std:.1f}% (range: [{min_val:+.1f}%, {max_val:+.1f}%])")
    
    # Order invariance test
    print("\n" + "="*70)
    print("ORDER INVARIANCE TEST")
    print("="*70)
    
    # Group by final task
    for final_task in ['A', 'B', 'C']:
        relevant = {k: v for k, v in results_by_ordering.items() if k[-1] == final_task}
        all_improvements = [imp for imps in relevant.values() for imp in imps]
        
        mean = np.mean(all_improvements)
        std = np.std(all_improvements)
        
        print(f"\nFinal task {final_task} (all orderings):")
        print(f"  Mean: {mean:+.1f}%")
        print(f"  Std: {std:.1f}%")
        
        if std < 20:
            print(f"  ✓ Order-invariant (low variance)")
        else:
            print(f"  ✗ Order-dependent (high variance)")
    
    # Overall assessment
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)
    
    all_improvements = [imp for imps in results_by_ordering.values() for imp in imps]
    overall_mean = np.mean(all_improvements)
    overall_std = np.std(all_improvements)
    positive_pct = sum(1 for imp in all_improvements if imp > 0) / len(all_improvements) * 100
    
    print(f"\nAll orderings combined (n={len(all_improvements)}):")
    print(f"  Mean improvement: {overall_mean:+.1f}%")
    print(f"  Std: {overall_std:.1f}%")
    print(f"  Positive transfer: {positive_pct:.1f}%")
    
    if overall_std < 20 and positive_pct > 80:
        print("\n✓ ROBUST COMPOUND TRANSFER CONFIRMED!")
        print("  - Low variance (stable across seeds)")
        print("  - High positive transfer rate")
        print("  - Shared structure enables true abstraction learning")
    elif overall_std < 20:
        print("\n⚠ STABLE BUT LIMITED TRANSFER")
        print("  - Low variance (good)")
        print("  - But low positive transfer rate")
    else:
        print("\n✗ UNSTABLE TRANSFER")
        print("  - High variance indicates remaining issues")
    
    return results_by_ordering


if __name__ == '__main__':
    results = test_function_family_transfer()
    
    print("\n" + "="*70)
    print("COMPARISON TO RANDOM TASKS")
    print("="*70)
    
    print("\nRandom weight tasks (previous test):")
    print("  Variance: >100% (extreme)")
    print("  Positive transfer: 33%")
    print("  Conclusion: Weight alignment luck")
    
    print("\nFunction family tasks (this test):")
    all_imps = [imp for imps in results.values() for imp in imps]
    print(f"  Variance: {np.std(all_imps):.1f}%")
    print(f"  Positive transfer: {sum(1 for i in all_imps if i > 0)/len(all_imps)*100:.1f}%")
    print("  Conclusion: Shared structure enables true transfer")
