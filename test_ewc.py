"""
Test: EWC (Elastic Weight Consolidation) for Compound Transfer

Tests if Meta^3 weight consolidation prevents catastrophic interference
and enables robust compound transfer.

Expected: Function family tasks with EWC should show:
- Mean transfer: +20% to +40% (vs -10% without EWC)
- Variance: <15% (vs 18.6% without EWC)
- Positive rate: >80% (vs 30% without EWC)
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_function_task(n_inputs: int, n_outputs: int, seed: int):
    """Generate y = tanh(Wx + b) task."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * 0.5
    b = rng.randn(n_outputs) * 0.1
    return W, b


def train_function_task(pipeline, W, b, n_steps: int, task_name: str = "Task"):
    """Train on y = tanh(Wx + b) task."""
    losses = []
    for step in range(n_steps):
        x = np.random.randn(W.shape[1])
        linear = W @ x + b
        y = np.tanh(linear)
        
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:])
    print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def test_ewc_compound_transfer():
    """
    Test EWC on function family compound transfer.
    
    Compare with/without EWC to validate catastrophic interference prevention.
    """
    print("\n" + "="*70)
    print("EWC COMPOUND TRANSFER TEST")
    print("="*70)
    
    print("\nTask: Function family (y = tanh(Wx + b))")
    print("Ordering: A -> B -> C")
    print("Seeds: 5")
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    n_seeds = 5
    
    results_without_ewc = []
    results_with_ewc = []
    baselines = []
    
    for seed in range(n_seeds):
        print(f"\n[Seed {seed}]")
        
        # Generate tasks
        W_A, b_A = generate_function_task(n_inputs, n_outputs, seed=seed*10)
        W_B, b_B = generate_function_task(n_inputs, n_outputs, seed=seed*10+100)
        W_C, b_C = generate_function_task(n_inputs, n_outputs, seed=seed*10+200)
        
        # Baseline: C cold
        pipeline_cold = MetaNPipeline.create_default(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )
        baseline = train_function_task(pipeline_cold, W_C, b_C, test_steps, "C (cold)")
        baselines.append(baseline)
        
        # WITHOUT EWC
        print("\n  Without EWC:")
        pipeline_no_ewc = MetaNPipeline.create_default(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )
        
        train_function_task(pipeline_no_ewc, W_A, b_A, train_steps, "  A")
        pipeline_no_ewc.reset_task_state()
        
        train_function_task(pipeline_no_ewc, W_B, b_B, train_steps, "  B")
        pipeline_no_ewc.reset_task_state()
        
        loss_no_ewc = train_function_task(pipeline_no_ewc, W_C, b_C, test_steps, "  C")
        improvement_no_ewc = (baseline - loss_no_ewc) / baseline * 100
        results_without_ewc.append(improvement_no_ewc)
        print(f"    Improvement: {improvement_no_ewc:+.1f}%")
        
        # WITH EWC
        print("\n  With EWC:")
        pipeline_ewc = MetaNPipeline.create_with_ewc(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs,
            ewc_lambda=1000.0
        )
        
        train_function_task(pipeline_ewc, W_A, b_A, train_steps, "  A")
        pipeline_ewc.consolidate_task()  # Consolidate after A
        pipeline_ewc.reset_task_state()
        
        train_function_task(pipeline_ewc, W_B, b_B, train_steps, "  B")
        pipeline_ewc.consolidate_task()  # Consolidate after B
        pipeline_ewc.reset_task_state()
        
        loss_ewc = train_function_task(pipeline_ewc, W_C, b_C, test_steps, "  C")
        improvement_ewc = (baseline - loss_ewc) / baseline * 100
        results_with_ewc.append(improvement_ewc)
        print(f"    Improvement: {improvement_ewc:+.1f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    mean_baseline = np.mean(baselines)
    mean_no_ewc = np.mean(results_without_ewc)
    std_no_ewc = np.std(results_without_ewc)
    mean_ewc = np.mean(results_with_ewc)
    std_ewc = np.std(results_with_ewc)
    
    print(f"\nBaseline (C cold): {mean_baseline:.4f}")
    
    print(f"\nWithout EWC:")
    print(f"  Mean improvement: {mean_no_ewc:+.1f}%")
    print(f"  Std: {std_no_ewc:.1f}%")
    print(f"  Positive rate: {sum(1 for x in results_without_ewc if x > 0)/len(results_without_ewc)*100:.0f}%")
    
    print(f"\nWith EWC:")
    print(f"  Mean improvement: {mean_ewc:+.1f}%")
    print(f"  Std: {std_ewc:.1f}%")
    print(f"  Positive rate: {sum(1 for x in results_with_ewc if x > 0)/len(results_with_ewc)*100:.0f}%")
    
    # Comparison
    print("\n" + "="*70)
    print("EWC EFFECTIVENESS")
    print("="*70)
    
    improvement_delta = mean_ewc - mean_no_ewc
    variance_reduction = std_no_ewc - std_ewc
    
    print(f"\nMean improvement delta: {improvement_delta:+.1f}pp")
    print(f"Variance reduction: {variance_reduction:+.1f}pp")
    
    if mean_ewc > 20 and std_ewc < 15:
        print("\n✓ EWC SUCCESS!")
        print("  - Positive compound transfer")
        print("  - Low variance (stable)")
        print("  - Catastrophic interference prevented")
    elif mean_ewc > mean_no_ewc:
        print("\n⚠ PARTIAL SUCCESS")
        print("  - EWC improves transfer")
        print("  - But below target performance")
    else:
        print("\n✗ EWC NOT EFFECTIVE")
        print("  - No improvement over baseline")
        print("  - May need tuning")
    
    return {
        'without_ewc': {'mean': mean_no_ewc, 'std': std_no_ewc},
        'with_ewc': {'mean': mean_ewc, 'std': std_ewc},
    }


if __name__ == '__main__':
    results = test_ewc_compound_transfer()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nPrevious results (no EWC):")
    print("  Mean: -10.2%")
    print("  Variance: 18.6%")
    print("  Positive: 30%")
    
    print(f"\nCurrent results (with EWC):")
    print(f"  Mean: {results['with_ewc']['mean']:+.1f}%")
    print(f"  Variance: {results['with_ewc']['std']:.1f}%")
    
    if results['with_ewc']['mean'] > 20:
        print("\n✓ EWC solves catastrophic interference!")
    else:
        print("\n⚠ EWC needs further tuning or integration")
