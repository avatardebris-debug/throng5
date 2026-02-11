"""
Compound Transfer Validation with Adaptive Pipeline

Now that we know GlobalDynamicsOptimizer enables compound transfer,
run proper statistical validation with N=10 seeds.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_task(n_inputs: int, n_outputs: int, seed: int):
    """Generate a linear task."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * 0.4
    bias = rng.randn(n_outputs) * 0.1
    return W, bias


def train_on_task(pipeline, W, bias, n_steps: int):
    """Train and return final loss."""
    losses = []
    for _ in range(n_steps):
        x = np.random.randn(W.shape[1])
        y = W @ x + bias
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    return np.mean(losses[-20:])


def run_single_trial(seed: int, condition: str):
    """Run a single trial for one condition."""
    n_inputs, n_outputs = 16, 8
    
    # Generate tasks (use seed offset for variation)
    W_A, b_A = generate_task(n_inputs, n_outputs, seed * 10 + 1)
    W_B, b_B = generate_task(n_inputs, n_outputs, seed * 10 + 2)
    W_C, b_C = generate_task(n_inputs, n_outputs, seed * 10 + 3)
    
    pipeline = MetaNPipeline.create_adaptive(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    if condition == 'cold':
        # C only
        loss = train_on_task(pipeline, W_C, b_C, 50)
    
    elif condition == 'A_C':
        # A then C
        train_on_task(pipeline, W_A, b_A, 100)
        pipeline.reset_task_state()
        loss = train_on_task(pipeline, W_C, b_C, 50)
    
    elif condition == 'B_C':
        # B then C
        train_on_task(pipeline, W_B, b_B, 100)
        pipeline.reset_task_state()
        loss = train_on_task(pipeline, W_C, b_C, 50)
    
    elif condition == 'ABC':
        # A then B then C
        train_on_task(pipeline, W_A, b_A, 100)
        pipeline.reset_task_state()
        train_on_task(pipeline, W_B, b_B, 100)
        pipeline.reset_task_state()
        loss = train_on_task(pipeline, W_C, b_C, 50)
    
    return loss


def main():
    print("="*60)
    print("COMPOUND TRANSFER VALIDATION (Adaptive Pipeline)")
    print("="*60)
    
    n_trials = 10
    conditions = ['cold', 'A_C', 'B_C', 'ABC']
    
    results = {cond: [] for cond in conditions}
    
    print(f"\nRunning {n_trials} trials per condition...")
    
    for seed in range(n_trials):
        print(f"\n[Trial {seed+1}/{n_trials}]")
        for cond in conditions:
            loss = run_single_trial(seed, cond)
            results[cond].append(loss)
            print(f"  {cond:8s}: {loss:.4f}")
    
    # Statistical analysis
    print("\n" + "="*60)
    print("STATISTICAL RESULTS")
    print("="*60)
    
    for cond in conditions:
        mean = np.mean(results[cond])
        std = np.std(results[cond])
        print(f"{cond:8s}: {mean:.4f} ± {std:.4f}")
    
    # Compute improvements
    print("\n" + "="*60)
    print("TRANSFER IMPROVEMENTS (vs Cold)")
    print("="*60)
    
    cold_mean = np.mean(results['cold'])
    
    for cond in ['A_C', 'B_C', 'ABC']:
        cond_mean = np.mean(results[cond])
        improvement = (cold_mean - cond_mean) / cold_mean * 100
        print(f"{cond:8s}: {improvement:+.1f}%")
    
    # Compound transfer check
    print("\n" + "="*60)
    print("COMPOUND TRANSFER VALIDATION")
    print("="*60)
    
    abc_mean = np.mean(results['ABC'])
    ac_mean = np.mean(results['A_C'])
    bc_mean = np.mean(results['B_C'])
    
    compound_better_than_ac = abc_mean < ac_mean
    compound_better_than_bc = abc_mean < bc_mean
    compound_better_than_cold = abc_mean < cold_mean
    
    print(f"A+B→C better than A→C:  {compound_better_than_ac}")
    print(f"A+B→C better than B→C:  {compound_better_than_bc}")
    print(f"A+B→C better than cold: {compound_better_than_cold}")
    
    if compound_better_than_ac and compound_better_than_bc and compound_better_than_cold:
        improvement = (cold_mean - abc_mean) / cold_mean * 100
        print(f"\n✓ COMPOUND TRANSFER CONFIRMED! (+{improvement:.1f}%)")
        print("  Meta^N with GlobalDynamicsOptimizer achieves compound generalization.")
    else:
        print("\n⚠ PARTIAL COMPOUND TRANSFER")
        print("  Some conditions show improvement, but not all.")
    
    return results


if __name__ == '__main__':
    results = main()
