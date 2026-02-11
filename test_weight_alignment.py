"""
Test: Weight Alignment Hypothesis

User's insight: A→C→B gets +40% but C→A→B gets -33.3%.
Both are supervised→supervised→RL, only difference is A vs C first.

Hypothesis: Success is due to LUCKY WEIGHT ALIGNMENT, not true transfer.

Test: Run A→C→B with different random seeds.
- If results are stable → architecture learning
- If results swing wildly → weight alignment luck

This will reveal if we need a representation optimizer (Meta^3/4).
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_supervised_task(n_inputs: int, n_outputs: int, seed: int, complexity: float = 0.5):
    """Generate a supervised linear task."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * complexity
    bias = rng.randn(n_outputs) * 0.1
    return W, bias


def train_supervised(pipeline, W, bias, n_steps: int):
    """Train on supervised task, return final loss."""
    losses = []
    for step in range(n_steps):
        x = np.random.randn(W.shape[1])
        y = W @ x + bias
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    return np.mean(losses[-20:])


def train_rl(pipeline, n_steps: int):
    """Train on RL task, return final loss."""
    losses = []
    n_inputs = 16
    
    for step in range(n_steps):
        x = np.random.randn(n_inputs)
        result = pipeline.step(x, target=None, reward=0.0)
        output_sum = np.sum(result['output'])
        reward = 1.0 if output_sum > 0 else -1.0
        
        if step > 0:
            result = pipeline.step(x, target=None, reward=reward)
            losses.append(result['loss'])
    
    return np.mean(losses[-20:]) if losses else 1.0


def test_seed_stability():
    """
    Test if A→C→B results are stable across different random seeds.
    
    If stable: True transfer learning (architecture)
    If unstable: Weight alignment luck
    """
    print("\n" + "="*70)
    print("WEIGHT ALIGNMENT HYPOTHESIS TEST")
    print("="*70)
    
    print("\nHypothesis:")
    print("  If A->C->B results vary wildly with different seeds,")
    print("  then success is due to lucky weight alignment,")
    print("  not true abstraction learning.")
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    n_seeds = 10
    
    results_acb = []
    results_cab = []
    baselines_b = []
    
    print(f"\nRunning {n_seeds} trials with different random seeds...")
    
    for seed in range(n_seeds):
        print(f"\n[Seed {seed}]")
        
        # Generate tasks with this seed
        W_A, b_A = generate_supervised_task(n_inputs, n_outputs, seed=seed*10, complexity=0.3)
        W_C, b_C = generate_supervised_task(n_inputs, n_outputs, seed=seed*10+456, complexity=0.5)
        
        # Baseline: B cold
        pipeline_cold = MetaNPipeline.create_default(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )
        baseline_b = train_rl(pipeline_cold, test_steps)
        baselines_b.append(baseline_b)
        print(f"  B (cold): {baseline_b:.4f}")
        
        # A→C→B
        pipeline_acb = MetaNPipeline.create_default(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )
        train_supervised(pipeline_acb, W_A, b_A, train_steps)
        pipeline_acb.reset_task_state()
        meta2 = pipeline_acb.stack.get_layer(2)
        if meta2:
            meta2.task_detector.reset()
        
        train_supervised(pipeline_acb, W_C, b_C, train_steps)
        pipeline_acb.reset_task_state()
        if meta2:
            meta2.task_detector.reset()
        
        loss_acb = train_rl(pipeline_acb, test_steps)
        improvement_acb = (baseline_b - loss_acb) / baseline_b * 100
        results_acb.append(improvement_acb)
        print(f"  A->C->B: {loss_acb:.4f} ({improvement_acb:+.1f}%)")
        
        # C->A->B
        pipeline_cab = MetaNPipeline.create_default(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )
        train_supervised(pipeline_cab, W_C, b_C, train_steps)
        pipeline_cab.reset_task_state()
        meta2 = pipeline_cab.stack.get_layer(2)
        if meta2:
            meta2.task_detector.reset()
        
        train_supervised(pipeline_cab, W_A, b_A, train_steps)
        pipeline_cab.reset_task_state()
        if meta2:
            meta2.task_detector.reset()
        
        loss_cab = train_rl(pipeline_cab, test_steps)
        improvement_cab = (baseline_b - loss_cab) / baseline_b * 100
        results_cab.append(improvement_cab)
        print(f"  C->A->B: {loss_cab:.4f} ({improvement_cab:+.1f}%)")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    mean_acb = np.mean(results_acb)
    std_acb = np.std(results_acb)
    min_acb = np.min(results_acb)
    max_acb = np.max(results_acb)
    
    mean_cab = np.mean(results_cab)
    std_cab = np.std(results_cab)
    min_cab = np.min(results_cab)
    max_cab = np.max(results_cab)
    
    print(f"\nA→C→B (n={n_seeds}):")
    print(f"  Mean: {mean_acb:+.1f}%")
    print(f"  Std:  {std_acb:.1f}%")
    print(f"  Range: [{min_acb:+.1f}%, {max_acb:+.1f}%]")
    
    print(f"\nC→A→B (n={n_seeds}):")
    print(f"  Mean: {mean_cab:+.1f}%")
    print(f"  Std:  {std_cab:.1f}%")
    print(f"  Range: [{min_cab:+.1f}%, {max_cab:+.1f}%]")
    
    # Hypothesis test
    print("\n" + "="*70)
    print("HYPOTHESIS TEST")
    print("="*70)
    
    # High variance = weight alignment luck
    # Low variance = true transfer
    
    variance_threshold = 20.0  # If std > 20%, it's unstable
    
    print(f"\nVariance threshold: {variance_threshold}%")
    print(f"A→C→B std: {std_acb:.1f}%")
    print(f"C→A→B std: {std_cab:.1f}%")
    
    if std_acb > variance_threshold or std_cab > variance_threshold:
        print("\n✗ WEIGHT ALIGNMENT LUCK CONFIRMED")
        print("  High variance indicates results depend on random weight alignment")
        print("  NOT true abstraction learning")
        print("\n  This means:")
        print("  - Current system doesn't learn transferable representations")
        print("  - Success is due to lucky weight compatibility")
        print("  - We need a representation optimizer (Meta^3/4)")
    else:
        print("\n✓ TRUE TRANSFER LEARNING")
        print("  Low variance indicates stable transfer")
        print("  System learns task-independent abstractions")
    
    # Additional check: correlation between orderings
    correlation = np.corrcoef(results_acb, results_cab)[0, 1]
    print(f"\nCorrelation between A→C→B and C→A→B: {correlation:.3f}")
    if abs(correlation) > 0.5:
        print("  High correlation → weight alignment matters")
    else:
        print("  Low correlation → orderings have independent effects")
    
    return {
        'acb_mean': mean_acb,
        'acb_std': std_acb,
        'cab_mean': mean_cab,
        'cab_std': std_cab,
        'correlation': correlation,
    }


if __name__ == '__main__':
    results = test_seed_stability()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if results['acb_std'] > 20 or results['cab_std'] > 20:
        print("\n**Weight alignment hypothesis CONFIRMED**")
        print("\nThe system is NOT learning transferable abstractions.")
        print("It's carrying forward task-specific weights that")
        print("happen to align in some cases and conflict in others.")
        print("\nNext steps:")
        print("  1. Add representation optimizer (Meta^3/4)")
        print("  2. Use tasks with shared structure (not random weights)")
        print("  3. Test on tasks with compositional structure")
        print("  4. Implement holographic/abstract representations")
    else:
        print("\n**True transfer learning detected**")
        print("\nThe system shows stable transfer across seeds.")
        print("This suggests learning of task-independent abstractions.")
