"""
Diagnostic Test: Why does compound transfer fail?

Tests multiple hypotheses:
1. Insufficient meta-layers (need Meta^0-5, not just 0-2)
2. reset_task_state() clears too much knowledge
3. Need GlobalDynamicsOptimizer for adaptive gating
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


def test_hypothesis_1_more_meta_layers():
    """Hypothesis: Need more meta-layers for compound transfer."""
    print("\n" + "="*60)
    print("HYPOTHESIS 1: More Meta-Layers")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    W_A, b_A = generate_task(n_inputs, n_outputs, 42)
    W_B, b_B = generate_task(n_inputs, n_outputs, 123)
    W_C, b_C = generate_task(n_inputs, n_outputs, 456)
    
    # Test with full Meta^0-5 pipeline
    print("\n[Testing with Meta^0-5 (default pipeline)]")
    pipeline = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    loss_a = train_on_task(pipeline, W_A, b_A, 100)
    print(f"  Task A: loss={loss_a:.4f}")
    
    pipeline.reset_task_state()
    loss_b = train_on_task(pipeline, W_B, b_B, 100)
    print(f"  Task B: loss={loss_b:.4f}")
    
    pipeline.reset_task_state()
    loss_c = train_on_task(pipeline, W_C, b_C, 50)
    print(f"  Task C (after A+B): loss={loss_c:.4f}")
    
    # Compare to cold start
    pipeline_cold = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    loss_c_cold = train_on_task(pipeline_cold, W_C, b_C, 50)
    print(f"  Task C (cold): loss={loss_c_cold:.4f}")
    
    improvement = (loss_c_cold - loss_c) / loss_c_cold * 100
    print(f"\n  Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("  ✓ More meta-layers help!")
    else:
        print("  ✗ More meta-layers don't help")
    
    return improvement


def test_hypothesis_2_no_reset():
    """Hypothesis: reset_task_state() clears too much."""
    print("\n" + "="*60)
    print("HYPOTHESIS 2: Don't Reset Task State")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    W_A, b_A = generate_task(n_inputs, n_outputs, 42)
    W_B, b_B = generate_task(n_inputs, n_outputs, 123)
    W_C, b_C = generate_task(n_inputs, n_outputs, 456)
    
    print("\n[Training A→B→C WITHOUT reset_task_state()]")
    pipeline = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    loss_a = train_on_task(pipeline, W_A, b_A, 100)
    print(f"  Task A: loss={loss_a:.4f}")
    
    # NO RESET - keep all state
    loss_b = train_on_task(pipeline, W_B, b_B, 100)
    print(f"  Task B (no reset): loss={loss_b:.4f}")
    
    # NO RESET - keep all state
    loss_c = train_on_task(pipeline, W_C, b_C, 50)
    print(f"  Task C (no reset): loss={loss_c:.4f}")
    
    # Compare to cold start
    pipeline_cold = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    loss_c_cold = train_on_task(pipeline_cold, W_C, b_C, 50)
    print(f"  Task C (cold): loss={loss_c_cold:.4f}")
    
    improvement = (loss_c_cold - loss_c) / loss_c_cold * 100
    print(f"\n  Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("  ✓ Not resetting helps!")
    else:
        print("  ✗ Not resetting doesn't help (catastrophic forgetting)")
    
    return improvement


def test_hypothesis_3_adaptive_pipeline():
    """Hypothesis: Need GlobalDynamicsOptimizer."""
    print("\n" + "="*60)
    print("HYPOTHESIS 3: Adaptive Pipeline (GlobalDynamicsOptimizer)")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    W_A, b_A = generate_task(n_inputs, n_outputs, 42)
    W_B, b_B = generate_task(n_inputs, n_outputs, 123)
    W_C, b_C = generate_task(n_inputs, n_outputs, 456)
    
    print("\n[Testing with adaptive pipeline]")
    pipeline = MetaNPipeline.create_adaptive(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    loss_a = train_on_task(pipeline, W_A, b_A, 100)
    print(f"  Task A: loss={loss_a:.4f}")
    
    pipeline.reset_task_state()
    loss_b = train_on_task(pipeline, W_B, b_B, 100)
    print(f"  Task B: loss={loss_b:.4f}")
    
    pipeline.reset_task_state()
    loss_c = train_on_task(pipeline, W_C, b_C, 50)
    print(f"  Task C (after A+B): loss={loss_c:.4f}")
    
    # Compare to cold start
    pipeline_cold = MetaNPipeline.create_adaptive(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    loss_c_cold = train_on_task(pipeline_cold, W_C, b_C, 50)
    print(f"  Task C (cold): loss={loss_c_cold:.4f}")
    
    improvement = (loss_c_cold - loss_c) / loss_c_cold * 100
    print(f"\n  Improvement: {improvement:+.1f}%")
    
    if improvement > 0:
        print("  ✓ Adaptive pipeline helps!")
    else:
        print("  ✗ Adaptive pipeline doesn't help")
    
    return improvement


if __name__ == '__main__':
    print("="*60)
    print("DIAGNOSING COMPOUND TRANSFER FAILURE")
    print("="*60)
    
    h1 = test_hypothesis_1_more_meta_layers()
    h2 = test_hypothesis_2_no_reset()
    h3 = test_hypothesis_3_adaptive_pipeline()
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    print(f"H1 (More meta-layers):      {h1:+.1f}%")
    print(f"H2 (No reset):              {h2:+.1f}%")
    print(f"H3 (Adaptive pipeline):     {h3:+.1f}%")
    
    best = max([(h1, "More meta-layers"), (h2, "No reset"), (h3, "Adaptive pipeline")], 
               key=lambda x: x[0])
    
    if best[0] > 0:
        print(f"\n✓ SOLUTION FOUND: {best[1]} ({best[0]:+.1f}% improvement)")
    else:
        print("\n✗ NO SOLUTION FOUND")
        print("  Compound transfer may require architectural changes.")
