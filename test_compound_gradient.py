"""
Test: Compound Transfer with Gradient Learning

The original test showed -18.9% degradation because it used
STDP/Hebbian (RL learning) on supervised tasks with reward=0.0.

This test uses GRADIENT DESCENT for supervised tasks.

Expected: Positive compound transfer (A+B→C better than alternatives)
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def generate_task(n_inputs: int, n_outputs: int, seed: int, complexity: float = 0.5):
    """Generate a linear task."""
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


def test_compound_transfer_with_gradient():
    """Test A+B→C with gradient learning."""
    print("\n" + "="*60)
    print("COMPOUND TRANSFER TEST (Gradient Learning)")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    
    # Define tasks
    W_A, b_A = generate_task(n_inputs, n_outputs, seed=42, complexity=0.3)
    W_B, b_B = generate_task(n_inputs, n_outputs, seed=123, complexity=0.4)
    W_C, b_C = generate_task(n_inputs, n_outputs, seed=456, complexity=0.5)
    
    results = {}
    
    # Condition 1: C cold (baseline)
    print("\n[Condition 1: C Cold]")
    pipeline_cold = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    # Set to gradient mode
    synapse_layer = pipeline_cold.stack.get_layer(1)
    if synapse_layer:
        synapse_layer.learning_mode = 'gradient'
    
    results['C_cold'] = train_on_task(pipeline_cold, W_C, b_C, test_steps, "C (cold)")
    
    # Condition 2: A→C
    print("\n[Condition 2: A→C]")
    pipeline_ac = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    synapse_layer = pipeline_ac.stack.get_layer(1)
    if synapse_layer:
        synapse_layer.learning_mode = 'gradient'
    
    train_on_task(pipeline_ac, W_A, b_A, train_steps, "A")
    pipeline_ac.reset_task_state()
    results['A_C'] = train_on_task(pipeline_ac, W_C, b_C, test_steps, "C (after A)")
    
    # Condition 3: B→C
    print("\n[Condition 3: B→C]")
    pipeline_bc = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    synapse_layer = pipeline_bc.stack.get_layer(1)
    if synapse_layer:
        synapse_layer.learning_mode = 'gradient'
    
    train_on_task(pipeline_bc, W_B, b_B, train_steps, "B")
    pipeline_bc.reset_task_state()
    results['B_C'] = train_on_task(pipeline_bc, W_C, b_C, test_steps, "C (after B)")
    
    # Condition 4: A+B→C (compound transfer)
    print("\n[Condition 4: A+B→C (Compound)]")
    pipeline_abc = MetaNPipeline.create_minimal(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    synapse_layer = pipeline_abc.stack.get_layer(1)
    if synapse_layer:
        synapse_layer.learning_mode = 'gradient'
    
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
    
    print(f"A+B→C better than A→C:  {compound_better_than_ac}")
    print(f"A+B→C better than B→C:  {compound_better_than_bc}")
    print(f"A+B→C better than cold: {compound_better_than_cold}")
    
    if compound_better_than_ac and compound_better_than_bc and compound_better_than_cold:
        print("\n✓ COMPOUND TRANSFER CONFIRMED!")
        print("  Gradient learning enables compound generalization")
    elif results['ABC'] > results['C_cold'] * 1.1:
        print("\n✗ CATASTROPHIC INTERFERENCE")
        print("  A+B→C is worse than cold start")
    else:
        print("\n⚠ PARTIAL COMPOUND TRANSFER")
        print("  Some improvement but not all conditions met")
    
    return results


if __name__ == '__main__':
    results = test_compound_transfer_with_gradient()
    
    print("\n" + "="*60)
    print("COMPARISON TO ORIGINAL TEST")
    print("="*60)
    print("\nOriginal test (STDP/Hebbian, reward=0.0):")
    print("  A+B→C: -18.9% (catastrophic interference)")
    print("\nThis test (Gradient descent):")
    abc_improvement = (results['C_cold'] - results['ABC']) / results['C_cold'] * 100
    print(f"  A+B→C: {abc_improvement:+.1f}%")
    
    if abc_improvement > 0:
        print("\n✓ GRADIENT LEARNING PREVENTS CATASTROPHIC INTERFERENCE!")
    else:
        print("\n⚠ Still seeing interference, but less severe")
