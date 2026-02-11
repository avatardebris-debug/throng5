"""
Comprehensive Compound Transfer Test: All Task Orderings

Tests all permutations of A (supervised), B (RL), C (supervised)
to verify compound transfer is robust to task ordering.

Also documents what each task actually does.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from itertools import permutations


def generate_supervised_task(n_inputs: int, n_outputs: int, seed: int, complexity: float = 0.5):
    """Generate a supervised linear task."""
    rng = np.random.RandomState(seed)
    W = rng.randn(n_outputs, n_inputs) * complexity
    bias = rng.randn(n_outputs) * 0.1
    return W, bias


def train_supervised(pipeline, W, bias, n_steps: int, task_name: str = "Task", verbose: bool = True):
    """
    Train on supervised linear regression task.
    
    Task: Learn to predict y = W @ x + bias
    Signal: MSE loss between prediction and target
    Expected mechanism: gradient descent
    """
    losses = []
    for step in range(n_steps):
        x = np.random.randn(W.shape[1])
        y = W @ x + bias
        result = pipeline.step(x, target=y, reward=0.0)
        losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:])
    if verbose:
        print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def train_rl(pipeline, n_steps: int, task_name: str = "Task", verbose: bool = True):
    """
    Train on RL task with reward signal.
    
    Task: Maximize output sum (simple reward-based learning)
    Signal: +1 if sum(output) > 0, else -1
    Expected mechanism: STDP/Hebbian (bio-inspired)
    """
    losses = []
    n_inputs = 16
    
    for step in range(n_steps):
        x = np.random.randn(n_inputs)
        result = pipeline.step(x, target=None, reward=0.0)
        output_sum = np.sum(result['output'])
        reward = 1.0 if output_sum > 0 else -1.0
        
        # Send reward on next step
        if step > 0:
            result = pipeline.step(x, target=None, reward=reward)
            losses.append(result['loss'])
    
    final_loss = np.mean(losses[-20:]) if losses else 1.0
    if verbose:
        print(f"  {task_name}: {n_steps} steps, final loss={final_loss:.4f}")
    return final_loss


def test_task_ordering(ordering: tuple, tasks: dict, n_inputs: int, n_outputs: int, 
                       train_steps: int, test_steps: int, verbose: bool = True):
    """
    Test a specific task ordering.
    
    Args:
        ordering: tuple like ('A', 'B', 'C')
        tasks: dict with task definitions
        ...
    
    Returns:
        final_loss on last task
    """
    pipeline = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    meta2 = pipeline.stack.get_layer(2)
    
    if verbose:
        print(f"\n[Ordering: {' → '.join(ordering)}]")
    
    # Train on all tasks in order
    for i, task_name in enumerate(ordering[:-1]):  # All but last
        task_info = tasks[task_name]
        
        if task_info['type'] == 'supervised':
            train_supervised(pipeline, task_info['W'], task_info['b'], 
                           train_steps, f"{task_name} (supervised)", verbose)
        else:  # RL
            train_rl(pipeline, train_steps, f"{task_name} (RL)", verbose)
        
        if meta2 and verbose:
            print(f"    Mechanism: {meta2.current_mechanism}")
        
        # Reset for next task
        pipeline.reset_task_state()
        if meta2:
            meta2.task_detector.reset()
    
    # Test on final task
    final_task = ordering[-1]
    task_info = tasks[final_task]
    
    if task_info['type'] == 'supervised':
        final_loss = train_supervised(pipeline, task_info['W'], task_info['b'],
                                     test_steps, f"{final_task} (test)", verbose)
    else:  # RL
        final_loss = train_rl(pipeline, test_steps, f"{final_task} (test)", verbose)
    
    if meta2 and verbose:
        print(f"    Final mechanism: {meta2.current_mechanism}")
    
    return final_loss


def test_all_orderings():
    """
    Test all permutations of A, B, C tasks.
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE COMPOUND TRANSFER TEST: ALL TASK ORDERINGS")
    print("="*70)
    
    print("\nTask Definitions:")
    print("  A: Supervised linear regression (complexity=0.3)")
    print("     → Expected mechanism: gradient")
    print("  B: RL reward-based learning (maximize output sum)")
    print("     → Expected mechanism: rl (STDP/Hebbian)")
    print("  C: Supervised linear regression (complexity=0.5)")
    print("     → Expected mechanism: gradient")
    
    n_inputs, n_outputs = 16, 8
    train_steps = 100
    test_steps = 50
    
    # Define tasks
    W_A, b_A = generate_supervised_task(n_inputs, n_outputs, seed=42, complexity=0.3)
    W_C, b_C = generate_supervised_task(n_inputs, n_outputs, seed=456, complexity=0.5)
    
    tasks = {
        'A': {'type': 'supervised', 'W': W_A, 'b': b_A},
        'B': {'type': 'rl'},
        'C': {'type': 'supervised', 'W': W_C, 'b': b_C},
    }
    
    # Get baseline (cold start on each task)
    print("\n" + "="*70)
    print("BASELINES (Cold Start)")
    print("="*70)
    
    baselines = {}
    for task_name in ['A', 'B', 'C']:
        pipeline = MetaNPipeline.create_default(
            n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
        )
        
        task_info = tasks[task_name]
        if task_info['type'] == 'supervised':
            loss = train_supervised(pipeline, task_info['W'], task_info['b'],
                                  test_steps, f"{task_name} (cold)")
        else:
            loss = train_rl(pipeline, test_steps, f"{task_name} (cold)")
        
        baselines[task_name] = loss
    
    # Test all orderings
    print("\n" + "="*70)
    print("ALL TASK ORDERINGS")
    print("="*70)
    
    results = {}
    
    # All 6 permutations of A, B, C
    all_orderings = list(permutations(['A', 'B', 'C']))
    
    for ordering in all_orderings:
        final_loss = test_task_ordering(ordering, tasks, n_inputs, n_outputs,
                                       train_steps, test_steps, verbose=True)
        results[ordering] = final_loss
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print("\nBaselines (cold start):")
    for task, loss in baselines.items():
        print(f"  {task}: {loss:.4f}")
    
    print("\nAll orderings (testing on final task):")
    for ordering, loss in sorted(results.items(), key=lambda x: x[1]):
        final_task = ordering[-1]
        baseline = baselines[final_task]
        improvement = (baseline - loss) / baseline * 100
        print(f"  {' → '.join(ordering)}: {loss:.4f} ({improvement:+.1f}% vs cold)")
    
    # Find best ordering for each final task
    print("\n" + "="*70)
    print("BEST ORDERING FOR EACH FINAL TASK")
    print("="*70)
    
    for final_task in ['A', 'B', 'C']:
        relevant = {k: v for k, v in results.items() if k[-1] == final_task}
        best_ordering = min(relevant, key=relevant.get)
        best_loss = relevant[best_ordering]
        baseline = baselines[final_task]
        improvement = (baseline - best_loss) / baseline * 100
        
        print(f"\nFinal task {final_task}:")
        print(f"  Best ordering: {' → '.join(best_ordering)}")
        print(f"  Loss: {best_loss:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")
        
        # Show all orderings for this final task
        print(f"  All orderings ending in {final_task}:")
        for ordering, loss in sorted(relevant.items(), key=lambda x: x[1]):
            imp = (baseline - loss) / baseline * 100
            print(f"    {' → '.join(ordering)}: {loss:.4f} ({imp:+.1f}%)")
    
    # Overall analysis
    print("\n" + "="*70)
    print("OVERALL ANALYSIS")
    print("="*70)
    
    positive_transfer = sum(1 for ordering, loss in results.items() 
                           if loss < baselines[ordering[-1]])
    total = len(results)
    
    print(f"\nPositive transfer: {positive_transfer}/{total} orderings ({positive_transfer/total*100:.1f}%)")
    
    avg_improvement = np.mean([
        (baselines[ordering[-1]] - loss) / baselines[ordering[-1]] * 100
        for ordering, loss in results.items()
    ])
    print(f"Average improvement: {avg_improvement:+.1f}%")
    
    if positive_transfer >= total * 0.8:
        print("\n✓ ROBUST COMPOUND TRANSFER!")
        print("  Positive transfer in most orderings")
    elif positive_transfer >= total * 0.5:
        print("\n⚠ PARTIAL COMPOUND TRANSFER")
        print("  Positive transfer in some orderings")
    else:
        print("\n✗ LIMITED COMPOUND TRANSFER")
        print("  Need to investigate further")
    
    return results, baselines


if __name__ == '__main__':
    results, baselines = test_all_orderings()
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("\nCurrent limitations:")
    print("  1. Only one RL algorithm (STDP/Hebbian)")
    print("  2. Simple RL task (maximize output sum)")
    print("  3. No Q-learning vs policy gradient selection")
    print("\nFuture enhancements:")
    print("  1. Add Q-learning for discrete action spaces")
    print("  2. Add policy gradient for continuous control")
    print("  3. Meta^2 learns to select RL algorithm type")
    print("  4. Test on GridWorld, CartPole, etc.")
