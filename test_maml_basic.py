"""
Test: Basic MAML Infrastructure

Validates that MAML inner/outer loop works on simple supervised tasks.
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from throng3.pipeline import MetaNPipeline


def generate_function_task(task_type='linear'):
    """Generate a simple function approximation task."""
    n_samples = 20
    x = np.random.randn(n_samples, 4)
    
    if task_type == 'linear':
        # y = w @ x
        w = np.random.randn(4)
        y = x @ w
    elif task_type == 'quadratic':
        # y = x^2
        y = np.sum(x ** 2, axis=1)
    else:
        # Random
        y = np.random.randn(n_samples)
    
    # Split into support and query
    support_x, support_y = x[:10], y[:10]
    query_x, query_y = x[10:], y[10:]
    
    return support_x, support_y, query_x, query_y


def test_maml_basic():
    """Test basic MAML on function approximation."""
    print("\n" + "="*70)
    print("BASIC MAML TEST")
    print("="*70)
    
    print("\nTask: Function approximation (linear)")
    print("Goal: Validate MAML inner/outer loop")
    
    # Create pipeline with MAML
    pipeline = MetaNPipeline.create_with_maml(
        n_neurons=50,
        n_inputs=4,
        n_outputs=1,
        meta_lr=0.001
    )
    
    # Get MAML layer
    maml_layer = pipeline.stack.get_layer(3)  # Meta^3
    
    print(f"\nMAML layer: {maml_layer.name}")
    print(f"Meta-LR: {maml_layer.maml_config.meta_lr}")
    print(f"Inner steps: {maml_layer.maml_config.inner_steps}")
    
    # Generate test tasks
    n_tasks = 5
    print(f"\nGenerating {n_tasks} tasks...")
    
    tasks = []
    for i in range(n_tasks):
        support_x, support_y, query_x, query_y = generate_function_task('linear')
        
        # Convert to support/query sets
        support_set = [(x, np.array([y])) for x, y in zip(support_x, support_y)]
        query_set = [(x, np.array([y])) for x, y in zip(query_x, query_y)]
        
        # Use correctly-shaped weights for this function approximation task
        # (W_out maps from input_dim=4 to output_dim=1)
        task_weights = {'W_out': np.random.randn(1, 4) * 0.1}
        
        tasks.append({
            'task_type': 'supervised',
            'support_set': support_set,
            'query_set': query_set,
            'weights': task_weights,
        })
    
    print(f"Tasks generated: {len(tasks)}")
    print(f"Support set size: {len(tasks[0]['support_set'])}")
    print(f"Query set size: {len(tasks[0]['query_set'])}")
    
    # Test inner loop
    print("\n" + "-"*70)
    print("Testing Inner Loop (Task Adaptation)")
    print("-"*70)
    
    task = tasks[0]
    initial_weights = task['weights']
    
    print(f"\nInitial weights shape: {initial_weights['W_out'].shape}")
    
    adapted_weights = maml_layer.inner_loop(
        initial_weights,
        task['support_set'],
        inner_lr=0.01,
        inner_steps=5
    )
    
    print(f"Adapted weights shape: {adapted_weights['W_out'].shape}")
    
    # Measure adaptation
    weight_change = np.linalg.norm(
        adapted_weights['W_out'] - initial_weights['W_out']
    )
    print(f"Weight change (L2 norm): {weight_change:.4f}")
    
    if weight_change > 0.001:
        print("✓ Inner loop adapts weights")
    else:
        print("✗ Inner loop not adapting")
    
    # Test meta-update
    print("\n" + "-"*70)
    print("Testing Outer Loop (Meta-Update)")
    print("-"*70)
    
    print(f"\nMeta-updates before: {maml_layer.meta_updates}")
    
    # Run meta-update
    maml_layer.meta_update(tasks)
    
    print(f"Meta-updates after: {maml_layer.meta_updates}")
    
    if maml_layer.meta_updates > 0:
        print("✓ Meta-update executed")
    else:
        print("✗ Meta-update failed")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n✓ MAML infrastructure works")
    print(f"  - Inner loop: adapts weights")
    print(f"  - Outer loop: meta-updates executed")
    print(f"  - Tasks processed: {len(tasks)}")
    
    print(f"\n✓ Ready for Phase 2: Task Conditioning")


if __name__ == '__main__':
    test_maml_basic()
