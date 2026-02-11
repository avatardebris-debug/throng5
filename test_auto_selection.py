"""
Test: Automatic Learning Mechanism Selection

Verify that Meta^2 can automatically detect task type and
select the appropriate learning mechanism without manual configuration.

Test scenario:
1. Run 100 steps of supervised task → should detect and use 'gradient'
2. Switch to RL task → should detect and switch to 'rl'
3. Verify mechanism switches automatically
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def test_auto_mechanism_selection():
    """
    Test that Meta^2 automatically selects learning mechanism.
    """
    print("\n" + "="*60)
    print("AUTOMATIC MECHANISM SELECTION TEST")
    print("="*60)
    
    n_inputs, n_outputs = 16, 8
    
    # Create pipeline with Meta^0-2 (need Meta^2 for auto-selection)
    pipeline = MetaNPipeline.create_default(
        n_neurons=100, n_inputs=n_inputs, n_outputs=n_outputs
    )
    
    # Get Meta^2 layer to monitor mechanism selection
    meta2 = pipeline.stack.get_layer(2)
    
    # Phase 1: Supervised task (100 steps)
    print("\n[Phase 1: Supervised Task - 100 steps]")
    print("Expected: Meta^2 detects supervised, selects 'gradient'")
    
    np.random.seed(42)
    W_sup = np.random.randn(n_outputs, n_inputs) * 0.5
    b_sup = np.random.randn(n_outputs) * 0.1
    
    for step in range(100):
        x = np.random.randn(n_inputs)
        y = W_sup @ x + b_sup
        result = pipeline.step(x, target=y, reward=0.0)
        
        # Check mechanism every 20 steps
        if step % 20 == 19:
            if meta2:
                task_chars = meta2.task_detector.get_characteristics()
                if task_chars:
                    print(f"  Step {step+1}: signal_type={task_chars.signal_type}, "
                          f"mechanism={meta2.current_mechanism}, "
                          f"confidence={task_chars.confidence:.2f}")
    
    # Check final mechanism
    if meta2:
        final_mechanism_1 = meta2.current_mechanism
        task_chars_1 = meta2.task_detector.get_characteristics()
        print(f"\nPhase 1 Result:")
        print(f"  Detected: {task_chars_1.signal_type if task_chars_1 else 'unknown'}")
        print(f"  Selected mechanism: {final_mechanism_1}")
        
        if final_mechanism_1 == 'gradient':
            print("  ✓ Correct! Selected gradient for supervised task")
        else:
            print(f"  ⚠ Expected 'gradient', got '{final_mechanism_1}'")
    
    # Phase 2: RL task (100 steps)
    print("\n[Phase 2: RL Task - 100 steps]")
    print("Expected: Meta^2 detects RL, switches to 'rl'")
    
    # Reset task detector for new task
    if meta2:
        meta2.task_detector.reset()
    
    # Simple RL task: reward based on output magnitude
    for step in range(100):
        x = np.random.randn(n_inputs)
        result = pipeline.step(x, target=None, reward=np.random.randn())
        
        # Check mechanism every 20 steps
        if step % 20 == 19:
            if meta2:
                task_chars = meta2.task_detector.get_characteristics()
                if task_chars:
                    print(f"  Step {step+1}: signal_type={task_chars.signal_type}, "
                          f"mechanism={meta2.current_mechanism}, "
                          f"confidence={task_chars.confidence:.2f}")
    
    # Check final mechanism
    if meta2:
        final_mechanism_2 = meta2.current_mechanism
        task_chars_2 = meta2.task_detector.get_characteristics()
        print(f"\nPhase 2 Result:")
        print(f"  Detected: {task_chars_2.signal_type if task_chars_2 else 'unknown'}")
        print(f"  Selected mechanism: {final_mechanism_2}")
        
        if final_mechanism_2 == 'rl':
            print("  ✓ Correct! Selected RL for RL task")
        else:
            print(f"  ⚠ Expected 'rl', got '{final_mechanism_2}'")
    
    # Analysis
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if meta2:
        print(f"Phase 1 (Supervised): {final_mechanism_1}")
        print(f"Phase 2 (RL):         {final_mechanism_2}")
        
        if final_mechanism_1 == 'gradient' and final_mechanism_2 == 'rl':
            print("\n✓ AUTO-SELECTION WORKS!")
            print("  Meta^2 correctly detects task type and selects mechanism")
            return True
        elif final_mechanism_1 == 'gradient':
            print("\n⚠ PARTIAL SUCCESS")
            print("  Detected supervised correctly, but didn't switch to RL")
            return False
        else:
            print("\n✗ AUTO-SELECTION NEEDS WORK")
            print("  Meta^2 not detecting task types correctly")
            return False
    else:
        print("\n✗ Meta^2 not found in pipeline")
        return False


if __name__ == '__main__':
    success = test_auto_mechanism_selection()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if success:
        print("\n✓ Meta^2 can automatically select learning mechanisms!")
        print("\nNext: Test compound transfer with auto-selection")
        print("  - Task A: Supervised (should use gradient)")
        print("  - Task B: RL (should use RL)")
        print("  - Task C: Hybrid (should use hybrid)")
    else:
        print("\nNeed to debug task detection or mechanism selection")
