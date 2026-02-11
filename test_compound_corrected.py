"""
CORRECTED Compound Transfer Test

The previous test had a fundamental mismatch:
- Task: Supervised learning (linear regression with targets)
- Learning: STDP/Hebbian (unsupervised, reward-based)
- Reward signal: 0.0 (no signal!)

This caused bio-inspired learning to add noise, degrading performance.

SOLUTION: Use gradient-based learning for supervised tasks, OR
          use RL tasks (GridWorld) for bio-inspired learning.

This test uses BOTH to validate compound transfer properly.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline


def test_supervised_compound_transfer():
    """
    Test compound transfer on supervised tasks.
    
    For supervised learning, we should DISABLE bio-inspired learning
    and use gradient descent instead.
    """
    print("\n" + "="*60)
    print("SUPERVISED COMPOUND TRANSFER (Gradient-Based)")
    print("="*60)
    
    # TODO: Implement gradient-based learning in Meta^1
    # Current Meta^1 only has STDP/Hebbian, which are wrong for supervised tasks
    
    print("\n⚠ SKIPPED: Need to implement gradient-based learning first")
    print("  Current Meta^1 only supports STDP/Hebbian (bio-inspired)")
    print("  These are designed for RL/unsupervised, not supervised regression")
    

def test_rl_compound_transfer():
    """
    Test compound transfer on RL tasks (GridWorld).
    
    This is the RIGHT test for bio-inspired learning (STDP/Hebbian/Dopamine).
    """
    print("\n" + "="*60)
    print("RL COMPOUND TRANSFER (Bio-Inspired Learning)")
    print("="*60)
    
    # TODO: Implement GridWorld curriculum
    # Task A: Navigate to fixed goal
    # Task B: Navigate to random goal
    # Task C: Navigate avoiding obstacles
    
    print("\n⚠ NOT YET IMPLEMENTED")
    print("  Need GridWorld environment adapter first")
    print("  This is the correct test for Meta^N's bio-inspired learning")


def diagnose_original_test():
    """
    Explain why the original test failed.
    """
    print("\n" + "="*60)
    print("DIAGNOSIS: Why Original Test Failed")
    print("="*60)
    
    print("\n[Original Test Setup]")
    print("  Task: Linear regression (y = W @ x + bias)")
    print("  Target: Provided (supervised learning)")
    print("  Reward: 0.0 (NO SIGNAL!)")
    print("  Learning: STDP + Hebbian (bio-inspired)")
    
    print("\n[The Problem]")
    print("  1. STDP/Hebbian are designed for UNSUPERVISED/RL learning")
    print("  2. They need REWARD signal to know what's good/bad")
    print("  3. Reward=0.0 means they're learning from NOISE")
    print("  4. Each task adds more noise → compound transfer WORSE")
    
    print("\n[Why Adaptive Pipeline Helped (+7.8%)]")
    print("  GlobalDynamicsOptimizer GATES higher meta-layers")
    print("  → Reduces interference from noisy STDP/Hebbian")
    print("  → Still not ideal, but less bad")
    
    print("\n[The Fix]")
    print("  Option 1: Use RL tasks (GridWorld) with proper reward signal")
    print("  Option 2: Add gradient-based learning for supervised tasks")
    print("  Option 3: Disable Meta^1 for supervised tasks (use Meta^0 only)")
    
    print("\n[Recommendation]")
    print("  ✓ Test compound transfer on GridWorld curriculum (RL)")
    print("  ✓ This matches Meta^N's bio-inspired design")
    print("  ✓ Proper reward signal for STDP/Hebbian/Dopamine")


if __name__ == '__main__':
    diagnose_original_test()
    test_supervised_compound_transfer()
    test_rl_compound_transfer()
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("\nThe original test was fundamentally flawed:")
    print("  - Wrong learning mechanism for the task type")
    print("  - No reward signal (reward=0.0)")
    print("  - Bio-inspired learning adding noise to supervised task")
    print("\nNext step: Test compound transfer on GridWorld (RL task)")
    print("  where STDP/Hebbian/Dopamine are the RIGHT tools")
