"""
Test all 5 baselines with a short run (20 episodes each).

This validates that all baselines work before running full experiments.
"""
import subprocess
import sys


BASELINES = [
    'tabula_rasa',
    'maml_only',
    'static_concepts',
    'llm_at_start',
    'full_system'
]


def test_baseline(baseline: str, episodes: int = 20):
    """Test a single baseline."""
    print(f"\n{'='*70}")
    print(f"Testing: {baseline}")
    print(f"{'='*70}\n")
    
    cmd = [
        sys.executable,
        "run_baseline_experiment.py",
        "--game", "Breakout",
        "--baseline", baseline,
        "--episodes", str(episodes),
        "--runs", "1",
        "--output-dir", "experiments/test_baselines"
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"\n✅ {baseline} test PASSED")
        return True
    else:
        print(f"\n❌ {baseline} test FAILED")
        return False


def main():
    print("="*70)
    print("BASELINE VALIDATION TEST")
    print("Testing all 5 baselines with 20 episodes each")
    print("="*70)
    
    results = {}
    
    for baseline in BASELINES:
        try:
            success = test_baseline(baseline, episodes=20)
            results[baseline] = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"\n❌ {baseline} CRASHED: {e}")
            results[baseline] = "CRASH"
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for baseline, status in results.items():
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {baseline:25s} {status}")
    
    passed = sum(1 for s in results.values() if s == "PASS")
    total = len(results)
    
    print(f"\n{passed}/{total} baselines passed")
    
    if passed == total:
        print("\n🎉 All baselines working! Ready for full experiments.")
    else:
        print("\n⚠️  Some baselines failed. Fix before running full experiments.")


if __name__ == "__main__":
    main()
