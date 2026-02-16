"""
Run full baseline experiments in parallel.

Launches all 5 baselines × 5 runs = 25 total experiments.
Each run is 500 episodes of Breakout.
"""
import subprocess
import sys
import time
from pathlib import Path


BASELINES = [
    'tabula_rasa',
    'maml_only', 
    'static_concepts',
    'llm_at_start',
    'full_system'
]

EPISODES = 500
RUNS = 5


def run_experiment(baseline: str, run_num: int):
    """Launch a single experiment run."""
    cmd = [
        sys.executable,
        "run_baseline_experiment.py",
        "--game", "Breakout",
        "--baseline", baseline,
        "--episodes", str(EPISODES),
        "--runs", "1",
        "--output-dir", f"experiments/full_breakout"
    ]
    
    log_file = Path(f"experiments/logs/{baseline}_run{run_num}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting: {baseline} run {run_num}")
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return proc, log_file


def main():
    print("="*70)
    print("FULL META-LEARNING EXPERIMENT")
    print("="*70)
    print(f"Baselines: {len(BASELINES)}")
    print(f"Runs per baseline: {RUNS}")
    print(f"Episodes per run: {EPISODES}")
    print(f"Total experiments: {len(BASELINES) * RUNS}")
    print("="*70)
    
    # Launch all experiments
    processes = []
    
    for baseline in BASELINES:
        for run in range(1, RUNS + 1):
            proc, log_file = run_experiment(baseline, run)
            processes.append((baseline, run, proc, log_file))
            time.sleep(2)  # Stagger starts
    
    print(f"\n✅ Launched {len(processes)} experiments")
    print("\nMonitoring progress...")
    print("(This will take several hours)")
    
    # Monitor progress
    completed = 0
    while completed < len(processes):
        time.sleep(30)  # Check every 30 seconds
        
        for baseline, run, proc, log_file in processes:
            if proc.poll() is not None:  # Process finished
                if proc.returncode == 0:
                    print(f"  ✅ {baseline} run {run} COMPLETE")
                else:
                    print(f"  ❌ {baseline} run {run} FAILED (code {proc.returncode})")
                completed += 1
        
        print(f"Progress: {completed}/{len(processes)} complete")
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults saved to: experiments/full_breakout/")
    print("Logs saved to: experiments/logs/")


if __name__ == "__main__":
    main()
