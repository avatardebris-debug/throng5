"""
Analyze baseline experiment logs and extract results.
"""
import re
from pathlib import Path
from collections import defaultdict
import json


def parse_log(log_path: Path) -> dict:
    """Parse a baseline experiment log file."""
    results = {
        'baseline': '',
        'episodes': 0,
        'rewards': [],
        'errors': [],
        'status': 'unknown'
    }
    
    try:
        text = log_path.read_text(errors='replace')
    except Exception as e:
        results['status'] = f'read_error: {e}'
        return results
    
    # Extract baseline name from filename
    name = log_path.stem  # e.g. tabula_rasa_run1
    parts = name.rsplit('_run', 1)
    results['baseline'] = parts[0]
    results['run'] = int(parts[1]) if len(parts) > 1 else 0
    
    # Check for errors
    if 'Error' in text or 'Traceback' in text:
        results['status'] = 'error'
        # Get last error line
        for line in text.split('\n'):
            if 'Error' in line:
                results['errors'].append(line.strip())
    
    # Look for episode rewards
    # Pattern: "Ep X: reward=Y" or "Episode X: Y" etc.
    reward_pattern = re.compile(r'(?:reward|score)[=:\s]+([0-9.]+)', re.IGNORECASE)
    for match in reward_pattern.finditer(text):
        try:
            results['rewards'].append(float(match.group(1)))
        except ValueError:
            pass
    
    # Look for final results
    if 'Episodes: 500' in text:
        results['status'] = 'completed'
        results['episodes'] = 500
    elif 'Episodes:' in text:
        ep_match = re.search(r'Episodes:\s*(\d+)', text)
        if ep_match:
            results['episodes'] = int(ep_match.group(1))
            results['status'] = 'completed'
    
    # Check for specific patterns
    mean_match = re.search(r'Mean.*?([0-9.]+)', text)
    if mean_match:
        results['mean_reward'] = float(mean_match.group(1))
    
    return results


def main():
    log_dir = Path('experiments/logs')
    
    if not log_dir.exists():
        print("No experiment logs found!")
        return
    
    print("=" * 70)
    print("BASELINE EXPERIMENT RESULTS ANALYSIS")
    print("=" * 70)
    
    # Group by baseline
    baselines = defaultdict(list)
    
    for log_file in sorted(log_dir.glob('*.log')):
        result = parse_log(log_file)
        baselines[result['baseline']].append(result)
    
    for baseline, runs in sorted(baselines.items()):
        print(f"\n{'='*50}")
        print(f"Baseline: {baseline}")
        print(f"{'='*50}")
        
        completed = sum(1 for r in runs if r['status'] == 'completed')
        errored = sum(1 for r in runs if r['status'] == 'error')
        
        print(f"  Runs: {len(runs)} ({completed} completed, {errored} errored)")
        
        for run in runs:
            status_icon = "✅" if run['status'] == 'completed' else "❌"
            print(f"  {status_icon} Run {run.get('run', '?')}: "
                  f"status={run['status']}, "
                  f"episodes={run['episodes']}, "
                  f"rewards_found={len(run['rewards'])}")
            if run['errors']:
                for err in run['errors'][:2]:
                    print(f"      Error: {err[:80]}")
    
    # Also check JSON results
    json_dir = Path('experiments/test_baselines')
    if json_dir.exists():
        print(f"\n{'='*50}")
        print("JSON Results (from validation run)")
        print(f"{'='*50}")
        
        for json_file in sorted(json_dir.glob('*.json')):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                print(f"\n  {json_file.stem}:")
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float, str)):
                            print(f"    {k}: {v}")
                        elif isinstance(v, list) and len(v) <= 20:
                            if v and isinstance(v[0], (int, float)):
                                import numpy as np
                                print(f"    {k}: mean={np.mean(v):.2f} ± {np.std(v):.2f}")
            except Exception as e:
                print(f"  {json_file.stem}: Error reading: {e}")
    
    # Also check full_stack results
    fullstack_dir = Path('experiments/full_stack')
    if fullstack_dir.exists():
        print(f"\n{'='*50}")
        print("Full Stack Results")
        print(f"{'='*50}")
        for f in fullstack_dir.glob('*.json'):
            try:
                data = json.load(open(f))
                print(f"\n  {f.stem}:")
                print(f"    {json.dumps(data, indent=4, default=str)}")
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    main()
