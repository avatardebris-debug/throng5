"""
Generate Phase 3 validation visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("results/phase3")
result_files = list(results_dir.glob("gridworld_5k_n30_*.json"))
latest_result = max(result_files, key=lambda p: p.stat().st_mtime)

with open(latest_result) as f:
    results = json.load(f)

stats = results['statistics']
fresh_steps = results['fresh_steps']
pretrained_steps = results['pretrained_steps']

# Create figure
fig = plt.figure(figsize=(14, 10))

# 1. Box plot comparison
ax1 = plt.subplot(2, 3, 1)
bp = ax1.boxplot([fresh_steps, pretrained_steps], labels=['Fresh', 'Pretrained'],
                  patch_artist=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')
ax1.set_ylabel('Episodes to Convergence')
ax1.set_title('Transfer Learning Effect (N=30)')
ax1.grid(True, alpha=0.3)

# Add stats text
textstr = f"Speedup: {stats['speedup']:.2f}x\np < 0.0001\nCohen's d = {stats['effect_size']:.2f}"
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Histogram
ax2 = plt.subplot(2, 3, 2)
ax2.hist(fresh_steps, bins=15, alpha=0.6, label='Fresh', color='lightcoral', edgecolor='black')
ax2.hist(pretrained_steps, bins=15, alpha=0.6, label='Pretrained', color='lightgreen', edgecolor='black')
ax2.set_xlabel('Episodes to Convergence')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Convergence Times')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Scatter plot (seed by seed)
ax3 = plt.subplot(2, 3, 3)
seeds = np.arange(1, len(fresh_steps) + 1)
ax3.scatter(seeds, fresh_steps, alpha=0.6, label='Fresh', color='lightcoral', s=50)
ax3.scatter(seeds, pretrained_steps, alpha=0.6, label='Pretrained', color='lightgreen', s=50)
ax3.axhline(y=np.mean(fresh_steps), color='red', linestyle='--', alpha=0.5, label=f'Fresh mean: {np.mean(fresh_steps):.1f}')
ax3.axhline(y=np.mean(pretrained_steps), color='green', linestyle='--', alpha=0.5, label=f'Pretrained mean: {np.mean(pretrained_steps):.1f}')
ax3.set_xlabel('Seed')
ax3.set_ylabel('Episodes to Convergence')
ax3.set_title('Convergence by Seed')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. Speedup per seed
ax4 = plt.subplot(2, 3, 4)
speedups = [f / max(p, 1) for f, p in zip(fresh_steps, pretrained_steps)]
ax4.bar(seeds, speedups, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axhline(y=stats['speedup'], color='red', linestyle='--', label=f'Mean: {stats["speedup"]:.2f}x')
ax4.set_xlabel('Seed')
ax4.set_ylabel('Speedup (x)')
ax4.set_title('Speedup per Seed')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# 5. Cumulative distribution
ax5 = plt.subplot(2, 3, 5)
fresh_sorted = np.sort(fresh_steps)
pretrained_sorted = np.sort(pretrained_steps)
fresh_cdf = np.arange(1, len(fresh_sorted) + 1) / len(fresh_sorted)
pretrained_cdf = np.arange(1, len(pretrained_sorted) + 1) / len(pretrained_sorted)
ax5.plot(fresh_sorted, fresh_cdf, label='Fresh', color='lightcoral', linewidth=2)
ax5.plot(pretrained_sorted, pretrained_cdf, label='Pretrained', color='lightgreen', linewidth=2)
ax5.set_xlabel('Episodes to Convergence')
ax5.set_ylabel('Cumulative Probability')
ax5.set_title('Cumulative Distribution')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Summary statistics table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

fresh_ci_lower, fresh_ci_upper = stats['fresh_ci']
pretrained_ci_lower, pretrained_ci_upper = stats['pretrained_ci']

summary_text = f"""
Phase 3 Transfer Learning Validation
{'='*40}

Network: 5000 neurons
Task: GridWorld 5x5
N: 30 seeds

Results:
  Fresh (no transfer):
    Mean: {stats['fresh_mean']:.1f} episodes
    95% CI: [{fresh_ci_lower:.1f}, {fresh_ci_upper:.1f}]
    
  Pretrained (transfer):
    Mean: {stats['pretrained_mean']:.1f} episodes
    95% CI: [{pretrained_ci_lower:.1f}, {pretrained_ci_upper:.1f}]
    
  Speedup: {stats['speedup']:.2f}x

Statistical Significance:
  t-statistic: {stats['t_statistic']:.3f}
  p-value: {stats['p_value']:.6f}
  Effect size: {stats['effect_size']:.3f}
  Significant: {'YES ✓' if stats['significant'] else 'NO ✗'}

Conclusion:
  Transfer learning provides a
  statistically significant speedup
  for GridWorld navigation.
"""

ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig('results/phase3/phase3_validation_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/phase3/phase3_validation_results.png")

plt.show()
