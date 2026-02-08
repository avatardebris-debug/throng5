"""Visualization utilities for benchmark results."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional


def plot_speedup_comparison(results_path: str, output_dir: str = None):
    """
    Create bar chart comparing pretrained vs fresh training.
    
    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save figure (default: same as results)
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    categories = ['Fresh\n(No Transfer)', 'Pretrained\n(With Transfer)']
    means = [stats['fresh_mean'], stats['pretrained_mean']]
    
    # Error bars (95% CI)
    fresh_ci = stats['fresh_ci']
    pretrained_ci = stats['pretrained_ci']
    errors = [
        [stats['fresh_mean'] - fresh_ci[0], pretrained_ci[1] - stats['pretrained_mean']],
        [fresh_ci[1] - stats['fresh_mean'], stats['pretrained_mean'] - pretrained_ci[0]]
    ]
    
    # Plot bars
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(categories, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add error bars
    ax.errorbar(categories, means, yerr=errors, fmt='none', color='black', 
                capsize=10, capthick=2, linewidth=2)
    
    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f} steps',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add speedup annotation
    speedup = stats['speedup']
    ax.text(0.5, max(means) * 0.9, f'Speedup: {speedup:.2f}x',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Styling
    ax.set_ylabel('Steps to Convergence', fontsize=14, fontweight='bold')
    ax.set_title('Transfer Learning Performance', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add significance indicator
    if stats['significant']:
        sig_text = f"✓ Statistically Significant (p={stats['p_value']:.4f})"
        color = 'green'
    else:
        sig_text = f"✗ Not Significant (p={stats['p_value']:.4f})"
        color = 'red'
    
    ax.text(0.5, -0.15, sig_text, ha='center', fontsize=12, 
            color=color, fontweight='bold', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'speedup_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def plot_statistical_summary(results_path: str, output_dir: str = None):
    """
    Create visualization of statistical metrics.
    
    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save figure
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Speedup
    ax = axes[0]
    speedup = stats['speedup']
    colors = ['green' if speedup > 1.5 else 'orange' if speedup > 1.0 else 'red']
    ax.bar(['Speedup'], [speedup], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=1.5, color='green', linestyle='--', linewidth=2, label='Target (1.5x)')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (1.0x)')
    ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax.set_title('Transfer Speedup', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 2. P-value
    ax = axes[1]
    p_value = stats['p_value']
    colors = ['green' if p_value < 0.05 else 'red']
    ax.bar(['P-value'], [p_value], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=0.05, color='green', linestyle='--', linewidth=2, label='Significance (α=0.05)')
    ax.set_ylabel('P-value', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Significance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(0.1, p_value * 1.2)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Effect Size
    ax = axes[2]
    effect_size = stats['effect_size']
    # Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large
    if effect_size >= 0.8:
        color = 'green'
        label = 'Large'
    elif effect_size >= 0.5:
        color = 'orange'
        label = 'Medium'
    elif effect_size >= 0.2:
        color = 'yellow'
        label = 'Small'
    else:
        color = 'red'
        label = 'Negligible'
    
    ax.bar(['Effect Size'], [effect_size], color=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Large (0.8)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium (0.5)')
    ax.axhline(y=0.2, color='yellow', linestyle='--', linewidth=1, alpha=0.5, label='Small (0.2)')
    ax.set_ylabel("Cohen's d", fontsize=12, fontweight='bold')
    ax.set_title(f'Effect Size ({label})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'statistical_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def plot_distribution_comparison(results_path: str, output_dir: str = None):
    """
    Create histogram comparing distributions of convergence times.
    
    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save figure
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    pretrained = data['pretrained_steps']
    fresh = data['fresh_steps']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    bins = np.linspace(min(min(pretrained), min(fresh)), 
                       max(max(pretrained), max(fresh)), 15)
    
    ax.hist(fresh, bins=bins, alpha=0.6, color='#e74c3c', 
            label='Fresh (No Transfer)', edgecolor='black', linewidth=1.5)
    ax.hist(pretrained, bins=bins, alpha=0.6, color='#2ecc71', 
            label='Pretrained (With Transfer)', edgecolor='black', linewidth=1.5)
    
    # Add mean lines
    ax.axvline(np.mean(fresh), color='#e74c3c', linestyle='--', 
               linewidth=2, label=f'Fresh Mean: {np.mean(fresh):.1f}')
    ax.axvline(np.mean(pretrained), color='#2ecc71', linestyle='--', 
               linewidth=2, label=f'Pretrained Mean: {np.mean(pretrained):.1f}')
    
    # Styling
    ax.set_xlabel('Steps to Convergence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Convergence Times', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'distribution_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def generate_all_figures(results_path: str, output_dir: str = None):
    """
    Generate all visualization figures for a results file.
    
    Args:
        results_path: Path to results JSON file
        output_dir: Directory to save figures (default: same as results)
    """
    print(f"\nGenerating visualizations for: {results_path}\n")
    
    plot_speedup_comparison(results_path, output_dir)
    plot_statistical_summary(results_path, output_dir)
    plot_distribution_comparison(results_path, output_dir)
    
    print("\n✅ All visualizations generated!\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <results_json_path> [output_dir]")
        sys.exit(1)
    
    results_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    generate_all_figures(results_path, output_dir)
