"""Statistical analysis tools for benchmark results."""

import numpy as np
from typing import List, Tuple
from scipy import stats


class StatisticalAnalyzer:
    """
    Provides statistical analysis for transfer learning experiments.
    
    Uses scipy.stats for robust statistical testing.
    """
    
    @staticmethod
    def compute_speedup(pretrained_steps: List[int], fresh_steps: List[int]) -> float:
        """
        Compute speedup ratio from transfer learning.
        
        Speedup = Mean(fresh) / Mean(pretrained)
        
        Args:
            pretrained_steps: Steps to convergence with transfer
            fresh_steps: Steps to convergence from scratch
            
        Returns:
            Speedup ratio (>1.0 means transfer helped)
        """
        if not pretrained_steps or not fresh_steps:
            return 0.0
        
        mean_pretrained = np.mean(pretrained_steps)
        mean_fresh = np.mean(fresh_steps)
        
        if mean_pretrained == 0:
            return 0.0
        
        return mean_fresh / mean_pretrained
    
    @staticmethod
    def t_test(pretrained: List[int], fresh: List[int]) -> Tuple[float, float]:
        """
        Perform independent samples t-test.
        
        Tests null hypothesis: pretrained and fresh have same mean.
        
        Args:
            pretrained: Steps to convergence with transfer
            fresh: Steps to convergence from scratch
            
        Returns:
            (t_statistic, p_value)
            - t_statistic: Test statistic
            - p_value: Probability of null hypothesis (p<0.05 = significant)
        """
        if len(pretrained) < 2 or len(fresh) < 2:
            return (0.0, 1.0)  # Not enough samples
        
        # Independent samples t-test
        t_stat, p_value = stats.ttest_ind(fresh, pretrained)
        
        return (float(t_stat), float(p_value))
    
    @staticmethod
    def effect_size(pretrained: List[int], fresh: List[int]) -> float:
        """
        Compute Cohen's d effect size.
        
        Measures magnitude of difference between groups.
        - d < 0.2: negligible
        - 0.2 <= d < 0.5: small
        - 0.5 <= d < 0.8: medium
        - d >= 0.8: large
        
        Args:
            pretrained: Steps to convergence with transfer
            fresh: Steps to convergence from scratch
            
        Returns:
            Cohen's d effect size
        """
        if len(pretrained) < 2 or len(fresh) < 2:
            return 0.0
        
        mean_pretrained = np.mean(pretrained)
        mean_fresh = np.mean(fresh)
        
        var_pretrained = np.var(pretrained, ddof=1)
        var_fresh = np.var(fresh, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(pretrained), len(fresh)
        pooled_std = np.sqrt(((n1 - 1) * var_pretrained + (n2 - 1) * var_fresh) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d
        d = (mean_fresh - mean_pretrained) / pooled_std
        
        return float(d)
    
    @staticmethod
    def confidence_interval(
        samples: List[float],
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean.
        
        Args:
            samples: Sample data
            alpha: Significance level (0.05 = 95% CI)
            
        Returns:
            (lower_bound, upper_bound) of confidence interval
        """
        if len(samples) < 2:
            mean = np.mean(samples) if samples else 0.0
            return (mean, mean)
        
        mean = np.mean(samples)
        sem = stats.sem(samples)  # Standard error of mean
        
        # t-distribution critical value
        df = len(samples) - 1
        t_crit = stats.t.ppf(1 - alpha/2, df)
        
        margin = t_crit * sem
        
        return (float(mean - margin), float(mean + margin))
    
    @staticmethod
    def summarize_results(
        pretrained_steps: List[int],
        fresh_steps: List[int]
    ) -> dict:
        """
        Generate comprehensive statistical summary.
        
        Args:
            pretrained_steps: Steps to convergence with transfer
            fresh_steps: Steps to convergence from scratch
            
        Returns:
            Dictionary with all statistical metrics
        """
        analyzer = StatisticalAnalyzer()
        
        speedup = analyzer.compute_speedup(pretrained_steps, fresh_steps)
        t_stat, p_value = analyzer.t_test(pretrained_steps, fresh_steps)
        effect = analyzer.effect_size(pretrained_steps, fresh_steps)
        
        pretrained_ci = analyzer.confidence_interval(pretrained_steps)
        fresh_ci = analyzer.confidence_interval(fresh_steps)
        
        return {
            'speedup': speedup,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect,
            'pretrained_mean': np.mean(pretrained_steps) if pretrained_steps else 0.0,
            'fresh_mean': np.mean(fresh_steps) if fresh_steps else 0.0,
            'pretrained_ci': pretrained_ci,
            'fresh_ci': fresh_ci,
            'n_pretrained': len(pretrained_steps),
            'n_fresh': len(fresh_steps),
            'significant': p_value < 0.05 if p_value else False,
        }
