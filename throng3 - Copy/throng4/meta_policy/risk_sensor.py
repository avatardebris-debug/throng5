"""
RiskSensor — Plateau detection and risk assessment.

Throng5 role: Risk Evaluator — assesses long-term position and danger.
Owns all plateau/performance-trend analysis.
"""

from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RiskConfig:
    """Risk sensor configuration."""
    plateau_window: int = 15
    plateau_threshold: float = 0.05


class RiskSensor:
    """
    Detects plateaus, estimates risk, and recommends retirement.
    
    In Throng5, this becomes the Risk Evaluator:
      - Position evaluation independent of immediate reward
      - Offense vs defense scoring
      - Long-term condition assessment
    """
    
    def __init__(self, plateau_window: int = 15, plateau_threshold: float = 0.05):
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
    
    def is_plateauing(self, rewards: deque) -> bool:
        """
        Detect plateau from reward history.
        
        Compares recent window average to previous window average.
        Returns True if improvement < threshold.
        """
        if len(rewards) < self.plateau_window * 2:
            return False
        
        reward_list = list(rewards)
        recent = np.mean(reward_list[-self.plateau_window:])
        previous = np.mean(reward_list[-2*self.plateau_window:-self.plateau_window])
        
        if abs(previous) < 1e-8:
            return abs(recent) < 1e-8  # Both near zero = plateau
        
        improvement = (recent - previous) / abs(previous)
        return improvement < self.plateau_threshold
    
    def plateau_duration(self, rewards: deque) -> int:
        """How many episodes since last significant improvement."""
        if len(rewards) < 10:
            return 0
        
        reward_list = list(rewards)
        best_so_far = float('-inf')
        duration = 0
        
        for r in reversed(reward_list):
            if r > best_so_far * 1.1:  # 10% improvement
                break
            duration += 1
            best_so_far = max(best_so_far, r)
        
        return duration
    
    def risk_level(self, rewards: deque) -> str:
        """
        Assess current risk level.
        
        Returns: 'stable', 'plateaued', 'declining', 'critical'
        """
        if len(rewards) < 20:
            return 'stable'
        
        if not self.is_plateauing(rewards):
            return 'stable'
        
        duration = self.plateau_duration(rewards)
        
        if duration > 50:
            return 'critical'
        elif duration > 30:
            return 'declining'
        else:
            return 'plateaued'
