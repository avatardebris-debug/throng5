"""
Throng4 — Unified Compute Graph Architecture

Meta^N recursive self-optimization with dual-head ANN substrate.
"""

__version__ = '4.0.0-alpha'

from throng4.layers import ANNLayer
from throng4.learning import DQNLearner, DQNConfig

__all__ = ['ANNLayer', 'DQNLearner', 'DQNConfig']
