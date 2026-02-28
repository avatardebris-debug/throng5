"""
throng4.runners — FastLoop (high-speed sim) and SlowLoop (offline consolidation).
"""
from .fast_loop import FastLoop
from .slow_loop import SlowLoop, ConsolidationReport

__all__ = ['FastLoop', 'SlowLoop', 'ConsolidationReport']
