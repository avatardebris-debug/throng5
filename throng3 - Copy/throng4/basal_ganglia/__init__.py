"""
Basal Ganglia — Dream simulation engine for hypothesis testing.

Bridge Step 4: Ports throng2's SNN as a lightweight "dreamer" that
simulates multiple hypotheses/policies in parallel.

Components:
  - CompressedStateEncoder: Converts observations to lightweight representations
  - DreamerEngine: Runs parallel SNN-based world model simulations
  - Amygdala: Detects danger from dream results, triggers policy overrides
  - DreamerTeacher: Teaches main policy from dream insights (Options framework)
"""

from throng4.basal_ganglia.compressed_state import CompressedStateEncoder
from throng4.basal_ganglia.dreamer_engine import DreamerEngine, DreamResult
from throng4.basal_ganglia.amygdala import Amygdala, DangerSignal
from throng4.basal_ganglia.hypothesis_profiler import (
    DreamerTeacher,
    HypothesisProfile,
    OptionsLibrary,
    BehavioralOption,
    TeachingSignal,
)

__all__ = [
    'CompressedStateEncoder',
    'DreamerEngine',
    'DreamResult',
    'Amygdala',
    'DangerSignal',
    'DreamerTeacher',
    'HypothesisProfile',
    'OptionsLibrary',
    'BehavioralOption',
    'TeachingSignal',
]

