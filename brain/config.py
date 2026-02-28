"""
config.py — Central configuration for Throng 5 Brain module.

All paths, constants, and system-wide settings in one place.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

BRAIN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BRAIN_ROOT.parent  # throng5/

LOG_DIR = PROJECT_ROOT / "logs"
TELEMETRY_DIR = LOG_DIR / "telemetry"
TRAINING_LOG_DIR = LOG_DIR / "training"
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

# Reference code (read-only)
THRONG3_COPY_DIR = PROJECT_ROOT / "throng3 - Copy"
THRONG4_NEW_DIR = PROJECT_ROOT / "throng4_new"
DEEP_RL_ZOO_DIR = THRONG3_COPY_DIR / "deep_rl_zoo-main"
THRONG2_ARCHIVE_DIR = THRONG3_COPY_DIR / "archive" / "2026-02-20" / "throng2-master-copy"
THRONG35_REGIONS_DIR = THRONG4_NEW_DIR / "throng35" / "regions"

# ROM directories (drop .nes/.sfc/.md files here)
ROMS_DIR = PROJECT_ROOT / "roms"
NES_ROMS_DIR = ROMS_DIR / "nes"
SNES_ROMS_DIR = ROMS_DIR / "snes"
GENESIS_ROMS_DIR = ROMS_DIR / "genesis"

# ── Feature System ────────────────────────────────────────────────────

CORE_FEATURE_SIZE = 20
EXT_FEATURE_MAX = 32
ABSTRACT_VEC_SIZE = 84  # CORE + EXT + MASK = 20 + 32 + 32

# ── Brain Region Defaults ─────────────────────────────────────────────

DEFAULT_MESSAGE_HISTORY_SIZE = 500
EMERGENCY_HALT_TIMEOUT_SEC = 5.0  # Max time a region stays halted before auto-resume
FAST_PATH_BUDGET_MS = 16.7        # ~60fps — max time for subconscious fast path
SLOW_PATH_BUDGET_MS = 1000.0      # 1 second — max for conscious processing per step

# ── Overnight Loop ────────────────────────────────────────────────────

OVERNIGHT_REPLAY_BATCH_SIZE = 64
OVERNIGHT_DREAM_STEPS = 20
OVERNIGHT_LLM_COOLDOWN_SEC = 300  # 5 min between LLM calls during dream loop

# ── Version ───────────────────────────────────────────────────────────

VERSION = "5.0.0-alpha"
PHASE = "phase1-foundation"
