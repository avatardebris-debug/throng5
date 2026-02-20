"""
Path configuration for the throng4 agent pipeline.

All runtime paths are resolved from two roots so nothing is hardcoded
in multiple places. Override with environment variables if needed.

    OPENCLAW_WORKSPACE_ROOT  (default: ~/.openclaw)
    THRONG_ROOT              (default: auto-detected repo root)
"""

import os
from pathlib import Path


def _find_throng_root() -> Path:
    """Walk up from this file until we find train_tetris_curriculum.py."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "train_tetris_curriculum.py").exists():
            return parent
    # Fallback: cwd
    return Path.cwd()


# ------------------------------------------------------------------
# Roots — override via environment variables
# ------------------------------------------------------------------
OPENCLAW_ROOT: Path = Path(
    os.environ.get("OPENCLAW_WORKSPACE_ROOT", str(Path.home() / ".openclaw"))
)

THRONG_ROOT: Path = Path(
    os.environ.get("THRONG_ROOT", str(_find_throng_root()))
)

# ------------------------------------------------------------------
# Derived paths
# ------------------------------------------------------------------

# Tetra reads/writes hypothesis files here
MEMORY_DIR: Path = OPENCLAW_ROOT / "workspace" / "memory"

# Persistent rule libraries (one JSON per game)
RULES_DIR: Path = OPENCLAW_ROOT / "rules"

# Game trajectory logs written by the RL agent
LOGS_DIR: Path = THRONG_ROOT / "atari_logs"

# Eval audit outputs
AUDITS_DIR: Path = THRONG_ROOT / "eval_audits"

# Required JSON keys Tetra must include for a valid hypothesis response
REQUIRED_HYPOTHESIS_KEYS = {"id", "description", "object", "feature", "direction", "confidence"}


def ensure_dirs():
    """Create all required runtime directories."""
    for d in (MEMORY_DIR, RULES_DIR, LOGS_DIR, AUDITS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def rules_path(game_id: str) -> Path:
    """Return the rules JSON path for a given game_id."""
    safe_name = game_id.replace("/", "_").replace("\\", "_")
    return RULES_DIR / f"{safe_name}_rules.json"


def summary():
    """Print a summary of the current path config."""
    print("=== throng4 path config ===")
    print(f"  OPENCLAW_ROOT : {OPENCLAW_ROOT}")
    print(f"  THRONG_ROOT   : {THRONG_ROOT}")
    print(f"  MEMORY_DIR    : {MEMORY_DIR}")
    print(f"  RULES_DIR     : {RULES_DIR}")
    print(f"  LOGS_DIR      : {LOGS_DIR}")
    print(f"  AUDITS_DIR    : {AUDITS_DIR}")


if __name__ == "__main__":
    ensure_dirs()
    summary()
