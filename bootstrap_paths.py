"""
bootstrap_paths.py — Put throng4 (and related packages) on sys.path for throng5.

The canonical full throng4 tree lives under ``throng3 - Copy/throng4/``
(llm_policy, abstract_features, config, etc.). A slimmer copy exists under
``throng4_new/throng4/``. This module prepends those roots once so imports
like ``from throng4.learning.abstract_features import ...`` work from the
repo root without manual PYTHONPATH setup.

Path order is CI-sensitive: changing which tree wins can break tests that
expect specific throng4 modules (e.g. ``llm_policy``). Keep the complete
``throng3 - Copy`` tree first; ``throng4_new`` is a slimmer fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# First match wins for duplicate package names; prefer the complete tree.
_PATH_CANDIDATES = (
    _ROOT / "throng3 - Copy",
    _ROOT / "throng4_new",
)


def ensure_throng_paths() -> None:
    # Insert in reverse so the first candidate ends up at sys.path[0].
    for candidate in reversed(_PATH_CANDIDATES):
        if candidate.is_dir():
            entry = str(candidate)
            if entry not in sys.path:
                sys.path.insert(0, entry)
