"""One-off script to split modes.py into modes/ package."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "brain" / "games" / "montezuma" / "runner"
src_lines = (ROOT / "modes.py").read_text(encoding="utf-8").splitlines(keepends=True)
header_end = 27

COMMON_HEADER = '''"""Montezuma runner mode: {name}."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from brain.orchestrator import WholeBrain
from brain.environments.human_recorder import HumanRecorder
from brain.planning.meta_planner import MetaPlanner
from brain.planning.self_model import SelfModel
from brain.planning.ram_semantic_mapper import RAMSemanticMapper
from brain.games.montezuma.runner.constants import (
    GAME_ID, N_ACTIONS, RAM_SIZE, RESULTS_DIR, RECORDINGS_DIR,
    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_ROOM, RAM_LIVES,
    RAM_SCORE_1, RAM_SCORE_2, RAM_SKULL_X, RAM_SKULL_Y, RAM_ITEMS,
    ACTION_NAMES, KEYBOARD_MAP,
)
from brain.games.montezuma.runner.exploration import ExplorationManager
from brain.games.montezuma.runner.fall_predictor import FallPredictor
from brain.games.montezuma.runner.intentional import IntentionalController
from brain.games.montezuma.runner.setup import get_game_state, make_modules, make_env
from brain.games.montezuma.runner.modes.shared import _save_stats, _print_final_report
'''

slices = [
    (28, 201, "human.py", False),
    (208, 340, "watch.py", True),
    (347, 493, "ground.py", True),
    (500, 679, "train.py", True),
    (686, 786, "plan.py", True),
    (793, 864, "rehearse.py", True),
    (871, 915, "shared.py", False),
]

out_dir = ROOT / "modes"
out_dir.mkdir(exist_ok=True)

for start, end, fname, use_common in slices:
    body = "".join(src_lines[start - 1 : end])
    if fname == "shared.py":
        content = (
            '"""Shared helpers for Montezuma runner modes."""\n\n'
            "from __future__ import annotations\n\n"
            "import json\n\n"
            "import numpy as np\n\n"
            "from brain.games.montezuma.runner.constants import RESULTS_DIR\n\n"
            + body
        )
    elif fname == "human.py":
        content = "".join(src_lines[:header_end]) + body
    elif use_common:
        name = fname.replace(".py", "")
        content = COMMON_HEADER.format(name=name) + "\n" + body
    else:
        content = body
    (out_dir / fname).write_text(content, encoding="utf-8")

init = '''"""Montezuma runner modes package."""

from brain.games.montezuma.runner.modes.human import mode_human
from brain.games.montezuma.runner.modes.watch import mode_watch
from brain.games.montezuma.runner.modes.ground import mode_ground
from brain.games.montezuma.runner.modes.train import mode_train
from brain.games.montezuma.runner.modes.plan import mode_plan
from brain.games.montezuma.runner.modes.rehearse import mode_rehearse

__all__ = [
    "mode_human", "mode_watch", "mode_ground", "mode_train", "mode_plan", "mode_rehearse",
]
'''
(out_dir / "__init__.py").write_text(init, encoding="utf-8")
print("split complete")
