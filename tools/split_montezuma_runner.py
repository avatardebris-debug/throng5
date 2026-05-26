"""One-off: split run_montezuma_brain.py into brain/games/montezuma/runner/."""

from pathlib import Path

root = Path(__file__).resolve().parents[1] / "run_montezuma_brain.py"
lines = root.read_text(encoding="utf-8").splitlines(keepends=True)
base = Path(__file__).resolve().parents[1] / "brain" / "games" / "montezuma" / "runner"
base.mkdir(parents=True, exist_ok=True)

const = (
    '"""Montezuma constants and RAM layout."""\n\n'
    "from __future__ import annotations\n\n"
    "from pathlib import Path\n\n"
    + "".join(lines[56:107])
)

ctrl_header = (
    '"""Montezuma training controllers."""\n\n'
    "from __future__ import annotations\n\n"
    "import numpy as np\n\n"
    "from brain.planning.landmark_graph import LandmarkGraph\n"
    "from brain.planning.goal_regression import GoalRegression\n"
    "from brain.planning.dead_end_detector import DeadEndDetector\n"
    "from brain.planning.causal_model import CausalModel\n"
    "from brain.planning.subgoal_planner import SubgoalPlanner\n"
    "from brain.planning.llm_strategy import LLMStrategy\n"
    "from brain.planning.counterfactual import CounterfactualReasoner\n"
    "from brain.planning.skill_library import SkillLibrary\n"
    "from brain.games.montezuma.runner.constants import (\n"
    "    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_ROOM, RAM_LIVES,\n"
    "    RAM_SCORE_1, RAM_SCORE_2, RAM_SKULL_X, RAM_SKULL_Y, RAM_ITEMS,\n"
    ")\n\n"
)
ctrl = ctrl_header + "".join(lines[110:423]) + "".join(lines[434:751])

setup = (
    '"""Montezuma env and planning module setup."""\n\n'
    "from __future__ import annotations\n\n"
    "import gymnasium as gym\n"
    "import numpy as np\n\n"
    "from brain.planning.ram_semantic_mapper import RAMSemanticMapper\n"
    "from brain.planning.reward_discovery import RewardDiscovery\n"
    "from brain.planning.object_graph import ObjectGraph\n"
    "from brain.planning.safety import SafetyConstraints\n"
    "from brain.planning.temporal import TemporalReasoner\n"
    "from brain.planning.skill_library import SkillLibrary\n"
    "from brain.planning.procedural_memory import ProceduralMemory\n"
    "from brain.games.montezuma.runner.constants import (\n"
    "    GAME_ID, RAM_SIZE,\n"
    "    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_ROOM, RAM_LIVES,\n"
    "    RAM_SCORE_1, RAM_SCORE_2, RAM_SKULL_X, RAM_SKULL_Y, RAM_ITEMS,\n"
    ")\n\n"
    + "".join(lines[753:797])
)

modes_header = (
    '"""Montezuma runner modes."""\n\n'
    "from __future__ import annotations\n\n"
    "import json\n"
    "import os\n"
    "import sys\n"
    "import time\n"
    "from pathlib import Path\n\n"
    "import numpy as np\n\n"
    "from brain.orchestrator import WholeBrain\n"
    "from brain.environments.human_recorder import HumanRecorder\n"
    "from brain.planning.meta_planner import MetaPlanner\n"
    "from brain.planning.self_model import SelfModel\n"
    "from brain.games.montezuma.runner.constants import (\n"
    "    GAME_ID, N_ACTIONS, RAM_SIZE, RESULTS_DIR, RECORDINGS_DIR,\n"
    "    RAM_PLAYER_X, RAM_PLAYER_Y, RAM_ROOM, RAM_LIVES,\n"
    "    RAM_SCORE_1, RAM_SCORE_2, RAM_SKULL_X, RAM_SKULL_Y, RAM_ITEMS,\n"
    "    ACTION_NAMES, KEYBOARD_MAP,\n"
    ")\n"
    "from brain.games.montezuma.runner.exploration import ExplorationManager\n"
    "from brain.games.montezuma.runner.fall_predictor import FallPredictor\n"
    "from brain.games.montezuma.runner.intentional import IntentionalController\n"
    "from brain.games.montezuma.runner.setup import get_game_state, make_modules, make_env\n\n"
)
modes = modes_header + "".join(lines[803:1691])

(base / "__init__.py").write_text('"""Montezuma WholeBrain runner package."""\n', encoding="utf-8")
(base / "constants.py").write_text(const, encoding="utf-8")
# exploration.py, fall_predictor.py, intentional.py are maintained separately (no controllers shim)
(base / "setup.py").write_text(setup, encoding="utf-8")
(base / "modes.py").write_text(modes, encoding="utf-8")
print("OK:", base)
