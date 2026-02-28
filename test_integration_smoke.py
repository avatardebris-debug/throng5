"""
test_integration_smoke.py — End-to-end smoke test of the full brain stack.

Verifies that all systems work together:
  1. WholeBrain initializes with all subsystems
  2. Rehearsal Loop creates/stores action chains
  3. Planning Layer builds landmark graph and plans routes
  4. Semantic Grounding discovers objects from RAM
  5. LLM Strategy generates prompts and parses responses
  6. All reports generate without errors

Uses a simple DummyEnv — no real game needed.

Run:
    python test_integration_smoke.py
"""

from __future__ import annotations

import sys
import traceback
import numpy as np


# ── Dummy Environment ────────────────────────────────────────────────

class DummyEnv:
    """Minimal environment with save/load state support."""

    def __init__(self, n_obs: int = 84, n_actions: int = 4, ram_size: int = 128):
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.ram_size = ram_size
        self._step_count = 0
        self._state = np.zeros(n_obs, dtype=np.float32)
        self._ram = np.zeros(ram_size, dtype=np.uint8)
        self._saved_state = None
        self._saved_ram = None

        # Simulate game objects in RAM (valid indices for 128-byte RAM)
        # Player position
        self.ADDR_PX = 49    # 0x31
        self.ADDR_PY = 50    # 0x32
        # Enemy position
        self.ADDR_EX = 71    # 0x47
        self.ADDR_EY = 72    # 0x48
        # Key state
        self.ADDR_KEY = 96   # 0x60

        self._ram[self.ADDR_PX] = 50
        self._ram[self.ADDR_PY] = 100
        self._ram[self.ADDR_EX] = 120
        self._ram[self.ADDR_EY] = 80
        self._ram[self.ADDR_KEY] = 0

    @property
    def supports_save_state(self) -> bool:
        return True

    @property
    def observation_space(self):
        class Space:
            shape = (84,)
        return Space()

    @property
    def action_space(self):
        class Space:
            n = 4
        return Space()

    def reset(self) -> np.ndarray:
        self._step_count = 0
        self._state = np.random.randn(self.n_obs).astype(np.float32) * 0.1
        self._ram[self.ADDR_PX] = 50
        self._ram[self.ADDR_PY] = 100
        self._ram[self.ADDR_KEY] = 0
        return self._state.copy()

    def step(self, action: int):
        self._step_count += 1
        # Simulate movement
        if action == 0:  # right
            self._ram[self.ADDR_PX] = min(255, self._ram[self.ADDR_PX] + 2)
        elif action == 1:  # left
            self._ram[self.ADDR_PX] = max(0, self._ram[self.ADDR_PX] - 2)
        elif action == 2:  # up
            self._ram[self.ADDR_PY] = max(0, self._ram[self.ADDR_PY] - 2)
        elif action == 3:  # down
            self._ram[self.ADDR_PY] = min(255, self._ram[self.ADDR_PY] + 2)

        # Proximity reward
        dist_to_key = abs(int(self._ram[self.ADDR_PX]) - 200) + abs(int(self._ram[self.ADDR_PY]) - 80)
        reward = -0.01  # Small negative per step
        done = False

        # Key collection
        if dist_to_key < 10 and self._ram[self.ADDR_KEY] == 0:
            self._ram[self.ADDR_KEY] = 1
            reward = 100.0

        # Death
        dist_to_enemy = abs(int(self._ram[self.ADDR_PX]) - int(self._ram[self.ADDR_EX])) + abs(int(self._ram[self.ADDR_PY]) - int(self._ram[self.ADDR_EY]))
        if dist_to_enemy < 5:
            reward = -10.0
            done = True

        # Timeout
        if self._step_count > 500:
            done = True

        self._state = np.random.randn(self.n_obs).astype(np.float32) * 0.1
        self._state[0] = float(self._ram[self.ADDR_PX]) / 255.0
        self._state[1] = float(self._ram[self.ADDR_PY]) / 255.0

        return self._state.copy(), reward, done, {"ram": self._ram.copy()}

    def get_ram(self) -> np.ndarray:
        return self._ram.copy()

    def save_state(self) -> bytes:
        return bytes(self._ram.tolist()) + bytes(self._step_count.to_bytes(4, 'little'))

    def load_state(self, state: bytes) -> None:
        ram_bytes = state[:-4]
        self._ram = np.frombuffer(ram_bytes, dtype=np.uint8).copy()
        self._step_count = int.from_bytes(state[-4:], 'little')


# ── Test Functions ───────────────────────────────────────────────────

def test_brain_init():
    """Test WholeBrain initializes with all subsystems."""
    from brain.orchestrator import WholeBrain
    brain = WholeBrain(n_features=84, n_actions=4)

    assert brain.rehearsal is not None, "Rehearsal Loop not initialized"
    assert brain.planner is not None, "Planner not initialized"
    assert hasattr(brain, 'rehearse'), "rehearse() API missing"
    assert hasattr(brain, 'plan'), "plan() API missing"

    report = brain.report()
    assert "rehearsal" in report, "Rehearsal missing from report"
    assert "planning" in report, "Planning missing from report"

    return brain


def test_brain_step(brain):
    """Test basic step loop works."""
    env = DummyEnv()
    obs = env.reset()

    for i in range(20):
        result = brain.step(obs, prev_action=0, reward=0.0, done=False)
        action = result["action"] if isinstance(result, dict) else result[0]
        obs, reward, done, step_info = env.step(action)
        if done:
            obs = env.reset()
            brain.step(obs, prev_action=action, reward=reward, done=True)
            break

    return env


def test_rehearsal_bottleneck_tracker():
    """Test bottleneck tracker records deaths and identifies bottlenecks."""
    from brain.rehearsal.bottleneck_tracker import BottleneckTracker

    tracker = BottleneckTracker()
    features = np.random.randn(84).astype(np.float32)

    # Record some deaths
    for _ in range(12):
        tracker.record_death(features, {"episode_reward": -10})

    assert tracker.get_worst_bottleneck() is not None
    worst = tracker.get_worst_bottleneck()
    assert worst.deaths >= 10

    report = tracker.report()
    assert report["stuck_points"] > 0  # Should be flagged after 10 deaths


def test_rehearsal_action_chain():
    """Test action chain store/recall/promote/degrade."""
    from brain.rehearsal.action_chain import ActionChainStore

    store = ActionChainStore()
    features = np.random.randn(84).astype(np.float32)
    actions = [0, 1, 2, 3, 0, 1]

    h = store.store(features, actions, tier="compressed", success_rate=0.65)
    chain = store.recall(features)
    assert chain is not None
    assert chain.tier == "compressed"
    assert chain.confidence < 0.5  # Low confidence for compressed

    # Promote
    store.promote(h, "worldmodel", success_rate=0.70, trials=100)
    chain = store.recall(features)
    assert chain.tier == "worldmodel"
    assert chain.confidence > 0.5  # Higher after WM confirms


def test_planning_landmark_graph():
    """Test landmark graph add/route/plan."""
    from brain.planning.landmark_graph import LandmarkGraph

    graph = LandmarkGraph()
    f1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    f2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    f3 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    h1 = graph.add_landmark(f1, label="start")
    h2 = graph.add_landmark(f2, label="middle")
    h3 = graph.add_landmark(f3, label="goal", is_goal=True)

    graph.add_edge(f1, f2, actions=[0, 1, 0], confidence=9.0, success_rate=0.8)
    graph.add_edge(f2, f3, actions=[2, 3], confidence=9.0, success_rate=0.9)

    route = graph.plan_route(f1, f3)
    assert route is not None, "Should find a route from start to goal"
    assert len(route) >= 1, f"Route should have at least 1 step, got {len(route)}"

    report = graph.report()
    assert report["landmarks"] >= 3
    assert report["edges"] >= 2


def test_planning_dead_end_detector(brain):
    """Test dead end detector runs without error."""
    from brain.planning.dead_end_detector import DeadEndDetector

    detector = DeadEndDetector(brain, default_trials=10, rollout_length=5)
    features = np.random.randn(84).astype(np.float32)
    result = detector.check(features, n_trials=10)
    assert isinstance(result, bool)


def test_planning_causal_model():
    """Test causal model observes and predicts."""
    from brain.planning.causal_model import CausalModel

    model = CausalModel()
    before = np.random.randn(16).astype(np.float32)
    after = before.copy()
    after[3] += 0.5  # Feature 3 changed

    model.observe(before, action=2, features_after=after, reward=1.0)
    model.observe(before, action=2, features_after=after, reward=1.0)
    model.observe(before, action=2, features_after=after, reward=1.0)

    effect = model.predict_effects(before, action=2)
    assert effect is not None
    assert effect.observations >= 3

    report = model.report()
    assert report["total_observations"] >= 3


def test_semantic_ram_mapper():
    """Test RAM semantic mapper discovers objects."""
    from brain.planning.ram_semantic_mapper import RAMSemanticMapper

    mapper = RAMSemanticMapper(ram_size=128)
    env = DummyEnv()
    env.reset()

    # Simulate observation
    for _ in range(200):
        action = np.random.randint(4)
        ram_before = env.get_ram()
        _, reward, done, _ = env.step(action)
        ram_after = env.get_ram()

        mapper.observe(ram_after, action=action, reward=reward, done=done)
        if done:
            env.reset()

    registry = mapper.get_registry()
    assert len(registry) > 0, "Should discover some byte categories"

    subgoal_bytes = mapper.get_subgoal_bytes()
    entities = mapper.get_entity_groups()

    report = mapper.report()
    assert report["active_bytes"] > 0


def test_semantic_object_graph():
    """Test object graph entity management and description."""
    from brain.planning.object_graph import ObjectGraph

    graph = ObjectGraph()
    graph.add_entity("player", {"x": 50, "y": 100}, category="player")
    graph.add_entity("key", {"x": 200, "y": 80, "collected": False}, category="item")
    graph.add_entity("door", {"x": 30, "y": 30}, category="goal")
    graph.add_entity("dragon", {"x": 150, "y": 80}, category="enemy")

    graph.add_relation("door", "requires", "key")
    graph.add_relation("dragon", "blocks", "key")

    desc = graph.describe()
    assert "player" in desc.lower()
    assert "key" in desc.lower()
    assert "requires" in desc or "blocks" in desc

    blockers = graph.get_blockers("key")
    assert "dragon" in blockers

    reqs = graph.get_requirements("door")
    assert "key" in reqs


def test_semantic_llm_strategy(brain):
    """Test LLM strategy prompt building and parsing."""
    from brain.planning.llm_strategy import LLMStrategy
    from brain.planning.object_graph import ObjectGraph

    graph = ObjectGraph()
    graph.add_entity("player", {"x": 50, "y": 100}, category="player")
    graph.add_entity("key", {"x": 200, "y": 80, "collected": False}, category="item")
    graph.add_entity("door", {"x": 30, "y": 30}, category="goal")
    graph.add_relation("door", "requires", "key")

    strategy = LLMStrategy(brain, graph)
    result = strategy.request_plan("reach the door")

    assert "subgoals" in result
    assert len(result["subgoals"]) > 0, "Should produce at least one subgoal"
    assert result["source"] in ("llm", "fallback")

    # Test response parsing
    test_response = """1. [SUBGOAL] Collect the key (target: key)
2. [SUBGOAL] Navigate to the door (target: door)
3. [SUBGOAL] Open the door and exit"""

    subgoals = strategy.parse_subgoals(test_response)
    assert len(subgoals) == 3
    assert subgoals[0]["target"] == "key"


def test_human_recorder():
    """Test human recorder captures and analyzes frames."""
    from brain.environments.human_recorder import HumanRecorder

    recorder = HumanRecorder("test_session")
    env = DummyEnv()
    env.reset()
    recorder.start(env)

    for _ in range(50):
        action = np.random.randint(4)
        ram = env.get_ram()
        _, reward, done, _ = env.step(action)
        recorder.record(ram, action, reward, done)
        if done:
            env.reset()

    recorder.stop()
    analysis = recorder.analyze()
    assert analysis["total_frames"] == 50
    assert "position_candidates" in analysis

    sequences = recorder.get_subgoal_sequences()
    # May or may not have sequences depending on rewards


def test_full_report(brain):
    """Test that brain.report() includes all subsystems."""
    report = brain.report()
    assert "rehearsal" in report
    assert "planning" in report

    # Check rehearsal report structure
    rr = report["rehearsal"]
    assert "total_rehearsals" in rr
    assert "bottlenecks" in rr
    assert "chains" in rr

    # Check planning report structure
    pr = report["planning"]
    assert "has_plan" in pr
    assert "graph" in pr
    assert "causal_model" in pr


# ── Runner ───────────────────────────────────────────────────────────

def run_all():
    tests = [
        ("Brain Init", test_brain_init),
        ("Brain Step Loop", None),  # Needs brain
        ("Bottleneck Tracker", test_rehearsal_bottleneck_tracker),
        ("Action Chain Store", test_rehearsal_action_chain),
        ("Landmark Graph", test_planning_landmark_graph),
        ("Dead End Detector", None),  # Needs brain
        ("Causal Model", test_planning_causal_model),
        ("RAM Semantic Mapper", test_semantic_ram_mapper),
        ("Object Graph", test_semantic_object_graph),
        ("LLM Strategy", None),  # Needs brain
        ("Human Recorder", test_human_recorder),
        ("Full Report", None),  # Needs brain
    ]

    passed = 0
    failed = 0
    errors = []
    brain = None

    print("=" * 60)
    print("INTEGRATION SMOKE TEST")
    print("=" * 60)

    for name, test_fn in tests:
        try:
            if name == "Brain Init":
                brain = test_brain_init()
                print(f"  OK {name}")
                passed += 1
            elif name == "Brain Step Loop":
                test_brain_step(brain)
                print(f"  OK {name}")
                passed += 1
            elif name == "Dead End Detector":
                test_planning_dead_end_detector(brain)
                print(f"  OK {name}")
                passed += 1
            elif name == "LLM Strategy":
                test_semantic_llm_strategy(brain)
                print(f"  OK {name}")
                passed += 1
            elif name == "Full Report":
                test_full_report(brain)
                print(f"  OK {name}")
                passed += 1
            else:
                test_fn()
                print(f"  OK {name}")
                passed += 1
        except Exception as e:
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, str(e), tb))
            print(f"  FAIL {name}: {e}")

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print("\nFAILURES:")
        for name, err, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
