"""
orchestrator_wiring.py — Optional subsystem factory for WholeBrain.

Builds planning, learning, telemetry, dream, and rehearsal subsystems from config
flags. WholeBrain.__init__ delegates here after core regions are constructed.

Striatum OC context lives in brain/regions/oc_context.py (striatum facade).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from brain.orchestrator import WholeBrain


DEFAULT_ENABLED_SYSTEMS: Dict[str, bool] = {
    "world_model": True,
    "dreams": True,
    "causal_model": True,
    "skill_library": True,
    "attribution": True,
    "stage_classifier": True,
    "counterfactual": True,
    "hippocampus_store": True,
    "threat_gating": True,
    "probe_runner": True,
    "rehearsal_loop": True,
}


def _get_counterfactual():
    try:
        from brain.planning.counterfactual import CounterfactualReasoner
        return CounterfactualReasoner
    except ImportError:
        return None


def _attach_subsystem(
    brain: "WholeBrain",
    attr: str,
    factory: Callable[[], Any],
    *,
    enabled: bool = True,
    error_key: Optional[str] = None,
    default: Any = None,
) -> Any:
    """Attach optional subsystem; record failures in brain._init_errors."""
    key = error_key or attr
    if not enabled:
        setattr(brain, attr, default)
        return default
    try:
        obj = factory()
        setattr(brain, attr, obj)
        return obj
    except Exception as exc:
        brain._init_errors[key] = str(exc)
        setattr(brain, attr, default)
        return default


def _run_optional(
    brain: "WholeBrain",
    error_key: str,
    fn: Callable[[], Any],
    *,
    enabled: bool = True,
) -> Any:
    """Run side-effect init (no brain attr); record failures in _init_errors."""
    if not enabled:
        return None
    try:
        return fn()
    except Exception as exc:
        brain._init_errors[error_key] = str(exc)
        return None


def wire_subsystems(
    brain: "WholeBrain",
    *,
    session_name: str,
    n_features: int,
    n_actions: int,
    use_cnn: bool,
    use_torch: bool,
    enabled_systems: Optional[Dict[str, bool]] = None,
) -> None:
    """Attach optional subsystems to an initialized WholeBrain (regions + bus)."""
    brain._enabled = {**DEFAULT_ENABLED_SYSTEMS, **(enabled_systems or {})}
    brain._init_errors = {}

    _attach_subsystem(
        brain, "dreamer",
        lambda: __import__(
            "brain.learning.world_model", fromlist=["DreamerWorldModel"]
        ).DreamerWorldModel(
            n_features=n_features, n_actions=n_actions,
            latent_dim=64, hidden_dim=256,
        ),
    )

    _run_optional(
        brain, "successor_repr",
        lambda: brain.hippocampus.enable_sr(
            n_features=n_features, n_actions=n_actions,
        ),
    )

    if use_torch:
        _run_optional(
            brain, "option_critic",
            lambda: brain.striatum.enable_option_critic(n_options=4),
        )

    brain._wire_striatum_learning()

    # CNN → striatum end-to-end
    if use_cnn and use_torch and brain.sensory._use_cnn:
        cnn_params = brain.sensory.get_cnn_parameters()
        if cnn_params:
            brain.striatum.wire_cnn_encoder(
                brain.sensory.encode_for_training,
                cnn_params,
            )

    # DQN policy → dreamer
    if use_torch and brain.striatum._torch_dqn is not None:
        brain.basal_ganglia.set_policy_fn(brain.striatum._torch_dqn.select_action)

    _attach_subsystem(
        brain, "probe_runner",
        lambda: __import__(
            "brain.learning.probe_runner", fromlist=["ProbeRunner"]
        ).ProbeRunner(brain, probe_steps=500),
        enabled=brain._enabled["probe_runner"],
    )

    _attach_subsystem(
        brain, "stage_classifier",
        lambda: __import__(
            "brain.learning.stage_classifier", fromlist=["StageClassifier"]
        ).StageClassifier(n_features=n_features),
        enabled=brain._enabled["stage_classifier"],
    )

    def _make_planner():
        from brain.planning.landmark_graph import LandmarkGraph
        from brain.planning.causal_model import CausalModel
        from brain.planning.goal_regression import GoalRegression
        from brain.planning.dead_end_detector import DeadEndDetector
        from brain.planning.subgoal_planner import SubgoalPlanner

        graph = LandmarkGraph()
        causal = CausalModel()
        regressor = GoalRegression(graph, causal_model=causal)
        dead_end = DeadEndDetector(brain, default_trials=50, rollout_length=30)
        trap_checks = getattr(brain, "_game_mode", "action") == "puzzle"
        planner = SubgoalPlanner(
            brain, graph, regressor, dead_end, causal,
            enable_trap_checks=trap_checks,
        )
        brain._causal_model = causal
        brain._dead_end_detector = dead_end
        return planner

    _attach_subsystem(
        brain, "planner", _make_planner,
        enabled=brain._enabled["causal_model"],
        error_key="causal_model",
        default=None,
    )
    if brain.planner is None:
        brain._causal_model = None
        brain._dead_end_detector = None
    elif getattr(brain, "_dead_end_detector", None) is None and brain.planner is not None:
        brain._dead_end_detector = brain.planner.dead_end_detector

    brain._active_skill = None
    brain._skill_game_state = {}
    _attach_subsystem(
        brain, "skill_library",
        lambda: __import__(
            "brain.planning.skill_library", fromlist=["SkillLibrary"]
        ).SkillLibrary(),
        enabled=brain._enabled["skill_library"],
    )

    def _make_counterfactual():
        cf_class = _get_counterfactual()
        if cf_class is None:
            raise ImportError("CounterfactualReasoner unavailable")
        return cf_class(brain)

    _attach_subsystem(
        brain, "counterfactual", _make_counterfactual,
        enabled=brain._enabled["counterfactual"],
    )

    _attach_subsystem(
        brain, "_dream_loop",
        lambda: __import__(
            "brain.overnight.dream_loop", fromlist=["DreamLoop"]
        ).DreamLoop(brain, logger=brain.logger),
        enabled=brain._enabled["dreams"],
    )

    _attach_subsystem(
        brain, "attribution",
        lambda: __import__(
            "brain.telemetry.attribution_logger", fromlist=["AttributionLogger"]
        ).AttributionLogger(log_dir=f"logs/telemetry/{session_name}"),
        enabled=brain._enabled["attribution"],
    )

    brain._wm_enabled = brain._enabled["world_model"]
    brain._causal_enabled = brain.planner is not None

    def _make_near_death_replayer():
        from brain.learning.near_death_replayer import NearDeathReplayer
        wm_ref = getattr(brain.basal_ganglia, "_world_model", None)
        replayer = NearDeathReplayer(world_model=wm_ref)
        brain._nd_trigger_interval = 5
        return replayer

    _attach_subsystem(brain, "near_death_replayer", _make_near_death_replayer)

    def _make_knob_tuner():
        from brain.learning.adaptive_knobs import AdaptiveKnobController
        tuner = AdaptiveKnobController(
            window=20, tune_every=3, step_size=0.10, explore_every=10,
        )
        if brain.near_death_replayer is not None:
            tuner.attach(
                brain.near_death_replayer,
                brain.hippocampus.elite,
                brain,
            )
        return tuner

    _attach_subsystem(brain, "knob_tuner", _make_knob_tuner)

    _attach_subsystem(
        brain, "_rehearsal_loop",
        lambda: __import__(
            "brain.rehearsal.rehearsal_loop", fromlist=["RehearsalLoop"]
        ).RehearsalLoop(brain),
        enabled=brain._enabled["rehearsal_loop"],
        error_key="rehearsal_loop",
    )

    brain._env_ref = None
