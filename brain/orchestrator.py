"""
orchestrator.py — Whole Brain Orchestrator.

Instantiates all 7 brain regions, connects them via the MessageBus,
and provides a simple step() API for running the full brain pipeline.

Usage:
    from brain.orchestrator import WholeBrain

    brain = WholeBrain(n_actions=18)
    brain.set_adapter(atari_adapter)

    for step in range(total_steps):
        obs, reward, done, info = env.step(action)
        result = brain.step(obs, action, reward, done)
        action = result["action"]
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from brain.config import VERSION
from brain.message_bus import MessageBus
from brain.telemetry.session_logger import SessionLogger
from brain.regions.sensory_cortex import SensoryCortex
from brain.regions.basal_ganglia import BasalGanglia
from brain.regions.amygdala_thalamus import AmygdalaThalamus
from brain.regions.hippocampus import Hippocampus
from brain.regions.striatum import Striatum
from brain.regions.prefrontal_cortex import PrefrontalCortex
from brain.regions.motor_cortex import MotorCortex
from brain.environments.curiosity import CuriosityModule
from brain.telemetry.step_profiler import StepProfiler

def _get_counterfactual():
    try:
        from brain.planning.counterfactual import CounterfactualReasoner
        return CounterfactualReasoner
    except ImportError:
        return None


# Lazy imports for learner evolution
def _get_meta_controller():
    try:
        from brain.learning.meta_controller import MetaController
        return MetaController
    except ImportError:
        return None

def _get_learner_selector():
    try:
        from brain.learning.learner_selector import LearnerSelector
        from brain.learning.rl_registry import RLRegistry
        return LearnerSelector, RLRegistry
    except ImportError:
        return None, None


class WholeBrain:
    """
    Throng 5 Whole Brain — orchestrates all 7 brain regions.

    Processes one environment step through the full pipeline:
      Sensory Cortex → Basal Ganglia → Amygdala/Thalamus → Striatum → Motor Cortex
      (with Hippocampus and Prefrontal running in parallel on slow path)
    """

    def __init__(
        self,
        n_features: int = 84,
        n_actions: int = 18,
        session_name: str = "throng5",
        enable_logging: bool = True,
        use_torch: bool = False,
        use_cnn: bool = False,
        use_fft: bool = False,
        game_mode: str = "action",  # "action" or "puzzle"
        enabled_systems: Optional[Dict[str, bool]] = None,
    ):
        self.n_features = n_features
        self.n_actions = n_actions
        self._game_mode = game_mode

        # ── Subsystem enable/disable flags (for ablation testing) ─────
        # NOTE: 7 poisonous subsystems purged after ablation testing:
        #   curiosity, meta_controller, rehearsal, dead_end_detector,
        #   surprise_tracker, entropy_monitor, dream_action_bias
        _defaults = {
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
        }
        self._enabled = {**_defaults, **(enabled_systems or {})}
        self._init_errors: Dict[str, str] = {}  # Track init failures

        # ── Message Bus ───────────────────────────────────────────────
        self.bus = MessageBus(history_size=1000)

        # ── Logger ────────────────────────────────────────────────────
        self.logger = SessionLogger(session_name) if enable_logging else None

        # ── Brain Regions ─────────────────────────────────────────────
        self.sensory = SensoryCortex(
            self.bus, n_features=n_features, use_cnn=use_cnn, use_fft=use_fft,
        )
        self.basal_ganglia = BasalGanglia(
            self.bus, n_features=n_features, n_actions=n_actions,
        )
        self.amygdala = AmygdalaThalamus(self.bus, n_features=n_features)
        self.hippocampus = Hippocampus(self.bus)
        self.striatum = Striatum(
            self.bus, n_features=n_features, n_actions=n_actions,
            use_torch=use_torch,
        )
        self.prefrontal = PrefrontalCortex(self.bus)
        self.motor = MotorCortex(self.bus, n_actions=n_actions)

        self._regions = {
            "sensory_cortex": self.sensory,
            "basal_ganglia": self.basal_ganglia,
            "amygdala_thalamus": self.amygdala,
            "hippocampus": self.hippocampus,
            "striatum": self.striatum,
            "prefrontal_cortex": self.prefrontal,
            "motor_cortex": self.motor,
        }

        # ── State ─────────────────────────────────────────────────────
        self._step_count = 0
        self._episode_count = 0
        self._episode_reward = 0.0
        self._prev_features: Optional[np.ndarray] = None
        self._prev_raw_frames: Optional[np.ndarray] = None
        self._last_features: Optional[np.ndarray] = None

        # ── Step Profiler ───────────────────────────────────────────
        self.profiler = StepProfiler(enabled=True)

        # ── Training throttle intervals (reduce per-step cost) ───────
        self._dqn_train_interval = 2    # DQN gradient update every 2nd step
        self._wm_train_interval = 4     # World model train every 4th step
        self._shadow_interval = 8       # Shadow learners every 8th step

        # ── Wire CNN encoder to Striatum for end-to-end learning ──────
        if use_cnn and use_torch and self.sensory._use_cnn:
            cnn_params = self.sensory.get_cnn_parameters()
            if cnn_params:
                self.striatum.wire_cnn_encoder(
                    self.sensory.encode_for_training,
                    cnn_params,
                )

        # ── Wire DQN policy to dreamer so dreams use learned Q-values ──
        if use_torch and self.striatum._torch_dqn is not None:
            self.basal_ganglia.set_policy_fn(
                self.striatum._torch_dqn.select_action
            )

        # [PURGED] Curiosity module — intrinsic reward confused simple envs
        # [PURGED] Meta-Controller — shadow-trained 2 DQNs, wasted compute

        # ── Probe Runner (short empirical algorithm trials) ──────────
        self.probe_runner = None
        if self._enabled["probe_runner"]:
            try:
                from brain.learning.probe_runner import ProbeRunner
                self.probe_runner = ProbeRunner(self, probe_steps=500)
            except Exception as e:
                self._init_errors["probe_runner"] = str(e)

        # ── Stage Classifier (per-area learner specialization) ───────
        self.stage_classifier = None
        if self._enabled["stage_classifier"]:
            try:
                from brain.learning.stage_classifier import StageClassifier
                self.stage_classifier = StageClassifier(n_features=n_features)
            except Exception as e:
                self._init_errors["stage_classifier"] = str(e)

        # ── Plateau detection for LLM re-evaluation ─────────────────
        self._plateau_window = 200        # Episodes to check for plateau
        self._plateau_threshold = 0.02    # <2% improvement = plateau
        self._last_plateau_check = 0

        # [PURGED] Rehearsal Loop — inline rollouts mutated replay buffer

        # ── Planning Layer (long-term reasoning) ────────────────────────
        self.planner = None
        self._causal_model = None
        self._dead_end_detector = None
        if self._enabled["causal_model"]:
            try:
                from brain.planning.landmark_graph import LandmarkGraph
                from brain.planning.dead_end_detector import DeadEndDetector
                from brain.planning.causal_model import CausalModel
                from brain.planning.goal_regression import GoalRegression
                from brain.planning.subgoal_planner import SubgoalPlanner

                graph = LandmarkGraph()
                causal = CausalModel()
                detector = None  # [PURGED] DeadEndDetector — 200-trial forward sim per step
                regressor = GoalRegression(graph, causal_model=causal)
                self.planner = SubgoalPlanner(
                    self, graph, regressor, detector, causal,
                )
                self._causal_model = causal
                self._dead_end_detector = detector
            except Exception as e:
                self._init_errors["causal_model"] = str(e)

        # ── Skill Library (macro-skills for puzzle solving) ─────────────
        self.skill_library = None
        self._active_skill = None    # Currently executing skill
        self._skill_game_state = {}  # Game state for skill preconditions
        if self._enabled["skill_library"]:
            try:
                from brain.planning.skill_library import SkillLibrary
                self.skill_library = SkillLibrary()
            except Exception as e:
                self._init_errors["skill_library"] = str(e)

        # ── Counterfactual Reasoner (regret analysis on death) ─────────
        self.counterfactual = None
        if self._enabled["counterfactual"]:
            try:
                CFClass = _get_counterfactual()
                if CFClass is not None:
                    self.counterfactual = CFClass(self)
            except Exception as e:
                self._init_errors["counterfactual"] = str(e)

        # ── Overnight Dream Loop ──────────────────────────────────────
        self._dream_loop = None
        if self._enabled["dreams"]:
            try:
                from brain.overnight.dream_loop import DreamLoop
                self._dream_loop = DreamLoop(self, logger=self.logger)
            except Exception as e:
                self._init_errors["dreams"] = str(e)

        # [PURGED] Surprise Tracker — positive feedback loop with bad WM data

        # ── Decision Attribution ──────────────────────────────────────
        self.attribution = None
        if self._enabled["attribution"]:
            try:
                from brain.telemetry.attribution_logger import AttributionLogger
                log_dir = f"logs/telemetry/{session_name}"
                self.attribution = AttributionLogger(log_dir=log_dir)
            except Exception as e:
                self._init_errors["attribution"] = str(e)

        # [PURGED] Entropy Monitor — overrode epsilon AND injected WM noise

        # ── Throttle intervals for remaining systems ───────────────────
        self._causal_observe_interval = 2     # Causal model every 2nd step

        # ── Inline rehearsal state ────────────────────────────────────
        self._env_ref = None   # Set via set_env() for inline rehearsal

        if self.logger:
            self.logger.milestone("init", f"WholeBrain v{VERSION} initialized with {len(self._regions)} regions, mode={self._game_mode}")

    def set_adapter(self, adapter) -> None:
        """Set the environment adapter."""
        self.sensory.set_adapter(adapter)

    def set_env(self, env) -> None:
        """Set env reference for inline rehearsal (save/load support)."""
        self._env_ref = env

    def set_game_mode(self, mode: str) -> None:
        """Set game mode: 'action' (default) or 'puzzle' (enables dead-end checks)."""
        self._game_mode = mode
        if self.logger:
            self.logger.event("config", "game_mode", f"Mode set to {mode}")

    def set_game_state(self, game_state: dict) -> None:
        """Update game state dict consumed by skill library preconditions."""
        self._skill_game_state = game_state

    def activate_skill(
        self, skill_name: str, **params,
    ) -> bool:
        """
        Activate a macro-skill from the skill library.

        While active, the skill overrides the DQN action each step.
        Returns True if skill was activated, False if not available.
        """
        if self.skill_library is None:
            return False
        skill = self.skill_library.create(skill_name, **params)
        if skill is None:
            return False
        skill.start(**params)
        self._active_skill = skill
        if self.logger:
            self.logger.event("skill", "activate", f"{skill_name}: {params}")
        return True

    def step(
        self,
        obs: Any,
        prev_action: int = 0,
        reward: float = 0.0,
        done: bool = False,
    ) -> Dict[str, Any]:
        """
        Run one timestep through the whole brain.

        Returns dict with at least 'action' key.
        """
        self._step_count += 1
        self._episode_reward += reward
        self.profiler.step_start()

        # ── 1. Sensory Cortex ─────────────────────────────────────────
        self.profiler.start("sensory")
        perception = self.sensory.step({
            "obs": obs,
            "action": prev_action,
            "reward": reward,
            "done": done,
        })
        features = perception.get("features")
        if features is None and obs is not None:
            features = np.asarray(obs, dtype=np.float32).flatten()[:self.n_features]
            if len(features) < self.n_features:
                features = np.pad(features, (0, self.n_features - len(features)))
        raw_frames = self.sensory.get_last_preprocessed() if self.sensory._use_cnn else None
        self.profiler.stop("sensory")

        # ── 2. Basal Ganglia ──────────────────────────────────────────
        self.profiler.start("basal_ganglia")
        bg_output = self.basal_ganglia.step({
            "features": features,
            "reward": reward,
            "step": self._step_count,
        })
        self.profiler.stop("basal_ganglia")

        # ── 3. Amygdala/Thalamus ──────────────────────────────────────
        self.profiler.start("amygdala")
        threat_output = self.amygdala.step({
            "features": features,
            "dream_results": bg_output.get("dream_results"),
            "surprise_level": 0.0,
            "step": self._step_count,
        })
        self.profiler.stop("amygdala")

        # ── 4. Hippocampus ────────────────────────────────────────────
        if self._enabled["hippocampus_store"]:
            self.profiler.start("hippocampus")
            if self._prev_features is not None:
                self.hippocampus.step({
                    "state": self._prev_features,
                    "action": prev_action,
                    "reward": reward,
                    "next_state": features,
                    "done": done,
                    "td_error": 0.0,
                })
            self.profiler.stop("hippocampus")

        # ── 5. Striatum — action selection ────────────────────────────
        # [PURGED] dream_action_bias injection — uncalibrated dream values biased DQN

        self.profiler.start("striatum_select")
        striatum_output = self.striatum.step({"features": features})
        striatum_action = striatum_output.get("action", 0)
        self.profiler.stop("striatum_select")

        # ── 6. Learning ──────────────────────────────────────────────
        if self._prev_features is not None:
            # [PURGED] Curiosity intrinsic reward — confused simple envs

            self.profiler.start("striatum_learn")
            self.striatum.learn({
                "state": self._prev_features,
                "action": prev_action,
                "reward": reward,  # Raw reward, no augmentation
                "next_state": features,
                "done": done,
                "raw_frames": self._prev_raw_frames,
                "next_raw_frames": raw_frames,
                "skip_train": (self._step_count % self._dqn_train_interval != 0),
            })
            self.profiler.stop("striatum_learn")

            self.profiler.start("world_model")
            if self._enabled["world_model"]:
                self.basal_ganglia.learn({
                    "state": self._prev_features,
                    "action": prev_action,
                    "next_state": features,
                    "reward": reward,
                    "skip_train": (self._step_count % self._wm_train_interval != 0),
                })
            self.profiler.stop("world_model")

            # [PURGED] Meta-Controller shadow training — wasted compute

        # ── 6.5 Causal model + subgoal tracking ─────────────────────
        if (self.planner is not None
                and self._prev_features is not None
                and self._step_count % self._causal_observe_interval == 0):
            self.profiler.start("causal_observe")
            try:
                self.planner.observe_transition(
                    self._prev_features, prev_action, features, reward, done,
                )
            except Exception:
                pass
            self.profiler.stop("causal_observe")

        # [PURGED] Dead-end check — 200-trial forward simulation per step
        # [PURGED] Rehearsal trigger — inline rollouts mutated replay buffer

        # Track features and raw frames for next step
        self._prev_features = features
        self._prev_raw_frames = raw_frames
        self._last_features = features

        # [PURGED] Surprise tracking — positive feedback loop with bad WM data

        # ── 7. Skill Library Override ─────────────────────────────────
        skill_override = None
        if self._active_skill is not None and features is not None:
            try:
                skill_result = self._active_skill.step(
                    features, self._skill_game_state, reward,
                )
                if skill_result["status"] == "active":
                    skill_override = skill_result["action"]
                elif skill_result["status"] in ("complete", "timeout", "failed"):
                    self._active_skill = None  # Skill finished
            except Exception:
                self._active_skill = None

        # ── 8. Motor Cortex ───────────────────────────────────────────
        self.profiler.start("motor")
        motor_output = self.motor.step({
            "striatum_action": skill_override if skill_override is not None else striatum_action,
            "features": features,
            "striatum_halted": self.bus.is_halted("striatum"),
        })
        action = motor_output.get("action", 0)
        self.profiler.stop("motor")

        # [PURGED] Entropy monitoring — epsilon override + WM noise injection
        epsilon_used = striatum_output.get("epsilon", 0.15)

        self.profiler.step_end()

        # ── 9. Decision trace ────────────────────────────────────────
        if self.attribution is not None:
            try:
                from brain.telemetry.decision_trace import DecisionTrace
                trace = DecisionTrace(
                    step=self._step_count,
                    action_taken=action,
                    action_source=motor_output.get("source", "unknown"),
                    striatum_action=striatum_action or 0,
                    striatum_q_values=(
                        striatum_output.get("q_values", [])
                    ),
                    threat_score=threat_output.get("threat_score", 0.0),
                    curiosity_bonus=0.0,
                    surprise=0.0,
                    reward=reward,
                    episode_reward_so_far=self._episode_reward,
                    epsilon=epsilon_used,
                    dead_end_detected=False,
                    entropy_override=False,
                    region_times=self.profiler.report(),
                )
                self.attribution.record(trace)
            except Exception:
                pass

        # ── Episode boundary ──────────────────────────────────────────
        if done:
            self._on_episode_done()

        return {
            "action": action,
            "threat_score": threat_output.get("threat_score", 0.0),
            "operating_mode": threat_output.get("operating_mode", "execute"),
            "epsilon": epsilon_used,
            "context_score": bg_output.get("context_score", 0.0),
            "action_source": motor_output.get("source", "unknown"),
            "intrinsic_reward": intrinsic_reward,
            "surprise": surprise_val,
        }

    def _on_episode_done(self) -> None:
        """Handle episode completion."""
        self._episode_count += 1

        # ── Report to MetaController for learner evolution ────────────
        if self.meta_controller is not None:
            self.meta_controller.report_reward(
                self._active_learner_name,
                self._episode_reward,
            )

        # ── Feed stage classifier with per-episode learner performance ──
        if self.stage_classifier is not None and self._last_features is not None:
            stage_id = self.stage_classifier.classify(self._last_features)
            self.stage_classifier.record(
                stage_id, self._active_learner_name, self._episode_reward,
            )

        # ── Feed death events to rehearsal bottleneck tracker ─────────
        if self.rehearsal is not None and self._last_features is not None:
            if self._episode_reward < 0:
                self.rehearsal.tracker.record_death(
                    self._last_features,
                    context={
                        "episode": self._episode_count,
                        "episode_reward": self._episode_reward,
                        "steps": self._step_count,
                    },
                )
            else:
                self.rehearsal.tracker.record_success(self._last_features)

        # ── Counterfactual regret analysis on death ───────────────────
        if (self.counterfactual is not None
                and self._episode_reward < 0
                and self._last_features is not None
                and self._prev_features is not None):
            try:
                regret = self.counterfactual.find_regret(
                    self._prev_features,
                    actual_action=0,  # Last action before death
                    actual_reward=self._episode_reward,
                    n_alternatives=min(self.n_actions, 6),
                    n_steps=30,
                )
                if regret.get("regret", 0) > 0.5 and self._causal_model is not None:
                    # Feed regret back to causal model as dangerous action
                    self._causal_model.observe(
                        self._prev_features,
                        regret["actual_action"],
                        self._last_features,
                        reward=self._episode_reward,
                        is_dead_end=True,
                    )
            except Exception:
                pass

        # ── Export proven chains to Motor Cortex heuristics ───────────
        if self.rehearsal is not None:
            try:
                heuristics = self.rehearsal.chain_store.export_heuristics()
                if heuristics:
                    self.motor.install_heuristics(heuristics)
            except Exception:
                pass

        # ── Attribution episode summary ───────────────────────────────
        if self.attribution is not None:
            try:
                self.attribution.episode_summary(
                    self._episode_count, self._episode_reward,
                )
            except Exception:
                pass

        if self.logger:
            self.logger.training_step(
                "whole_brain", self._episode_count, self._step_count,
                {"episode_reward": self._episode_reward}
            )

        # Reset all regions
        for region in self._regions.values():
            region.reset_episode()

        self.bus.resume_all()
        self._episode_reward = 0.0

    # ── Probe & Plateau API ──────────────────────────────────────────

    def run_probe(self, obs_fn=None, reward_fn=None):
        """
        Run empirical probe with top learners. Returns ProbeResult.

        If obs_fn is None, uses random observations.
        """
        if self.probe_runner is None:
            return None

        if obs_fn is None:
            n_feat = self.sensory._n_features
            obs_fn = lambda: np.random.randn(n_feat).astype(np.float32)

        return self.probe_runner.run_probe(obs_fn, reward_fn)

    def request_plateau_review(self):
        """
        Request LLM re-evaluation of algorithm selection on plateau.
        Returns review result from Prefrontal Cortex.
        """
        if self.meta_controller is None:
            return None
        meta_report = self.meta_controller.report()
        return self.prefrontal.request_algorithm_review(
            meta_report,
            plateau_info={
                "episode_count": self._episode_count,
                "step_count": self._step_count,
            },
        )

    def rehearse(self, mode: str = "advance", env=None, features=None, **kwargs):
        """
        Run rehearsal in the specified mode.

        Modes:
            advance:  Pause → 3-tier validate → execute → repeat
            frontier: Play from start; Advance on death
            stuck:    10 failures → train flanking areas
            free:     Play normally, log stuck points for LLM

        Args:
            mode: One of 'advance', 'frontier', 'stuck', 'free'
            env: Environment (required for frontier/stuck/free modes)
            features: State features (required for advance/stuck modes)
        """
        if self.rehearsal is None:
            return {"status": "not_available"}

        if mode == "advance":
            if features is None and self._last_features is not None:
                features = self._last_features
            if features is None:
                return {"status": "no_features"}
            return self.rehearsal.run_advance(features, env, **kwargs)

        elif mode == "frontier":
            if env is None:
                return {"status": "no_env"}
            return self.rehearsal.run_frontier(env, **kwargs)

        elif mode == "stuck":
            if features is None and self._last_features is not None:
                features = self._last_features
            if features is None:
                return {"status": "no_features"}
            return self.rehearsal.run_stuck(features, env, **kwargs)

        elif mode == "free":
            if env is None:
                return {"status": "no_env"}
            return self.rehearsal.run_free(env, **kwargs)

        return {"status": "unknown_mode", "mode": mode}

    def plan(self, goal_features=None, goal_hash=None, goal_label="goal"):
        """
        Create or query a long-term plan via the SubgoalPlanner.

        Args:
            goal_features: Feature vector of the goal state
            goal_hash: Hash of goal landmark (alternative to features)
            goal_label: Human-readable label for the goal

        Returns:
            Plan dict with subgoals, or current plan status if already active.
        """
        if self.planner is None:
            return {"status": "not_available"}

        if goal_features is not None:
            current = self._last_features
            if current is None:
                return {"status": "no_current_state"}
            return self.planner.make_plan(current, goal_features, goal_label)

        if self.planner.has_plan:
            return self.planner.report()

        return {"status": "no_goal_specified"}

    def dream(self, n_replay: int = 50, n_dream_steps: int = 20,
             max_time: float = 3600.0) -> Dict[str, Any]:
        """
        Run overnight consolidation: replay + dream + heuristic extraction.

        Call between sessions (not during live play).
        """
        if self._dream_loop is None:
            return {"status": "not_available"}
        result = self._dream_loop.run(
            n_replay_cycles=n_replay,
            n_dream_steps=n_dream_steps,
            max_time_seconds=max_time,
        )
        # Install any heuristics extracted during dreaming
        if self.rehearsal is not None:
            try:
                heuristics = self.rehearsal.chain_store.export_heuristics()
                if heuristics:
                    self.motor.install_heuristics(heuristics)
            except Exception:
                pass
        return result

    def report(self) -> Dict[str, Dict]:
        """Get reports from all brain regions."""
        r = {name: region.report() for name, region in self._regions.items()}
        if self.stage_classifier is not None:
            r["stage_classifier"] = self.stage_classifier.report()
        if self.rehearsal is not None:
            r["rehearsal"] = self.rehearsal.report()
        if self.planner is not None:
            r["planning"] = self.planner.report()
        if self.counterfactual is not None:
            r["counterfactual"] = self.counterfactual.report()
        if self._causal_model is not None:
            r["causal_model"] = self._causal_model.report()
        if self.surprise_tracker is not None:
            r["surprise_tracker"] = self.surprise_tracker.report()
        if self.attribution is not None:
            r["attribution"] = self.attribution.report()
        if self.entropy_monitor is not None:
            r["entropy_monitor"] = self.entropy_monitor.report()
        return r

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Return full diagnostic information for ablation reporting."""
        return {
            "enabled_systems": dict(self._enabled),
            "init_errors": dict(self._init_errors),
            "active_subsystems": {
                "curiosity": self.curiosity is not None,
                "meta_controller": self.meta_controller is not None,
                "probe_runner": self.probe_runner is not None,
                "stage_classifier": self.stage_classifier is not None,
                "rehearsal": self.rehearsal is not None,
                "planner": self.planner is not None,
                "causal_model": self._causal_model is not None,
                "dead_end_detector": self._dead_end_detector is not None,
                "skill_library": self.skill_library is not None,
                "counterfactual": self.counterfactual is not None,
                "dream_loop": self._dream_loop is not None,
                "surprise_tracker": self.surprise_tracker is not None,
                "attribution": self.attribution is not None,
                "entropy_monitor": self.entropy_monitor is not None,
            },
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "profiler_report": self.profiler.report(),
        }

    def close(self) -> None:
        if self.logger:
            self.logger.milestone("shutdown", f"WholeBrain shutdown after {self._step_count} steps, {self._episode_count} episodes")
            self.logger.close()

    def __repr__(self) -> str:
        return f"WholeBrain(v{VERSION}, regions={len(self._regions)}, steps={self._step_count}, episodes={self._episode_count}, mode={self._game_mode})"
