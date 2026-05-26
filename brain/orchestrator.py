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

import logging
from typing import Any, Dict, Optional

import numpy as np

from brain.config import VERSION
from brain.telemetry.decision_trace import DecisionTrace
from brain.message_bus import FastSlotBus, MessageBus
from brain.telemetry.session_logger import SessionLogger
from brain.regions.sensory_cortex import SensoryCortex
from brain.regions.basal_ganglia import BasalGanglia
from brain.regions.amygdala_thalamus import AmygdalaThalamus
from brain.regions.hippocampus import Hippocampus
from brain.regions.striatum import Striatum
from brain.regions.prefrontal_cortex import PrefrontalCortex
from brain.regions.motor_cortex import MotorCortex
from brain.environments.curiosity import CuriosityModule  # noqa: F401 — referenced by gauntlet
from brain.telemetry.step_profiler import StepProfiler
from brain.orchestrator_wiring import wire_subsystems

_log = logging.getLogger(__name__)


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
        self._enabled: Dict[str, bool] = {}
        self._init_errors: Dict[str, str] = {}

        # ── Message Bus ───────────────────────────────────────────────
        # history disabled by default: was appending every BrainMessage every step
        self.bus = MessageBus(history_size=1000, enable_history=False)

        # FastSlotBus: zero-allocation signalling for hot-path inter-region data
        self.fast_bus = FastSlotBus()
        self.fast_bus.register("amygdala", {
            "threat_score": 0.0, "operating_mode": "execute", "epsilon": 0.15,
        })
        self.fast_bus.register("basal_ganglia", {
            "context_score": 0.0, "dream_results": None,
        })
        self.fast_bus.register("motor", {
            "action": 0, "source": "striatum",
        })

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

        # ── Wire Elite Replay: hippocampus tracks best episodes, striatum uses them ─
        self.striatum.set_elite_buffer(self.hippocampus.elite)
        # ── Wire HER: hippocampus injects relabelled transitions into striatum._nstep ─
        self.hippocampus.set_striatum_ref(self.striatum)
        self._last_action = 0

        # ── State ─────────────────────────────────────────────────────
        self._step_count = 0
        self._episode_count = 0
        self._episode_reward = 0.0
        self._prev_features: Optional[np.ndarray] = None
        self._prev_raw_frames: Optional[np.ndarray] = None
        self._last_features: Optional[np.ndarray] = None

        # ── Step Profiler ───────────────────────────────────────────
        self.profiler = StepProfiler(enabled=True)

        # ── Training throttle intervals (replace dict-lookup enable checks) ──
        # Using plain int counters: cheaper than self._enabled["x"] dict lookup
        self._dqn_train_interval = 2    # DQN gradient update every 2nd step
        self._wm_train_interval = 4     # World model train every 4th step
        self._causal_observe_interval = 2  # Causal model observe every 2nd step

        self._plateau_window = 200
        self._plateau_threshold = 0.02
        self._last_plateau_check = 0

        wire_subsystems(
            self,
            session_name=session_name,
            n_features=n_features,
            n_actions=n_actions,
            use_cnn=use_cnn,
            use_torch=use_torch,
            enabled_systems=enabled_systems,
        )

        # ── Pre-allocated step return dict (updated in-place each step) ─
        self._step_result: Dict[str, Any] = {
            "action": 0, "threat_score": 0.0, "operating_mode": "execute",
            "epsilon": 0.15, "context_score": 0.0, "action_source": "striatum",
        }

        if self.logger:
            self.logger.milestone("init", f"WholeBrain v{VERSION} initialized with {len(self._regions)} regions, mode={self._game_mode}")

    def _wire_striatum_learning(self) -> None:
        """Connect SR, Option-Critic context, and hippocampus → striatum."""
        if self.hippocampus._sr is not None:
            try:
                self.striatum.set_sr_module(self.hippocampus._sr)
            except Exception as e:
                self._init_errors["striatum_sr"] = str(e)

        if self.striatum.option_critic is not None:
            try:
                self.striatum.set_context_sources(
                    elite_buf=self.hippocampus.elite,
                    dreamer=self.dreamer,
                )
            except Exception as e:
                self._init_errors["striatum_oc_context"] = str(e)

    def set_adapter(self, adapter) -> None:
        """Set the environment adapter."""
        self.sensory.set_adapter(adapter)

    def set_env(self, env) -> None:
        """Set env reference for inline rehearsal (save/load support)."""
        self._env_ref = env

    def set_game_mode(self, mode: str) -> None:
        """Set game mode: 'action' (default) or 'puzzle' (enables trap rollouts)."""
        self._game_mode = mode
        if self.planner is not None:
            self.planner.enable_trap_checks = (mode == "puzzle")
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

    def _step_sensory(
        self,
        obs: Any,
        prev_action: int,
        reward: float,
        done: bool,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Run sensory phase and return features + optional raw frames."""
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
        return features, raw_frames

    def _step_regions(self, features: Optional[np.ndarray], reward: float) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run basal ganglia and amygdala phases."""
        self.profiler.start("basal_ganglia")
        bg_output = self.basal_ganglia.step({
            "features": features,
            "reward": reward,
            "step": self._step_count,
        })
        self.profiler.stop("basal_ganglia")

        self.profiler.start("amygdala")
        threat_output = self.amygdala.step({
            "features": features,
            "dream_results": bg_output.get("dream_results"),
            "surprise_level": 0.0,
            "step": self._step_count,
        })
        self.profiler.stop("amygdala")
        return bg_output, threat_output

    def _step_dreamer(self, prev_action: int, features: Optional[np.ndarray], reward: float) -> None:
        """Store/train Dreamer model on schedule."""
        if self.dreamer is None:
            return
        try:
            self.dreamer.store_transition(
                self._prev_features, prev_action, features, reward,
            )
            if self._step_count % 16 == 0:
                self.dreamer.train_step()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _log.debug("dreamer step failed: %s", exc)

    def _step_planner_observe(
        self,
        prev_action: int,
        features: Optional[np.ndarray],
        reward: float,
        done: bool,
    ) -> None:
        """Feed transition into planner causal observer."""
        if (
            self.planner is None
            or self._prev_features is None
            or self._step_count % self._causal_observe_interval != 0
        ):
            return
        self.profiler.start("causal_observe")
        try:
            self.planner.observe_transition(
                self._prev_features, prev_action, features, reward, done,
            )
        except AttributeError as exc:
            _log.warning("causal observe missing dependency: %s", exc)
        except (RuntimeError, TypeError, ValueError) as exc:
            _log.debug("causal observe failed: %s", exc)
        self.profiler.stop("causal_observe")

    def _step_skill(self, features: Optional[np.ndarray], reward: float) -> Optional[int]:
        """Run active skill override (if any)."""
        if self._active_skill is None or features is None:
            return None
        try:
            skill_result = self._active_skill.step(
                features, self._skill_game_state, reward,
            )
            status = skill_result.get("status")
            if status == "active":
                return skill_result.get("action")
            if status in ("complete", "timeout", "failed"):
                self._active_skill = None
                return None
            return None
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            _log.debug("skill step failed; disabling skill: %s", exc)
            self._active_skill = None
            return None

    def _step_motor(
        self,
        features: Optional[np.ndarray],
        striatum_action: int,
        skill_override: Optional[int],
    ) -> Dict[str, Any]:
        """Run motor cortex arbitration/action."""
        self.profiler.start("motor")
        motor_output = self.motor.step({
            "striatum_action": skill_override if skill_override is not None else striatum_action,
            "features": features,
            "striatum_halted": self.bus.is_halted("striatum"),
        })
        self._last_action = motor_output.get("action", 0)
        self.profiler.stop("motor")
        return motor_output

    def _step_attribution(
        self,
        action: int,
        striatum_action: int,
        striatum_output: Dict[str, Any],
        threat_output: Dict[str, Any],
        reward: float,
        epsilon_used: float,
        motor_output: Dict[str, Any],
    ) -> None:
        """Record per-step decision trace."""
        if self.attribution is None:
            return
        try:
            trace = DecisionTrace(
                step=self._step_count,
                action_taken=action,
                action_source=motor_output.get("source", "unknown"),
                striatum_action=striatum_action or 0,
                striatum_q_values=striatum_output.get("q_values", []),
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
        except (AttributeError, TypeError, ValueError) as exc:
            _log.debug("attribution record failed: %s", exc)

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
        self._last_action = prev_action
        self.profiler.step_start()

        features, raw_frames = self._step_sensory(obs, prev_action, reward, done)
        bg_output, threat_output = self._step_regions(features, reward)

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
        self.profiler.start("striatum_select")
        striatum_output = self.striatum.step({"features": features})
        striatum_action = striatum_output.get("action", 0)
        self.profiler.stop("striatum_select")

        # ── 6. Learning ──────────────────────────────────────────────
        if self._prev_features is not None:
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

            self._step_dreamer(prev_action, features, reward)

        self._step_planner_observe(prev_action, features, reward, done)

        # Track features and raw frames for next step
        self._prev_features = features
        self._prev_raw_frames = raw_frames
        self._last_features = features

        # ── 7. Skill Library Override ─────────────────────────────────
        skill_override = self._step_skill(features, reward)
        motor_output = self._step_motor(features, striatum_action, skill_override)
        action = motor_output.get("action", 0)

        epsilon_used = striatum_output.get("epsilon", 0.15)

        self.profiler.step_end()

        self._step_attribution(
            action=action,
            striatum_action=striatum_action,
            striatum_output=striatum_output,
            threat_output=threat_output,
            reward=reward,
            epsilon_used=epsilon_used,
            motor_output=motor_output,
        )

        # ── Episode boundary ──────────────────────────────────────────
        if done:
            self._on_episode_done()

        # Update pre-allocated return dict in-place (no new dict allocation)
        self._step_result["action"] = action
        self._step_result["threat_score"] = threat_output.get("threat_score", 0.0)
        self._step_result["operating_mode"] = threat_output.get("operating_mode", "execute")
        self._step_result["epsilon"] = epsilon_used
        self._step_result["context_score"] = bg_output.get("context_score", 0.0)
        self._step_result["action_source"] = motor_output.get("source", "unknown")
        return self._step_result

    def _on_episode_done(self) -> None:
        """Handle episode completion."""
        self._episode_count += 1

        # ── Feed stage classifier with per-episode learner performance ──
        if self.stage_classifier is not None and self._last_features is not None:
            stage_id = self.stage_classifier.classify(self._last_features)
            self.stage_classifier.record(
                stage_id, "default", self._episode_reward,
            )

        # ── Counterfactual regret analysis on death ───────────────────
        if (self.counterfactual is not None
                and self._episode_reward < 0
                and self._last_features is not None
                and self._prev_features is not None):
            try:
                regret = self.counterfactual.find_regret(
                    self._prev_features,
                    actual_action=self._last_action,
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
            except Exception as e:
                _log.debug("counterfactual regret failed: %s", e)

        # ── Near-Death Counterfactual Replay (every N episodes) ────────
        if (self.near_death_replayer is not None
                and self._episode_count % self._nd_trigger_interval == 0
                and not self.hippocampus.elite.is_empty):
            try:
                injected = self.near_death_replayer.build_replay_batch_direct(
                    elite_buf=self.hippocampus.elite,
                    striatum=self.striatum,
                )
                if self.knob_tuner is not None:
                    self.knob_tuner.record_trigger()
                if self.logger and injected > 0:
                    self.logger.event(
                        "near_death_replay", "trigger",
                        f"Injected {injected} counterfactual transitions "
                        f"(ep {self._episode_count})",
                    )
            except Exception as e:
                _log.debug("near_death replay failed: %s", e)

        # ── Attribution episode summary ───────────────────────────────
        if self.attribution is not None:
            try:
                self.attribution.episode_summary(
                    self._episode_count, self._episode_reward,
                )
            except Exception as e:
                _log.debug("attribution episode summary failed: %s", e)

        if self.logger:
            self.logger.training_step(
                "whole_brain", self._episode_count, self._step_count,
                {"episode_reward": self._episode_reward}
            )

        # Reset all regions
        for region in self._regions.values():
            region.reset_episode()

        self.bus.resume_all()
        # Knob tuner: record episode reward before resetting
        if self.knob_tuner is not None:
            try:
                self.knob_tuner.record_episode(self._episode_reward)
            except Exception as e:
                _log.debug("knob tuner record failed: %s", e)
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
        Placeholder for LLM re-evaluation of algorithm selection on plateau.
        MetaController was purged — this returns None until a replacement is wired.
        """
        return None

    def rehearse(self, **kwargs):
        """
        Run RehearsalLoop when wired via orchestrator_wiring.

        Without env/features, returns not_available (smoke-safe).
        Montezuma mode_rehearse passes mode, env, and features.
        """
        loop = getattr(self, "_rehearsal_loop", None)
        if loop is None:
            return {"status": "not_available"}

        mode = kwargs.get("mode", "advance")
        env = kwargs.get("env")
        features = kwargs.get("features")

        try:
            if mode == "advance":
                if env is None or features is None:
                    return {
                        "status": "not_available",
                        "reason": "advance requires env and features",
                    }
                result = loop.run_advance(features, env)
            elif mode == "frontier":
                if env is None:
                    return {"status": "not_available", "reason": "frontier requires env"}
                result = loop.run_frontier(env)
            elif mode == "stuck":
                if env is None or features is None:
                    return {
                        "status": "not_available",
                        "reason": "stuck requires env and features",
                    }
                result = loop.run_stuck(features, env)
            elif mode in ("free", "free_run"):
                if env is None:
                    return {"status": "not_available", "reason": "free requires env"}
                result = loop.run_free(
                    env, max_episodes=kwargs.get("max_episodes", 100),
                )
            else:
                return {"status": "not_available", "reason": f"unknown mode: {mode}"}

            return {"status": "ok", "mode": mode, **result}
        except Exception as exc:
            _log.warning("rehearse(%s) failed: %s", mode, exc)
            return {"status": "error", "mode": mode, "message": str(exc)}

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
        # Heuristics from dream loop installed directly into motor cortex
        return result

    def report(self) -> Dict[str, Dict]:
        """Get reports from all brain regions."""
        r = {name: region.report() for name, region in self._regions.items()}
        if self.stage_classifier is not None:
            r["stage_classifier"] = self.stage_classifier.report()
        if self.planner is not None:
            r["planning"] = self.planner.report()
        if self.counterfactual is not None:
            r["counterfactual"] = self.counterfactual.report()
        if self._causal_model is not None:
            r["causal_model"] = self._causal_model.report()
        if self.attribution is not None:
            r["attribution"] = self.attribution.report()
        return r

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Return full diagnostic information for ablation reporting."""
        return {
            "enabled_systems": dict(self._enabled),
            "init_errors": dict(self._init_errors),
            "active_subsystems": {
                "probe_runner": self.probe_runner is not None,
                "stage_classifier": self.stage_classifier is not None,
                "planner": self.planner is not None,
                "causal_model": self._causal_model is not None,
                "dead_end_detector": False,  # purged from hot path
                "skill_library": self.skill_library is not None,
                "counterfactual": self.counterfactual is not None,
                "dream_loop": self._dream_loop is not None,
                "attribution": self.attribution is not None,
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
