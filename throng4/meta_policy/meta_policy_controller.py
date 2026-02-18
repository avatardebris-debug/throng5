"""
Meta-Policy Controller — Top-level blind orchestrator (Central Controller).

Bridge 1 refactor: delegates to brain-region sub-modules:
  - PerceptionHub:     visual/causal pattern extraction
  - RiskSensor:        plateau detection, risk assessment
  - PolicyMonitor:     policy lifecycle (promotion, retirement, mode)
  - PrefrontalCortex:  LLM/Tetra interaction

Public API is UNCHANGED — all callers work identically.

CRITICAL CONSTRAINT: No game names, no external knowledge.
The LLM receives ONLY numerical patterns from the agent's own data.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque

from throng4.meta_policy.env_fingerprint import (
    EnvironmentFingerprint,
    fingerprint_environment,
)
from throng4.meta_policy.policy_tree import PolicyTree, PolicyNode
from throng4.meta_policy.blind_concepts import BlindConceptLibrary, DiscoveredConcept
from throng4.meta_policy.perception_hub import PerceptionHub
from throng4.meta_policy.risk_sensor import RiskSensor
from throng4.meta_policy.policy_monitor import PolicyMonitor
from throng4.meta_policy.prefrontal_cortex import PrefrontalCortex
from throng4.meta_policy.save_state_manager import SaveStateManager
from throng4.metastack_pipeline import MetaStackPipeline
from throng4.basal_ganglia.dreamer_engine import DreamerEngine, Hypothesis
from throng4.basal_ganglia.amygdala import Amygdala
from throng4.basal_ganglia.hypothesis_profiler import DreamerTeacher


@dataclass
class ControllerConfig:
    """Configuration for MetaPolicyController."""
    fingerprint_episodes: int = 20
    plateau_window: int = 15
    plateau_threshold: float = 0.05
    promote_after_episodes: int = 50
    concept_discovery_interval: int = 50
    max_concurrent_policies: int = 10
    llm_cooldown: int = 30


class MetaPolicyController:
    """
    Top-level blind orchestrator (Central Controller).
    
    Decides WHAT to learn, not HOW to learn.
    Operates without game names or external knowledge.
    
    Lifecycle:
        1. on_new_environment(env) → fingerprint, match/create policy, return pipeline
        2. on_step(state, action, reward, next_state) → record for perception
        3. on_episode_complete(reward) → update performance, detect plateau
        4. on_environment_done() → save weights, final concept discovery
    
    Delegates to brain-region sub-modules:
        - perception:      PerceptionHub (visual/causal extraction)
        - risk_sensor:     RiskSensor (plateau detection)
        - policy_monitor:  PolicyMonitor (policy lifecycle)
        - prefrontal:      PrefrontalCortex (LLM/Tetra dialogue)
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None, 
                 llm_client=None):
        self.config = config or ControllerConfig()
        
        # Brain-region sub-modules
        self.perception = PerceptionHub()
        self.risk_sensor = RiskSensor(
            plateau_window=self.config.plateau_window,
            plateau_threshold=self.config.plateau_threshold,
        )
        self.policy_monitor = PolicyMonitor(
            promote_after_episodes=self.config.promote_after_episodes,
            concept_discovery_interval=self.config.concept_discovery_interval,
        )
        self.prefrontal = PrefrontalCortex(llm_client=llm_client)
        self.save_state_manager = SaveStateManager()
        
        # Bridge Step 4: Basal Ganglia (dreamer) + Amygdala (danger detection)
        self.basal_ganglia = DreamerEngine(
            n_hypotheses=3,
            network_size='micro',
            state_size=64,
            n_actions=4,  # Updated on_new_environment
            dream_interval=5,
        )
        self.amygdala = Amygdala()
        
        # Bridge Step 4, Phase 7: Dreamer-as-Teacher pipeline
        self.dreamer_teacher = DreamerTeacher(
            n_actions=4,  # Updated on_new_environment
            state_dim=16,
        )
        self._last_observed_state: Optional[np.ndarray] = None
        
        # Core components
        self.policy_tree = PolicyTree()
        self.concept_library = BlindConceptLibrary()
        
        # Current state
        self.current_policy: Optional[PolicyNode] = None
        self.current_fingerprint: Optional[EnvironmentFingerprint] = None
        self.current_pipeline: Optional[MetaStackPipeline] = None
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=200)
        self.episode_history: List[Dict] = []
        
        # Stats
        self.environments_seen = 0
        self.total_concepts_discovered = 0
        self.total_policy_branches = 0
        self.total_policy_creates = 0
    
    # ── Backward-compatible properties ──────────────────────────
    # These expose sub-module state for callers that access them directly.
    
    @property
    def llm_client(self):
        return self.prefrontal.llm_client
    
    @property
    def current_visual_patterns(self):
        return self.perception.visual_patterns
    
    @property
    def current_causal_effects(self):
        return self.perception.causal_effects
    
    @property
    def active_hypothesis(self):
        return self.prefrontal.active_hypothesis
    
    @property
    def hypothesis_history(self):
        return self.prefrontal.hypothesis_history
    
    @property
    def total_hypotheses_tested(self):
        return self.prefrontal.total_hypotheses_tested
    
    @property
    def hypothesis_test_episodes(self):
        return self.prefrontal.hypothesis_test_episodes
    
    # ── Environment lifecycle ──────────────────────────────────
    
    def on_new_environment(self, env) -> MetaStackPipeline:
        """
        Called when entering a new environment.
        
        1. Run random episodes to build fingerprint
        2. Search PolicyTree for similar fingerprint
        3. Branch from match OR create new root policy
        4. Return initialized pipeline
        """
        self.environments_seen += 1
        self.episode_count = 0
        self.episode_rewards.clear()
        self.episode_history.clear()
        self.perception.reset()
        
        # Step 1: Fingerprint the environment (blind — no game name)
        print(f"\n{'=' * 60}")
        print(f"[MetaPolicy] New environment detected (#{self.environments_seen})")
        print(f"[MetaPolicy] Fingerprinting ({self.config.fingerprint_episodes} "
              f"exploration episodes)...")
        
        fp = fingerprint_environment(env, self.config.fingerprint_episodes)
        self.current_fingerprint = fp
        
        print(fp.summary())
        
        # Step 2: Search for similar policy
        match = self.policy_tree.find_best_match(fp)
        
        info = env.get_info()
        n_inputs = env.n_features
        n_outputs = info['n_actions']
        
        if match is not None:
            # Branch from matching policy
            sim = fp.similarity(match.fingerprint)
            print(f"[MetaPolicy] Found similar policy {match.id} "
                  f"(similarity={sim:.3f})")
            
            self.current_policy = self.policy_tree.branch(match.id, fp)
            self.total_policy_branches += 1
            
            # Create pipeline and transfer weights
            pipeline = MetaStackPipeline(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_hidden=128,
            )
            
            if match.weights:
                try:
                    adapted = self._adapt_weights(
                        match.weights, n_inputs, n_outputs
                    )
                    pipeline.ann.set_weights(adapted)
                    pipeline._sync_target_network()
                    print(f"[MetaPolicy] Transferred weights from policy {match.id}")
                except Exception as e:
                    print(f"[MetaPolicy] Weight transfer failed: {e}")
            
            # Apply transferable concepts
            transferable = self.concept_library.find_transferable(fp)
            if transferable:
                try:
                    weights = pipeline.ann.get_weights()
                    weights = self.concept_library.apply_concepts_to_weights(
                        weights, transferable
                    )
                    pipeline.ann.set_weights(weights)
                    pipeline._sync_target_network()
                    print(f"[MetaPolicy] Applied {len(transferable)} transferable concepts")
                except Exception as e:
                    print(f"[MetaPolicy] Concept application failed: {e}")
            
        else:
            # Create new root policy (truly novel environment)
            print(f"[MetaPolicy] No similar policy found — creating new root")
            
            self.current_policy = self.policy_tree.create_root(fp)
            self.total_policy_creates += 1
            
            pipeline = MetaStackPipeline(
                n_inputs=n_inputs,
                n_outputs=n_outputs,
                n_hidden=128,
            )
        
        self.current_pipeline = pipeline
        return pipeline
    
    # ── Per-step recording ─────────────────────────────────────
    
    def on_step(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray):
        """Record a step for concept discovery, perception, and dreamer."""
        # Feed concept library
        self.concept_library.record_step(state, action, reward, next_state)
        
        # Feed perception hub (visual + causal)
        self.perception.record(state, action, reward, next_state)
        
        # Feed basal ganglia world model (learns from every real step)
        self.basal_ganglia.learn(state, action, next_state, reward)
        self._last_observed_state = state
        
        # Record policy vs dreamer agreement (for dreamer_reliance tracking)
        dreamer_rec = self.dreamer_teacher.get_best_action(state)
        dreamer_action = dreamer_rec[0] if dreamer_rec else None
        self.dreamer_teacher.record_policy_action(
            state, action, dreamer_action, reward
        )
    
    # ── Episode lifecycle ──────────────────────────────────────
    
    def on_episode_complete(self, reward: float):
        """
        Called after each episode completes.
        
        1. Update performance tracker
        2. Check for plateau (via RiskSensor)
        3. Periodic concept discovery (via PolicyMonitor)
        4. Policy promotion check (via PolicyMonitor)
        """
        self.episode_count += 1
        self.episode_rewards.append(reward)
        self.episode_history.append({'reward': reward, 'episode': self.episode_count})
        
        if self.current_policy:
            self.current_policy.performance.update(reward)
        
        # Periodic concept discovery (PolicyMonitor decides when)
        if self.policy_monitor.should_discover_concepts(self.episode_count):
            new_concepts = self.concept_library.discover_concepts(
                self.episode_history, self.current_fingerprint
            )
            if new_concepts:
                self.total_concepts_discovered += len(new_concepts)
                if self.current_policy:
                    self.current_policy.discovered_concepts.extend(
                        [c.id for c in new_concepts]
                    )
        
        # Policy promotion check (PolicyMonitor decides)
        if self.policy_monitor.check_promotion(self.current_policy, self.episode_count):
            self.policy_tree.promote(self.current_policy.id)
        
        # Plateau detection (RiskSensor)
        if self._is_plateauing():
            if self.current_policy:
                self.current_policy.performance.plateau_count += 1
            
            # Check if should retire (with failure mode context)
            dominant_failure = self.perception.get_dominant_failure_mode()
            failure_mode_str = dominant_failure.value if dominant_failure else None
            
            if self.current_policy and self.policy_monitor.check_retirement(
                self.current_policy,
                self.risk_sensor.risk_level(self.episode_rewards),
                failure_mode_str
            ):
                print(f"[MetaPolicy] Policy {self.current_policy.id} is "
                      f"chronically plateaued - consider retirement")
        
        # Basal Ganglia dream check (if calibrated)
        if self.basal_ganglia.is_calibrated and self.basal_ganglia.should_dream():
            hypotheses = self.basal_ganglia.create_default_hypotheses(
                self.basal_ganglia.n_actions
            )
            # Use last observed state if available
            dream_state = np.zeros(self.basal_ganglia.state_size)
            dream_results = self.basal_ganglia.dream(
                dream_state, hypotheses, n_steps=10
            )
            if dream_results:
                # Feed DreamerTeacher — builds profiles, options, teaching signals
                dream_state_for_teacher = (
                    self._last_observed_state
                    if self._last_observed_state is not None
                    else dream_state
                )
                teaching_signals = self.dreamer_teacher.process_dream_results(
                    dream_results, dream_state_for_teacher, self.episode_count
                )
                
                # Dynamically adjust dream interval based on reliance
                self.basal_ganglia.dream_interval = (
                    self.dreamer_teacher.recommended_dream_interval
                )
                
                # Amygdala danger check
                danger = self.amygdala.assess_danger(
                    dream_results, current_step=self.episode_count
                )
                if self.amygdala.should_override(danger, self.episode_count):
                    self.amygdala.record_override(self.episode_count)
                    print(f"[Amygdala] {danger.summary()}")
                    print(f"[Amygdala] Override recommended: "
                          f"{danger.recommended_action.value}")
        
        # Update policy monitor mode
        risk_level = self.risk_sensor.risk_level(self.episode_rewards)
        self.policy_monitor.update_mode(self.episode_count, risk_level)
        
        # Check save-state triggers
        save_state = self.save_state_manager.check_triggers(
            perception=self.perception,
            episode=self.episode_count,
            rewards=list(self.episode_rewards),
            current_mode=self.policy_monitor.mode,
        )
        if save_state:
            print(f"[MetaPolicy] FLAGGED ep {self.episode_count}: "
                  f"{save_state.trigger.value} "
                  f"(importance={save_state.importance:.2f})")
        
        return self._get_meta_status()
    
    def on_environment_done(self):
        """
        Called when leaving an environment.
        
        1. Save pipeline weights to policy tree
        2. Run final concept discovery
        3. Update transferability scores
        """
        if self.current_policy and self.current_pipeline:
            # Save weights
            weights = self.current_pipeline.ann.get_weights()
            self.policy_tree.update_weights(self.current_policy.id, weights)
            
            # Final concept discovery
            new_concepts = self.concept_library.discover_concepts(
                self.episode_history, self.current_fingerprint
            )
            if new_concepts:
                self.total_concepts_discovered += len(new_concepts)
    
    # ── LLM/Tetra interaction (delegates to PrefrontalCortex) ──
    
    def get_abstract_llm_prompt(self) -> Optional[str]:
        """
        Generate an LLM prompt with ONLY numerical data.
        NO game names. NO domain hints.
        
        Returns None if LLM query not needed or on cooldown.
        """
        if not self._is_plateauing():
            return None
        
        if self.episode_count - self.prefrontal.last_llm_query < self.config.llm_cooldown:
            return None
        
        self.prefrontal.last_llm_query = self.episode_count
        
        self.perception.update_patterns()
        concepts = self.concept_library.find_transferable(self.current_fingerprint) if self.current_fingerprint else None
        
        return self.prefrontal.build_prompt(
            self.current_fingerprint, self.perception, self.episode_rewards,
            self.episode_count, self._plateau_duration(), concepts
        )
    
    def test_hypothesis_with_tetra(self, pipeline, force: bool = False) -> Dict:
        """
        Full dialogue loop with Tetra.
        
        Delegates to PrefrontalCortex. Public API unchanged.
        """
        return self.prefrontal.query_hypothesis(
            perception=self.perception,
            risk_sensor=self.risk_sensor,
            pipeline=pipeline,
            fingerprint=self.current_fingerprint,
            rewards=self.episode_rewards,
            episode_count=self.episode_count,
            concept_library=self.concept_library,
            force=force,
        )
    
    def report_hypothesis_results(self, test_rewards: List[float], 
                                   baseline_reward: float) -> Optional[str]:
        """
        Report hypothesis test results to Tetra for refinement.
        
        Delegates to PrefrontalCortex. Public API unchanged.
        """
        return self.prefrontal.report_results(test_rewards, baseline_reward)
    
    # ── Internal helpers ───────────────────────────────────────
    
    def _is_plateauing(self) -> bool:
        """Detect plateau. Delegates to RiskSensor."""
        return self.risk_sensor.is_plateauing(self.episode_rewards)
    
    def _plateau_duration(self) -> int:
        """How many episodes since last improvement. Delegates to RiskSensor."""
        return self.risk_sensor.plateau_duration(self.episode_rewards)
    
    def _adapt_weights(self, source_weights: dict, 
                        target_n_inputs: int, target_n_outputs: int) -> dict:
        """
        Adapt weight dimensions for cross-environment transfer.
        
        When source and target environments have different action spaces,
        this copies shared dimensions and initializes new ones randomly.
        """
        adapted = {}
        
        for key, value in source_weights.items():
            if not isinstance(value, np.ndarray):
                adapted[key] = value
                continue
            
            if value.ndim == 1:
                # Bias-like: resize to target
                if 'output' in key or key.endswith('_b'):
                    new = np.zeros(target_n_outputs)
                    shared = min(len(value), target_n_outputs)
                    new[:shared] = value[:shared]
                    if target_n_outputs > shared:
                        new[shared:] = np.random.randn(target_n_outputs - shared) * 0.01
                    adapted[key] = new
                else:
                    adapted[key] = value.copy()
            
            elif value.ndim == 2:
                rows, cols = value.shape
                
                if 'output' in key or key.endswith('_w') and cols != target_n_outputs:
                    new_rows = rows
                    new_cols = target_n_outputs
                    new = np.random.randn(new_rows, new_cols) * 0.01
                    shared_cols = min(cols, new_cols)
                    new[:, :shared_cols] = value[:, :shared_cols]
                    adapted[key] = new
                
                elif 'input' in key or key.startswith('w_'):
                    new_rows = target_n_inputs
                    new_cols = cols
                    new = np.random.randn(new_rows, new_cols) * 0.01
                    shared_rows = min(rows, new_rows)
                    new[:shared_rows, :] = value[:shared_rows, :]
                    adapted[key] = new
                else:
                    adapted[key] = value.copy()
            else:
                adapted[key] = value.copy()
        
        return adapted
    
    def _get_meta_status(self) -> dict:
        """Get current meta-status for logging."""
        return {
            'episode': self.episode_count,
            'policy_id': self.current_policy.id if self.current_policy else None,
            'policy_status': self.current_policy.status if self.current_policy else None,
            'avg_reward': float(np.mean(list(self.episode_rewards)[-25:])) if self.episode_rewards else 0,
            'plateauing': self._is_plateauing(),
            'concepts_in_library': len(self.concept_library.concepts),
            'policies_in_tree': len(self.policy_tree.nodes),
            'mode': self.policy_monitor.mode,
            'dreamer_calibrated': self.basal_ganglia.is_calibrated,
            'amygdala_alertness': self.amygdala.alertness,
            'dreamer_reliance': self.dreamer_teacher.dreamer_reliance,
            'dreamer_needed': self.dreamer_teacher.dreamer_is_needed,
            'active_options': len(self.dreamer_teacher.options.active_options),
            'dream_interval': self.basal_ganglia.dream_interval,
        }
    
    def summary(self) -> str:
        """Full meta-policy summary."""
        lines = [
            "=" * 50,
            "META-POLICY CONTROLLER REPORT",
            "=" * 50,
            f"Environments seen: {self.environments_seen}",
            f"Total concepts discovered: {self.total_concepts_discovered}",
            f"Policy branches: {self.total_policy_branches}",
            f"Policy creates: {self.total_policy_creates}",
            f"Hypotheses tested: {self.total_hypotheses_tested}",
            f"Current mode: {self.policy_monitor.mode}",
            "",
            self.policy_tree.summary(),
            "",
            self.concept_library.summary(),
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    """Test the full meta-policy controller."""
    import sys
    sys.path.insert(0, '.')
    from throng4.environments.atari_adapter import AtariAdapter
    
    print("=" * 60)
    print("META-POLICY CONTROLLER TEST (BLIND)")
    print("=" * 60)
    
    controller = MetaPolicyController(ControllerConfig(
        fingerprint_episodes=10,  # Fewer for quick test
        concept_discovery_interval=25,
        promote_after_episodes=30,
    ))
    
    # Test on 3 games (controller doesn't know game names)
    games = ['Breakout', 'Pong', 'SpaceInvaders']
    
    for game in games:
        print(f"\n{'#' * 60}")
        print(f"# Testing environment (name hidden from controller)")
        print(f"{'#' * 60}")
        
        env = AtariAdapter(game)
        
        # Controller sees only the environment interface, not the name
        pipeline = controller.on_new_environment(env)
        
        # Run 50 episodes
        for ep in range(50):
            state = env.reset()
            done = False
            episode_reward = 0.0
            steps = 0
            
            while not done and steps < 500:
                action = pipeline.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)
                
                # Record for concept discovery
                controller.on_step(state, action, reward, next_state)
                
                pipeline.update(state, action, reward, next_state, done)
                episode_reward += reward
                state = next_state
                steps += 1
            
            meta_status = controller.on_episode_complete(episode_reward)
            
            if (ep + 1) % 25 == 0:
                print(f"  Ep {ep+1}: avg={meta_status['avg_reward']:.1f}, "
                      f"policy={meta_status['policy_id']}, "
                      f"concepts={meta_status['concepts_in_library']}")
        
        controller.on_environment_done()
        env.close()
        
        # Check LLM prompt (should have NO game names)
        prompt = controller.get_abstract_llm_prompt()
        if prompt:
            print(f"\n  LLM prompt (BLIND — no game names):")
            print(f"  {prompt[:200]}...")
    
    # Final report
    print(f"\n{controller.summary()}")
    
    # Validation: verify no game names leaked
    summary = controller.summary()
    for game in games:
        assert game not in summary, f"LEAK: {game} found in summary!"
    
    print("\n✅ No game names leaked to controller!")
    print("✅ MetaPolicyController test complete!")
