"""
Meta-Policy Controller — Top-level blind orchestrator.

Sits above Meta^4 GoalHierarchy and coordinates:
1. Environment fingerprinting (what kind of environment is this?)
2. Policy matching/branching (find or create the right policy)
3. Concept discovery (what patterns transfer?)
4. Abstract LLM reasoning (numbers only, no game names)

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
from throng4.meta_policy.visual_patterns import VisualPatternExtractor, VisualPatterns
from throng4.meta_policy.causal_discovery import CausalDiscovery, ActionEffect
from throng4.meta_policy.hypothesis_executor import HypothesisExecutor, ExecutableStrategy
from throng4.metastack_pipeline import MetaStackPipeline


@dataclass
class ControllerConfig:
    """Configuration for MetaPolicyController."""
    fingerprint_episodes: int = 20        # Episodes for env characterization
    plateau_window: int = 15              # Episodes to detect plateau
    plateau_threshold: float = 0.05       # Improvement below this = plateau
    promote_after_episodes: int = 50      # Promote candidate → active
    concept_discovery_interval: int = 50  # Run discovery every N episodes
    max_concurrent_policies: int = 10     # Max active policies in tree
    llm_cooldown: int = 30                # Min episodes between LLM queries


class MetaPolicyController:
    """
    Top-level blind orchestrator.
    
    Decides WHAT to learn, not HOW to learn.
    Operates without game names or external knowledge.
    
    Lifecycle:
        1. on_new_environment(env) → fingerprint, match/create policy
        2. on_step(state, action, reward, next_state) → record for discovery
        3. on_episode_complete(reward) → track, detect patterns
        4. on_environment_done() → save policy, discover concepts
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None, 
                 llm_client=None):
        self.config = config or ControllerConfig()
        self.llm_client = llm_client  # Optional Tetra client
        self.hypothesis_test_episodes = 10  # Episodes to test each hypothesis
        
        # Core components
        self.policy_tree = PolicyTree()
        self.concept_library = BlindConceptLibrary()
        
        # Hypothesis testing components
        self.visual_extractor = VisualPatternExtractor()
        self.causal_discovery = CausalDiscovery()
        self.hypothesis_executor = HypothesisExecutor()
        
        # Current state
        self.current_policy: Optional[PolicyNode] = None
        self.current_fingerprint: Optional[EnvironmentFingerprint] = None
        self.current_pipeline: Optional[MetaStackPipeline] = None
        self.current_visual_patterns: Optional[VisualPatterns] = None
        self.current_causal_effects: Optional[Dict[int, ActionEffect]] = None
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards = deque(maxlen=200)
        self.episode_history: List[Dict] = []
        self.last_llm_query = -self.config.llm_cooldown
        
        # State/transition tracking for visual/causal discovery
        self.recent_states: deque = deque(maxlen=1000)
        self.recent_transitions: List[Dict] = []
        
        # Hypothesis testing
        self.active_hypothesis: Optional[ExecutableStrategy] = None
        self.hypothesis_start_episode: int = 0
        self.hypothesis_history: List[Dict] = []
        
        # Stats
        self.environments_seen = 0
        self.total_concepts_discovered = 0
        self.total_policy_branches = 0
        self.total_policy_creates = 0
        self.total_hypotheses_tested = 0
    
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
    
    def on_step(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray):
        """Record a step for concept discovery and causal/visual analysis."""
        # Track for concept discovery
        self.concept_library.record_step(state, action, reward, next_state)
        
        # Track states for visual pattern extraction
        self.recent_states.append(state)
        
        # Track transitions for causal discovery
        self.causal_discovery.record_transition(state, action, reward, next_state)
        self.recent_transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
        })
    
    def on_episode_complete(self, reward: float):
        """
        Called after each episode completes.
        
        1. Update performance tracker
        2. Check for plateau
        3. Periodic concept discovery
        4. Policy promotion check
        """
        self.episode_count += 1
        self.episode_rewards.append(reward)
        self.episode_history.append({'reward': reward, 'episode': self.episode_count})
        
        if self.current_policy:
            self.current_policy.performance.update(reward)
        
        # Periodic concept discovery
        if (self.episode_count > 0 and 
            self.episode_count % self.config.concept_discovery_interval == 0):
            new_concepts = self.concept_library.discover_concepts(
                self.episode_history, self.current_fingerprint
            )
            if new_concepts:
                self.total_concepts_discovered += len(new_concepts)
                if self.current_policy:
                    self.current_policy.discovered_concepts.extend(
                        [c.id for c in new_concepts]
                    )
        
        # Policy promotion check
        if (self.current_policy and 
            self.current_policy.status == 'candidate' and
            self.episode_count >= self.config.promote_after_episodes):
            if self.current_policy.performance.is_improving:
                self.policy_tree.promote(self.current_policy.id)
        
        # Plateau detection
        if self._is_plateauing():
            if self.current_policy:
                self.current_policy.performance.plateau_count += 1
            
            # Check if should retire
            if self.current_policy and self.policy_tree.should_retire(
                self.current_policy.id
            ):
                print(f"[MetaPolicy] Policy {self.current_policy.id} is "
                      f"chronically plateaued - consider retirement")
        
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
        if self.episode_history:
            new_concepts = self.concept_library.discover_concepts(
                self.episode_history, self.current_fingerprint
            )
            if new_concepts:
                self.total_concepts_discovered += len(new_concepts)
                if self.current_policy:
                    self.current_policy.discovered_concepts.extend(
                        [c.id for c in new_concepts]
                    )
        
        print(f"\n[MetaPolicy] Environment complete. "
              f"Episodes: {self.episode_count}, "
              f"Avg reward: {np.mean(list(self.episode_rewards)[-25:]):.1f}, "
              f"Concepts discovered: {len(self.current_policy.discovered_concepts) if self.current_policy else 0}")
    
    def _update_visual_causal_patterns(self):
        """Extract current visual and causal patterns from recent data."""
        if len(self.recent_states) > 50:
            self.current_visual_patterns = self.visual_extractor.extract_patterns(
                list(self.recent_states)
            )
        
        if len(self.recent_transitions) > 100:
            self.current_causal_effects = self.causal_discovery.discover_action_effects(
                self.recent_transitions[-500:]
            )
    
    def _build_llm_prompt(self) -> Optional[str]:
        """Build LLM prompt from current data. No cooldown/plateau checks."""
        fp = self.current_fingerprint
        if not fp:
            return None
        
        rewards = list(self.episode_rewards)
        if not rewards:
            return None
        
        # Build prompt from numbers only
        prompt = (
            f"An agent is learning in an unknown environment.\n"
            f"\n"
            f"Environment characteristics (from observation only):\n"
            f"  - {fp.action_count} available actions\n"
            f"  - {fp.state_dim}-dimensional state space\n"
            f"  - Reward density: {fp.reward_density:.0%} of steps have non-zero reward\n"
            f"  - Reward range: [{fp.reward_min:.1f}, {fp.reward_max:.1f}]\n"
            f"  - State change rate: {fp.state_change_rate:.3f}\n"
            f"  - Action diversity: {fp.action_diversity_score:.3f}\n"
        )
        
        # Add visual patterns
        if self.current_visual_patterns:
            prompt += f"\n{self.current_visual_patterns.summary()}\n"
        
        # Add causal discovery
        if self.current_causal_effects:
            prompt += f"\n{self.causal_discovery.get_summary(self.current_causal_effects)}\n"
        
        prompt += (
            f"\n"
            f"Current performance:\n"
            f"  - Episodes completed: {self.episode_count}\n"
            f"  - Current avg reward (last 25): {np.mean(rewards[-25:]):.2f}\n"
            f"  - Best reward: {max(rewards) if rewards else 0:.1f}\n"
            f"  - Plateau duration: {self._plateau_duration()} episodes\n"
        )
        
        # Add discovered concept info (abstracted)
        concepts = self.concept_library.find_transferable(fp)
        if concepts:
            prompt += (
                f"\n"
                f"Discovered patterns:\n"
            )
            for c in concepts[:3]:
                if c.pattern_type == 'state_cluster':
                    prompt += (
                        f"  - State feature {c.pattern_data.get('feature_index', '?')} "
                        f"correlates with reward (r={c.pattern_data.get('correlation', 0):.2f})\n"
                    )
                elif c.pattern_type == 'action_sequence':
                    prompt += (
                        f"  - Action sequence {c.pattern_data.get('sequence', [])} "
                        f"precedes avg reward {c.evidence_reward_boost:.2f}\n"
                    )
                elif c.pattern_type == 'reward_spike':
                    prompt += (
                        f"  - Reward spike of {c.pattern_data.get('spike_magnitude', 0):.1f} "
                        f"std devs at episode {c.pattern_data.get('episode_index', '?')}\n"
                    )
        
        prompt += (
            f"\n"
            f"What learning strategy adjustments would you suggest? "
            f"Consider: exploration rate, learning rate, "
            f"whether to focus on specific state features or action patterns."
        )
        
        return prompt

    def get_abstract_llm_prompt(self) -> Optional[str]:
        """
        Generate an LLM prompt with ONLY numerical data.
        NO game names. NO domain hints.
        
        Returns None if LLM query not needed or on cooldown.
        """
        if not self._is_plateauing():
            return None
        
        if self.episode_count - self.last_llm_query < self.config.llm_cooldown:
            return None
        
        self.last_llm_query = self.episode_count
        
        self._update_visual_causal_patterns()
        return self._build_llm_prompt()
    
    def _adapt_weights(self, source_weights: dict, 
                        target_n_inputs: int, target_n_outputs: int) -> dict:
        """
        Adapt weight dimensions for cross-environment transfer.
        
        When source and target environments have different action spaces,
        this copies shared dimensions and initializes new ones randomly.
        """
        adapted = {}
        
        for key, w in source_weights.items():
            src_shape = w.shape
            
            # Determine target shape based on weight role
            if key == 'W1':
                # Input → hidden: adapt input dim
                target_shape = (target_n_inputs, src_shape[1])
            elif key == 'b1':
                # Hidden bias: keep as-is
                target_shape = src_shape
            elif key == 'W_q':
                # Q-head: hidden → actions (adapt to new action count)
                target_shape = (src_shape[0], target_n_outputs)
            elif key == 'b_q':
                # Q-head bias: adapt to new action count
                target_shape = (target_n_outputs,)
            elif key in ('W_r', 'b_r'):
                # Reward head: always 1 output, never resize
                target_shape = src_shape
            else:
                target_shape = src_shape
            
            if src_shape == target_shape:
                adapted[key] = w.copy()
            else:
                # Create new weights, copy what fits
                new_w = np.random.randn(*target_shape).astype(np.float32) * 0.01
                
                # Copy overlapping dimensions
                slices = tuple(
                    slice(0, min(s, t)) 
                    for s, t in zip(src_shape, target_shape)
                )
                new_w[slices] = w[slices]
                
                adapted[key] = new_w
                print(f"  [MetaPolicy] Adapted {key}: {src_shape} → {target_shape}")
        
        return adapted
    
    def _is_plateauing(self) -> bool:
        """Detect plateau from reward history."""
        if len(self.episode_rewards) < self.config.plateau_window * 2:
            return False
        
        rewards = list(self.episode_rewards)
        recent = np.mean(rewards[-self.config.plateau_window:])
        previous = np.mean(rewards[-2*self.config.plateau_window:-self.config.plateau_window])
        
        if abs(previous) < 1e-8:
            return abs(recent) < 1e-8  # Both near zero = plateau
        
        improvement = (recent - previous) / abs(previous)
        return improvement < self.config.plateau_threshold
    
    def _plateau_duration(self) -> int:
        """How many episodes since last significant improvement."""
        if len(self.episode_rewards) < 10:
            return 0
        
        rewards = list(self.episode_rewards)
        best_so_far = float('-inf')
        duration = 0
        
        for r in reversed(rewards):
            if r > best_so_far * 1.1:  # 10% improvement
                break
            duration += 1
            best_so_far = max(best_so_far, r)
        
        return duration
    
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
            "",
            self.policy_tree.summary(),
            "",
            self.concept_library.summary(),
        ]
        return "\n".join(lines)
    
    def test_hypothesis_with_tetra(self, pipeline, force: bool = False) -> Dict:
        """
        Full dialogue loop with Tetra:
        1. Detect plateau (or force query)
        2. Extract visual/causal patterns
        3. Query Tetra with enhanced prompt
        4. Parse response into strategy
        5. Apply strategy to pipeline
        
        Args:
            pipeline: The current MetaStackPipeline
            force: If True, skip plateau check (for stress testing)
        
        Returns dict with status and strategy info.
        Caller should then run test episodes and call report_hypothesis_results().
        """
        if not self.llm_client:
            return {'status': 'no_llm_client'}
        
        if not force and not self._is_plateauing():
            return {'status': 'not_plateauing'}
        
        # Generate enhanced prompt (bypass cooldown when forced)
        if force:
            # Force: generate prompt directly, skip plateau/cooldown checks
            self._update_visual_causal_patterns()
            prompt = self._build_llm_prompt()
        else:
            prompt = self.get_abstract_llm_prompt()
        
        if not prompt:
            return {'status': 'cooldown'}
        
        print(f"\n[Tetra] Querying with enhanced prompt...")
        print(f"[Tetra] Prompt preview: {prompt[:200]}...")
        
        # Query Tetra
        tetra_response = self.llm_client.query(prompt)
        
        if "Error" in tetra_response:
            print(f"[Tetra] {tetra_response}")
            return {'status': 'error', 'error': tetra_response}
        
        print(f"[Tetra] Response: {tetra_response[:200]}...")
        
        # Parse into strategy
        strategy = self.hypothesis_executor.parse_hypothesis(
            tetra_response,
            visual_patterns=self.current_visual_patterns.__dict__ if self.current_visual_patterns else None,
            causal_effects=self.current_causal_effects,
        )
        
        print(f"[Tetra] Parsed strategy: {strategy.summary()}")
        
        # Apply to pipeline
        modifications = self.hypothesis_executor.apply_strategy(strategy, pipeline)
        
        print(f"[Tetra] Applied modifications: {modifications}")
        
        # Record baseline
        baseline_reward = np.mean(list(self.episode_rewards)[-25:])
        self.active_hypothesis = strategy
        self.hypothesis_start_episode = self.episode_count
        
        return {
            'status': 'hypothesis_applied',
            'strategy': strategy,
            'modifications': modifications,
            'baseline_reward': baseline_reward,
            'tetra_response': tetra_response,
        }
    
    def report_hypothesis_results(self, test_rewards: List[float], 
                                   baseline_reward: float) -> Optional[str]:
        """
        Report hypothesis test results to Tetra for refinement.
        
        Args:
            test_rewards: Rewards from test episodes
            baseline_reward: Baseline reward before hypothesis
            
        Returns:
            Tetra's refinement suggestion or None
        """
        if not self.llm_client or not self.active_hypothesis:
            return None
        
        avg_test = np.mean(test_rewards)
        improvement = avg_test - baseline_reward
        
        # Build report prompt
        report = (
            f"Hypothesis test results:\n"
            f"  Strategy: {self.active_hypothesis.name}\n"
            f"  Baseline reward: {baseline_reward:.1f}\n"
            f"  Test reward: {avg_test:.1f}\n"
            f"  Improvement: {improvement:+.1f}\n"
            f"\n"
            f"Should we continue with this strategy, refine it, or try something else?"
        )
        
        print(f"\n[Tetra] Reporting results...")
        print(f"[Tetra] Improvement: {improvement:+.1f}")
        
        refinement = self.llm_client.query(report)
        
        print(f"[Tetra] Refinement: {refinement[:200]}...")
        
        # Record in history
        self.hypothesis_history.append({
            'strategy': self.active_hypothesis.name,
            'baseline': baseline_reward,
            'test_avg': avg_test,
            'improvement': improvement,
            'refinement': refinement,
        })
        
        self.total_hypotheses_tested += 1
        
        return refinement



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
