"""
PrefrontalCortex — Owns all LLM/Tetra interaction.

Throng5 role: Prefrontal Cortex — strategic guidance, cross-brain communication,
abstract reasoning via LLM.
"""

from collections import deque
from typing import Dict, List, Optional

import numpy as np

from throng4.meta_policy.hypothesis_executor import HypothesisExecutor, ExecutableStrategy


class PrefrontalCortex:
    """
    All LLM/Tetra interaction: prompt building, hypothesis testing, result reporting.
    
    In Throng5, this becomes the Prefrontal Cortex:
      - Receives structured reports from all brain regions
      - Provides strategic guidance to Policy Monitor
      - Can ask sub-agents for explanations ("why is strategy B better?")
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.hypothesis_executor = HypothesisExecutor()
        
        # Hypothesis tracking
        self.active_hypothesis: Optional[ExecutableStrategy] = None
        self.hypothesis_start_episode: int = 0
        self.hypothesis_history: List[Dict] = []
        self.total_hypotheses_tested: int = 0
        self.hypothesis_test_episodes: int = 10
        
        # Cooldown
        self.last_llm_query: int = -30  # Start ready to query
    
    def build_prompt(self, fingerprint, perception, rewards: deque,
                     episode_count: int, plateau_duration: int,
                     concepts: list = None) -> Optional[str]:
        """
        Build LLM prompt from current data. NO game names. Numbers only.
        
        Args:
            fingerprint: Environment fingerprint
            perception: PerceptionHub with extracted patterns
            rewards: Recent episode rewards
            episode_count: Total episodes completed
            plateau_duration: Episodes since last improvement
            concepts: Discovered transferable concepts
        """
        if not fingerprint:
            return None
        
        reward_list = list(rewards)
        if not reward_list:
            return None
        
        fp = fingerprint
        
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
        
        # Add visual patterns from perception hub
        if perception.visual_patterns:
            prompt += f"\n{perception.visual_patterns.summary()}\n"
        
        # Add causal discovery from perception hub
        if perception.causal_effects:
            prompt += f"\n{perception.get_causal_summary()}\n"
        
        # Add failure mode analysis
        failure_summary = perception.get_failure_summary()
        if failure_summary and "No failures" not in failure_summary:
            prompt += f"\n{failure_summary}\n"
        
        # Add prediction error / surprise detection
        surprise = perception.get_surprise_level()
        anomaly = perception.get_anomaly_score()
        if surprise > 0.3 or anomaly > 0.5:
            prompt += f"\nAgent surprise/anomaly detection:\n"
            prompt += f"  - Surprise level: {surprise:.2f} (0=normal, 1=very surprised)\n"
            prompt += f"  - Anomaly score: {anomaly:.2f} (0=normal, 1=anomaly detected)\n"
        
        prompt += (
            f"\n"
            f"Current performance:\n"
            f"  - Episodes completed: {episode_count}\n"
            f"  - Current avg reward (last 25): {np.mean(reward_list[-25:]):.2f}\n"
            f"  - Best reward: {max(reward_list) if reward_list else 0:.1f}\n"
            f"  - Plateau duration: {plateau_duration} episodes\n"
        )
        
        # Add discovered concept info
        if concepts:
            prompt += "\nDiscovered patterns:\n"
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
    
    def query_hypothesis(self, perception, risk_sensor, pipeline,
                         fingerprint, rewards: deque, episode_count: int,
                         concept_library=None,
                         force: bool = False) -> Dict:
        """
        Full dialogue loop with Tetra: query, parse, apply hypothesis.
        
        Args:
            perception: PerceptionHub for pattern data
            risk_sensor: RiskSensor for plateau info
            pipeline: MetaStackPipeline to modify
            fingerprint: Environment fingerprint
            rewards: Episode reward history
            episode_count: Current episode count
            concept_library: For finding transferable concepts
            force: Skip plateau/cooldown checks
            
        Returns:
            dict with status and strategy info
        """
        if not self.llm_client:
            return {'status': 'no_llm_client'}
        
        if not force and not risk_sensor.is_plateauing(rewards):
            return {'status': 'not_plateauing'}
        
        # Build prompt (bypass cooldown when forced)
        if force:
            perception.update_patterns()
        else:
            # Check cooldown
            if episode_count - self.last_llm_query < 30:
                return {'status': 'cooldown'}
            self.last_llm_query = episode_count
            perception.update_patterns()
        
        # Get concepts if library available
        concepts = None
        if concept_library and fingerprint:
            concepts = concept_library.find_transferable(fingerprint)
        
        prompt = self.build_prompt(
            fingerprint, perception, rewards, episode_count,
            risk_sensor.plateau_duration(rewards), concepts
        )
        
        if not prompt:
            return {'status': 'cooldown'}
        
        print(f"\n[PrefrontalCortex] Querying Tetra...")
        print(f"[PrefrontalCortex] Prompt preview: {prompt[:200]}...")
        
        # Query Tetra
        tetra_response = self.llm_client.query(prompt)
        
        if "Error" in tetra_response:
            print(f"[PrefrontalCortex] {tetra_response}")
            return {'status': 'error', 'error': tetra_response}
        
        print(f"[PrefrontalCortex] Response: {tetra_response[:200]}...")
        
        # Parse into strategy
        strategy = self.hypothesis_executor.parse_hypothesis(
            tetra_response,
            visual_patterns=perception.visual_patterns.__dict__ if perception.visual_patterns else None,
            causal_effects=perception.causal_effects,
        )
        
        print(f"[PrefrontalCortex] Parsed strategy: {strategy.summary()}")
        
        # Apply to pipeline
        modifications = self.hypothesis_executor.apply_strategy(strategy, pipeline)
        
        print(f"[PrefrontalCortex] Applied modifications: {modifications}")
        
        # Record baseline
        baseline_reward = np.mean(list(rewards)[-25:])
        self.active_hypothesis = strategy
        self.hypothesis_start_episode = episode_count
        
        return {
            'status': 'hypothesis_applied',
            'strategy': strategy,
            'modifications': modifications,
            'baseline_reward': baseline_reward,
            'tetra_response': tetra_response,
        }
    
    def report_results(self, test_rewards: List[float],
                       baseline_reward: float) -> Optional[str]:
        """
        Report hypothesis test results to Tetra for refinement.
        
        Returns Tetra's refinement suggestion or None.
        """
        if not self.llm_client or not self.active_hypothesis:
            return None
        
        avg_test = np.mean(test_rewards)
        improvement = avg_test - baseline_reward
        
        report = (
            f"Hypothesis test results:\n"
            f"  Strategy: {self.active_hypothesis.name}\n"
            f"  Baseline reward: {baseline_reward:.1f}\n"
            f"  Test reward: {avg_test:.1f}\n"
            f"  Improvement: {improvement:+.1f}\n"
            f"\n"
            f"Should we continue with this strategy, refine it, or try something else?"
        )
        
        print(f"\n[PrefrontalCortex] Reporting results...")
        print(f"[PrefrontalCortex] Improvement: {improvement:+.1f}")
        
        refinement = self.llm_client.query(report)
        
        print(f"[PrefrontalCortex] Refinement: {refinement[:200]}...")
        
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
