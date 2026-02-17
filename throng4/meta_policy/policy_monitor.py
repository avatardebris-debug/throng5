"""
PolicyMonitor — Watches performance and manages policy lifecycle.

Throng5 role: Policy Monitor — watches dreams for positive Δ, flags save-states,
manages adaptive-mode-only overrides.
"""

from typing import Optional


class PolicyMonitor:
    """
    Monitors policy performance, handles promotion and retirement.
    
    In Throng5, this becomes the Policy Monitor:
      - Watches basal ganglia dreams for major positive Δ
      - Flags save-states for post-game review
      - Overrides base model only in adaptive mode
    """
    
    MODES = ('exploratory', 'learning', 'adaptive')
    
    def __init__(self, promote_after_episodes: int = 50,
                 concept_discovery_interval: int = 50):
        self.promote_after_episodes = promote_after_episodes
        self.concept_discovery_interval = concept_discovery_interval
        self._mode = 'exploratory'
    
    @property
    def mode(self) -> str:
        """Current operating mode: exploratory, learning, or adaptive."""
        return self._mode
    
    @mode.setter
    def mode(self, value: str):
        if value not in self.MODES:
            raise ValueError(f"Invalid mode: {value}. Must be one of {self.MODES}")
        self._mode = value
    
    def check_promotion(self, policy, episode_count: int):
        """Check if policy should be promoted from candidate to established."""
        if (policy and 
            policy.status == 'candidate' and
            episode_count >= self.promote_after_episodes):
            if policy.performance.is_improving:
                return True
        return False
    
    def should_discover_concepts(self, episode_count: int) -> bool:
        """Check if it's time for periodic concept discovery."""
        return (episode_count > 0 and 
                episode_count % self.concept_discovery_interval == 0)
    
    def check_retirement(self, policy, risk_level: str, 
                         dominant_failure_mode: Optional[str] = None) -> bool:
        """
        Check if policy should be retired.
        
        Uses failure mode to make smarter decisions:
        - Strategic failures → retire faster (wrong policy, unlikely to improve)
        - Temporal failures → be patient (timing issue, learnable)
        - Mechanical failures → investigate (environment constraint)
        - Unknown failures → use risk level only
        
        In Throng5, this will also check dream quality and 
        whether the basal ganglia sees better alternatives.
        """
        if not policy:
            return False
        
        # Critical risk always retires
        if risk_level == 'critical':
            return True
        
        # Use failure mode for nuanced decisions
        if dominant_failure_mode:
            if dominant_failure_mode == 'strategic':
                # Strategic failures = wrong policy, retire on declining risk
                return risk_level in ('declining', 'critical')
            
            elif dominant_failure_mode == 'temporal':
                # Temporal failures = timing issue, be more patient
                # Only retire on critical risk
                return risk_level == 'critical'
            
            elif dominant_failure_mode == 'mechanical':
                # Mechanical failures = environment constraint
                # Investigate but don't rush to retire
                return risk_level == 'critical'
        
        # Fallback: use risk level only
        return risk_level in ('critical', 'declining')
    
    def update_mode(self, episode_count: int, risk_level: str):
        """
        Auto-update operating mode based on progress.
        
        - exploratory: first 20 episodes
        - learning: episodes 20-50, stable performance
        - adaptive: after 50 episodes if stable
        """
        if episode_count < 20:
            self._mode = 'exploratory'
        elif risk_level in ('critical', 'declining'):
            self._mode = 'exploratory'  # Reset to exploratory on crisis
        elif episode_count >= 50 and risk_level == 'stable':
            self._mode = 'adaptive'
        else:
            self._mode = 'learning'

