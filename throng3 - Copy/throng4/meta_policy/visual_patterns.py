"""
Visual Pattern Extraction — Discover structural patterns from raw observations.

Extracts visual/structural information WITHOUT knowing game names:
- Entity count (independently moving regions in state space)
- Motion patterns (synchronized, random, tracking, bouncing)
- Spatial layout (grid, scattered, clustered, linear)
- Collision frequency (state changes when entities overlap)

This enables the LLM to reason about game structure abstractly.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class VisualPatterns:
    """Discovered visual/structural patterns from observations."""
    entity_count: int
    motion_type: str  # synchronized | random | tracking | bouncing | static
    spatial_layout: str  # grid | scattered | clustered | linear | sparse
    collision_frequency: float  # % of timesteps with collisions
    state_change_regions: List[int]  # which RAM bytes are active
    motion_periodicity: float  # 0-1, how periodic is the motion
    
    def summary(self) -> str:
        """Human-readable summary for LLM prompts."""
        return (
            f"Visual Patterns:\n"
            f"  - Entity count: ~{self.entity_count} independently moving regions\n"
            f"  - Motion: {self.motion_type} (periodicity={self.motion_periodicity:.2f})\n"
            f"  - Layout: {self.spatial_layout}\n"
            f"  - Collisions: {self.collision_frequency:.1%} of timesteps\n"
            f"  - Active state regions: {len(self.state_change_regions)} features"
        )


class VisualPatternExtractor:
    """
    Discover visual/structural patterns from state transitions.
    
    Works by analyzing which RAM bytes change, how they change,
    and correlations between changes — all without pixel access.
    """
    
    def __init__(self, min_change_threshold: float = 0.01):
        self.min_change_threshold = min_change_threshold
    
    def extract_patterns(self, state_history: List[np.ndarray]) -> VisualPatterns:
        """
        Analyze state transitions to discover patterns.
        
        Args:
            state_history: List of state vectors from recent episodes
            
        Returns:
            VisualPatterns with discovered structure
        """
        if len(state_history) < 10:
            return self._default_patterns()
        
        states = np.array(state_history)
        
        # Find which features are active (change over time)
        active_features = self._find_active_features(states)
        
        # Count independently moving entities
        entity_count = self._count_entities(states, active_features)
        
        # Classify motion patterns
        motion_type, periodicity = self._classify_motion(states, active_features)
        
        # Analyze spatial layout
        spatial_layout = self._analyze_layout(states, active_features)
        
        # Detect collision frequency
        collision_freq = self._detect_collisions(states)
        
        return VisualPatterns(
            entity_count=entity_count,
            motion_type=motion_type,
            spatial_layout=spatial_layout,
            collision_frequency=collision_freq,
            state_change_regions=active_features,
            motion_periodicity=periodicity,
        )
    
    def _find_active_features(self, states: np.ndarray) -> List[int]:
        """Find which state features change significantly over time."""
        # Compute variance across time for each feature
        variances = np.var(states, axis=0)
        
        # Features with variance above threshold are "active"
        active = np.where(variances > self.min_change_threshold)[0]
        
        return active.tolist()
    
    def _count_entities(self, states: np.ndarray, active_features: List[int]) -> int:
        """
        Count independently moving entities.
        
        Strategy: cluster active features by correlation.
        Features that move together = same entity.
        """
        if len(active_features) < 2:
            return 1
        
        # Extract active feature time series
        active_states = states[:, active_features]
        
        # Compute pairwise correlations
        correlations = np.corrcoef(active_states.T)
        
        # Count clusters of highly correlated features
        # (features with correlation > 0.7 = same entity)
        visited = set()
        entity_count = 0
        
        for i in range(len(active_features)):
            if i in visited:
                continue
            
            # Start new entity cluster
            entity_count += 1
            cluster = {i}
            visited.add(i)
            
            # Add correlated features to cluster
            for j in range(i + 1, len(active_features)):
                if j not in visited and abs(correlations[i, j]) > 0.7:
                    cluster.add(j)
                    visited.add(j)
        
        return max(1, entity_count)
    
    def _classify_motion(self, states: np.ndarray, 
                         active_features: List[int]) -> Tuple[str, float]:
        """
        Classify motion pattern and compute periodicity.
        
        Returns:
            (motion_type, periodicity_score)
        """
        if len(active_features) == 0:
            return "static", 0.0
        
        active_states = states[:, active_features]
        
        # Compute autocorrelation to detect periodicity
        autocorr = self._compute_autocorrelation(active_states)
        
        # Compute synchronization (do features move together?)
        sync_score = self._compute_synchronization(active_states)
        
        # Classify based on autocorr and sync
        if autocorr > 0.6:
            motion_type = "synchronized" if sync_score > 0.5 else "bouncing"
            periodicity = autocorr
        elif sync_score > 0.7:
            motion_type = "synchronized"
            periodicity = sync_score
        elif autocorr < 0.2 and sync_score < 0.3:
            motion_type = "random"
            periodicity = 0.0
        else:
            motion_type = "tracking"
            periodicity = (autocorr + sync_score) / 2
        
        return motion_type, periodicity
    
    def _compute_autocorrelation(self, states: np.ndarray, lag: int = 5) -> float:
        """Compute average autocorrelation across features."""
        if len(states) < lag * 2:
            return 0.0
        
        autocorrs = []
        for feature_idx in range(states.shape[1]):
            series = states[:, feature_idx]
            
            # Pearson correlation between series[:-lag] and series[lag:]
            if len(series) > lag:
                corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrs.append(abs(corr))
        
        return np.mean(autocorrs) if autocorrs else 0.0
    
    def _compute_synchronization(self, states: np.ndarray) -> float:
        """How synchronized are feature changes?"""
        if states.shape[1] < 2:
            return 0.0
        
        # Compute when each feature changes significantly
        changes = np.abs(np.diff(states, axis=0)) > self.min_change_threshold
        
        # Synchronization = how often features change at the same time
        change_counts = np.sum(changes, axis=1)
        
        # High sync = many features change together
        avg_simultaneous = np.mean(change_counts)
        max_possible = states.shape[1]
        
        return avg_simultaneous / max(1, max_possible)
    
    def _analyze_layout(self, states: np.ndarray, active_features: List[int]) -> str:
        """
        Classify spatial layout.
        
        Strategy: look at distribution of active features in state space.
        """
        if len(active_features) < 3:
            return "sparse"
        
        # Check if active features are evenly spaced (grid-like)
        gaps = np.diff(sorted(active_features))
        gap_variance = np.var(gaps)
        
        # Check clustering (are active features grouped?)
        active_density = len(active_features) / len(states[0])
        
        if gap_variance < 2.0 and active_density > 0.1:
            return "grid"
        elif gap_variance < 5.0:
            return "linear"
        elif active_density > 0.3:
            return "clustered"
        else:
            return "scattered"
    
    def _detect_collisions(self, states: np.ndarray) -> float:
        """
        Detect collision frequency.
        
        Collision = sudden large change in multiple features simultaneously.
        """
        if len(states) < 2:
            return 0.0
        
        # Compute state deltas
        deltas = np.abs(np.diff(states, axis=0))
        
        # Large change = delta > 3 std devs
        threshold = np.mean(deltas) + 3 * np.std(deltas)
        large_changes = deltas > threshold
        
        # Collision = multiple features change simultaneously
        simultaneous_changes = np.sum(large_changes, axis=1)
        collisions = simultaneous_changes > 2
        
        return np.mean(collisions)
    
    def _default_patterns(self) -> VisualPatterns:
        """Return default patterns when insufficient data."""
        return VisualPatterns(
            entity_count=1,
            motion_type="unknown",
            spatial_layout="unknown",
            collision_frequency=0.0,
            state_change_regions=[],
            motion_periodicity=0.0,
        )


if __name__ == "__main__":
    """Test visual pattern extraction on synthetic data."""
    print("=" * 60)
    print("VISUAL PATTERN EXTRACTOR TEST")
    print("=" * 60)
    
    extractor = VisualPatternExtractor()
    
    # Test 1: Grid-like synchronized motion (like Space Invaders)
    print("\nTest 1: Grid-like synchronized motion")
    states_grid = []
    for t in range(100):
        state = np.zeros(128)
        # 30 entities moving in sync (horizontal sweep)
        for i in range(30):
            pos = (i * 4) + (t % 20)  # Sweep back and forth
            if pos < 128:
                state[pos] = 1.0
        states_grid.append(state)
    
    patterns = extractor.extract_patterns(states_grid)
    print(patterns.summary())
    
    # Test 2: Random scattered motion
    print("\nTest 2: Random scattered motion")
    states_random = []
    for t in range(100):
        state = np.random.rand(128) * 0.1
        states_random.append(state)
    
    patterns = extractor.extract_patterns(states_random)
    print(patterns.summary())
    
    # Test 3: Bouncing motion (like Pong)
    print("\nTest 3: Bouncing motion (periodic)")
    states_bounce = []
    for t in range(100):
        state = np.zeros(128)
        # Ball bouncing
        ball_pos = int(64 + 30 * np.sin(t * 0.2))
        state[ball_pos] = 1.0
        # Paddles
        state[10] = 1.0
        state[118] = 1.0
        states_bounce.append(state)
    
    patterns = extractor.extract_patterns(states_bounce)
    print(patterns.summary())
    
    print("\n✅ Visual pattern extraction test complete!")
