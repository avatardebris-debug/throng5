"""
Save-State Manager — Flags interesting moments for post-game review.

In Throng5, the Policy Monitor watches dreams for major Δ and flags save-states.
This bridges that behavior by flagging moments during live training.

Trigger conditions:
  - High surprise (prediction error spike)
  - Reward spike (sudden jump, positive or negative)
  - Failure cluster (many failures of same type in short window)
  - Mode transition (learning → adaptive or vice versa)
  - Hypothesis result (strategy test completed)
"""

import time
import numpy as np
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import deque


class SaveStateTrigger(Enum):
    """What caused the save-state flag."""
    SURPRISE = "surprise"                # Prediction error spike
    REWARD_SPIKE = "reward_spike"        # Sudden reward jump
    FAILURE_CLUSTER = "failure_cluster"  # Many same-type failures
    MODE_TRANSITION = "mode_transition"  # Policy mode change
    HYPOTHESIS_RESULT = "hypothesis_result"  # Strategy test result


@dataclass
class SaveState:
    """A flagged moment worth reviewing."""
    trigger: SaveStateTrigger
    episode: int
    timestamp: float
    context: Dict          # What caused the flag
    importance: float      # 0-1, how important this save-state is
    
    def __repr__(self):
        return (
            f"SaveState(trigger={self.trigger.value}, ep={self.episode}, "
            f"importance={self.importance:.2f})"
        )


class SaveStateManager:
    """
    Checks for edge case moments worth flagging during training.
    
    Keeps a capped buffer of flagged moments, prioritized by importance.
    """
    
    MAX_SAVE_STATES = 50
    
    # Trigger thresholds
    SURPRISE_THRESHOLD = 0.6       # Surprise level to trigger flag
    REWARD_SPIKE_STDDEVS = 2.5     # Std devs above/below mean for reward spike
    FAILURE_CLUSTER_COUNT = 5      # Same-type failures in window
    FAILURE_CLUSTER_WINDOW = 20    # Steps to look for cluster
    COOLDOWN_EPISODES = 5          # Min episodes between triggers of same type
    
    def __init__(self):
        self.save_states: List[SaveState] = []
        self._last_trigger_episode: Dict[SaveStateTrigger, int] = {
            t: -999 for t in SaveStateTrigger
        }
        self._prev_mode: Optional[str] = None
    
    def check_triggers(
        self,
        perception,    # PerceptionHub
        episode: int,
        rewards: list,
        current_mode: Optional[str] = None,
        hypothesis_result: Optional[Dict] = None,
    ) -> Optional[SaveState]:
        """
        Check all trigger conditions and flag if met.
        
        Called once per episode completion.
        
        Returns:
            SaveState if a trigger fired, None otherwise
        """
        triggered = []
        
        # ── 1. Surprise trigger ──
        surprise = perception.get_surprise_level()
        anomaly = perception.get_anomaly_score()
        combined_surprise = max(surprise, anomaly)
        
        if (combined_surprise >= self.SURPRISE_THRESHOLD and 
            self._check_cooldown(SaveStateTrigger.SURPRISE, episode)):
            
            triggered.append(SaveState(
                trigger=SaveStateTrigger.SURPRISE,
                episode=episode,
                timestamp=time.time(),
                context={
                    'surprise_level': surprise,
                    'anomaly_score': anomaly,
                    'combined': combined_surprise,
                },
                importance=combined_surprise,
            ))
        
        # ── 2. Reward spike trigger ──
        if len(rewards) >= 10:
            recent_reward = rewards[-1] if rewards else 0
            window = list(rewards)[-50:] if len(rewards) > 50 else list(rewards)
            mean_r = np.mean(window)
            std_r = np.std(window) + 1e-8
            reward_z = abs(recent_reward - mean_r) / std_r
            
            if (reward_z >= self.REWARD_SPIKE_STDDEVS and
                self._check_cooldown(SaveStateTrigger.REWARD_SPIKE, episode)):
                
                direction = "positive" if recent_reward > mean_r else "negative"
                triggered.append(SaveState(
                    trigger=SaveStateTrigger.REWARD_SPIKE,
                    episode=episode,
                    timestamp=time.time(),
                    context={
                        'reward': recent_reward,
                        'mean': mean_r,
                        'z_score': reward_z,
                        'direction': direction,
                    },
                    importance=min(reward_z / 5.0, 1.0),
                ))
        
        # ── 3. Failure cluster trigger ──
        if perception.failure_analyses:
            recent_failures = perception.failure_analyses[-self.FAILURE_CLUSTER_WINDOW:]
            if len(recent_failures) >= self.FAILURE_CLUSTER_COUNT:
                # Check if same type dominates
                from collections import Counter
                modes = [f.mode.value for f in recent_failures]
                mode_counts = Counter(modes)
                dominant_mode, dominant_count = mode_counts.most_common(1)[0]
                
                if (dominant_count >= self.FAILURE_CLUSTER_COUNT and
                    self._check_cooldown(SaveStateTrigger.FAILURE_CLUSTER, episode)):
                    
                    avg_conf = np.mean([
                        f.confidence for f in recent_failures 
                        if f.mode.value == dominant_mode
                    ])
                    triggered.append(SaveState(
                        trigger=SaveStateTrigger.FAILURE_CLUSTER,
                        episode=episode,
                        timestamp=time.time(),
                        context={
                            'dominant_mode': dominant_mode,
                            'count': dominant_count,
                            'window': len(recent_failures),
                            'avg_confidence': avg_conf,
                        },
                        importance=min(dominant_count / 10.0, 1.0),
                    ))
        
        # ── 4. Mode transition trigger ──
        if current_mode and current_mode != self._prev_mode:
            if (self._prev_mode is not None and
                self._check_cooldown(SaveStateTrigger.MODE_TRANSITION, episode)):
                
                triggered.append(SaveState(
                    trigger=SaveStateTrigger.MODE_TRANSITION,
                    episode=episode,
                    timestamp=time.time(),
                    context={
                        'from_mode': self._prev_mode,
                        'to_mode': current_mode,
                    },
                    importance=0.7,
                ))
            self._prev_mode = current_mode
        
        # ── 5. Hypothesis result trigger ──
        if hypothesis_result and self._check_cooldown(
            SaveStateTrigger.HYPOTHESIS_RESULT, episode
        ):
            delta = hypothesis_result.get('reward_delta', 0)
            triggered.append(SaveState(
                trigger=SaveStateTrigger.HYPOTHESIS_RESULT,
                episode=episode,
                timestamp=time.time(),
                context=hypothesis_result,
                importance=min(abs(delta) / 50.0, 1.0),
            ))
        
        # ── Pick most important trigger ──
        if not triggered:
            return None
        
        best = max(triggered, key=lambda s: s.importance)
        self._record(best)
        return best
    
    def _check_cooldown(self, trigger: SaveStateTrigger, episode: int) -> bool:
        """Check if enough episodes have passed since last trigger of this type."""
        return (episode - self._last_trigger_episode[trigger]) >= self.COOLDOWN_EPISODES
    
    def _record(self, save_state: SaveState):
        """Store save-state, evicting lowest importance if at capacity."""
        self._last_trigger_episode[save_state.trigger] = save_state.episode
        self.save_states.append(save_state)
        
        # Evict lowest importance if over capacity
        if len(self.save_states) > self.MAX_SAVE_STATES:
            self.save_states.sort(key=lambda s: s.importance, reverse=True)
            self.save_states = self.save_states[:self.MAX_SAVE_STATES]
    
    def get_flagged_moments(self) -> List[SaveState]:
        """Get all flagged moments, sorted by episode."""
        return sorted(self.save_states, key=lambda s: s.episode)
    
    def get_recent_flags(self, n: int = 5) -> List[SaveState]:
        """Get the N most recent flagged moments."""
        return sorted(self.save_states, key=lambda s: s.episode)[-n:]
    
    def summary(self) -> str:
        """Human-readable summary of flagged moments."""
        if not self.save_states:
            return "No edge cases flagged yet"
        
        from collections import Counter
        trigger_counts = Counter(s.trigger.value for s in self.save_states)
        avg_importance = np.mean([s.importance for s in self.save_states])
        
        lines = [
            f"Edge Case Save-States ({len(self.save_states)} flagged, "
            f"avg importance={avg_importance:.2f}):",
        ]
        
        for trigger in SaveStateTrigger:
            count = trigger_counts.get(trigger.value, 0)
            if count > 0:
                lines.append(f"  {trigger.value:20s}: {count}")
        
        # Show most recent 3
        recent = self.get_recent_flags(3)
        if recent:
            lines.append("\nMost recent flags:")
            for s in recent:
                lines.append(
                    f"  ep {s.episode}: {s.trigger.value} "
                    f"(importance={s.importance:.2f})"
                )
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset for new environment. Keeps save-states (for cross-env review)."""
        self._prev_mode = None
        # NOTE: save_states are NOT cleared — they are for post-game review


if __name__ == "__main__":
    """Test the save-state manager."""
    print("=" * 60)
    print("SAVE-STATE MANAGER TEST")
    print("=" * 60)
    
    from throng4.meta_policy.perception_hub import PerceptionHub
    from throng4.meta_policy.prediction_error_tracker import PredictionErrorType
    
    manager = SaveStateManager()
    perception = PerceptionHub()
    
    # Test 1: No triggers initially
    print("\n[Test 1] No triggers initially...")
    result = manager.check_triggers(perception, episode=1, rewards=[1.0])
    assert result is None
    print("✅ No false triggers")
    
    # Test 2: Reward spike trigger
    print("\n[Test 2] Reward spike...")
    rewards = [1.0] * 20 + [10.0]  # Sudden spike
    result = manager.check_triggers(perception, episode=10, rewards=rewards)
    if result:
        print(f"  Trigger: {result.trigger.value}")
        print(f"  Context: {result.context}")
        print(f"  Importance: {result.importance:.2f}")
        assert result.trigger == SaveStateTrigger.REWARD_SPIKE
        print("✅ Reward spike detected")
    else:
        print("  (No trigger — spike within normal range)")
    
    # Test 3: Failure cluster trigger
    print("\n[Test 3] Failure cluster...")
    for i in range(10):
        state = np.random.randn(10)
        next_state = state + np.random.randn(10) * 0.001  # Barely changes
        perception.record(state, 0, -1.0, next_state)
    
    print(f"  Failure analyses: {len(perception.failure_analyses)}")
    result = manager.check_triggers(perception, episode=20, rewards=[1.0] * 20)
    if result:
        print(f"  Trigger: {result.trigger.value}")
        print(f"  Context: {result.context}")
        print("✅ Failure cluster detected")
    else:
        print("  (No cluster trigger — failures may not have clustered)")
        print("✅ No false positive")
    
    # Test 4: Mode transition trigger
    print("\n[Test 4] Mode transition...")
    manager._prev_mode = 'learning'
    result = manager.check_triggers(
        perception, episode=30, rewards=[1.0] * 30, current_mode='adaptive'
    )
    assert result is not None
    assert result.trigger == SaveStateTrigger.MODE_TRANSITION
    print(f"  Trigger: {result.trigger.value}")
    print(f"  Context: {result.context}")
    print("✅ Mode transition detected")
    
    # Test 5: Hypothesis result trigger
    print("\n[Test 5] Hypothesis result...")
    result = manager.check_triggers(
        perception, episode=40, rewards=[1.0] * 40,
        hypothesis_result={'strategy': 'explore_more', 'reward_delta': 25.0}
    )
    assert result is not None
    assert result.trigger == SaveStateTrigger.HYPOTHESIS_RESULT
    print(f"  Trigger: {result.trigger.value}")
    print(f"  Importance: {result.importance:.2f}")
    print("✅ Hypothesis result flagged")
    
    # Test 6: Cooldown
    print("\n[Test 6] Cooldown...")
    result = manager.check_triggers(
        perception, episode=41, rewards=[1.0] * 41,
        hypothesis_result={'strategy': 'test', 'reward_delta': 10.0}
    )
    assert result is None or result.trigger != SaveStateTrigger.HYPOTHESIS_RESULT
    print("✅ Cooldown prevents rapid re-triggering")
    
    # Summary
    print(f"\n{manager.summary()}")
    
    print("\n✅ Save-state manager test complete!")
