"""
Hypothesis Profiling & Dreamer-as-Teacher — Bridge Step 4, Phase 7.

Three components that prevent the dreamer from becoming a "save-scumming"
crutch and instead use it to build robust, transferable policies:

1. HypothesisProfile
   Tracks per-hypothesis success/failure contexts. Learns WHEN each
   hypothesis works vs. fails. This is the Options framework —
   discovering initiation/termination conditions for sub-policies.

2. OptionsLibrary
   Stores discovered behavioral options (sub-policies with contexts).
   Grows over time as the dreamer discovers strategy structure.
   An option = "use this action pattern when state looks like X,
   stop when state looks like Y."

3. DreamerTeacher
   Training signal pipeline that feeds dreamer insights back
   to the main policy. The dreamer's goal is to make itself
   unnecessary — it teaches the core policy until it's robust
   enough to handle most situations alone.

Design principle:
  The dreamer TEACHES, it doesn't DRIVE.
  After 1000 episodes, the main policy should be good enough
  that the dreamer's contribution is marginal.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque
from enum import Enum


# ══════════════════════════════════════════════════════════════
# 1. HYPOTHESIS PROFILE — Per-hypothesis context tracking
# ══════════════════════════════════════════════════════════════


@dataclass
class ContextSnapshot:
    """A compressed snapshot of the state when a hypothesis was evaluated."""
    state_signature: np.ndarray    # Low-dim representation of the state
    reward: float                  # Reward achieved in this context
    was_best: bool                 # Was this hypothesis the best choice?
    step: int                      # When this happened


class HypothesisProfile:
    """
    Tracks a single hypothesis's performance across different contexts.

    Learns to predict: "in states like X, this hypothesis works well."
    This is the initiation set of an Option in the Options framework.

    Over time, builds a simple context classifier:
      success_centroid = mean of states where this hypothesis ranked #1
      failure_centroid = mean of states where this hypothesis ranked last
      should_activate(state) = similarity to success > similarity to failure
    """

    MAX_HISTORY = 500  # Keep last N context snapshots

    def __init__(self, hypothesis_id: int, hypothesis_name: str,
                 state_dim: int = 16):
        self.hypothesis_id = hypothesis_id
        self.hypothesis_name = hypothesis_name
        self.state_dim = state_dim

        # Context tracking
        self._success_states: deque = deque(maxlen=self.MAX_HISTORY)
        self._failure_states: deque = deque(maxlen=self.MAX_HISTORY)
        self._all_rewards: deque = deque(maxlen=self.MAX_HISTORY)

        # Running centroids (online mean)
        self._success_centroid = np.zeros(state_dim)
        self._failure_centroid = np.zeros(state_dim)
        self._success_count = 0
        self._failure_count = 0

        # Performance stats
        self.total_evaluations = 0
        self.times_ranked_first = 0
        self.times_ranked_last = 0
        self.total_reward_when_best = 0.0

    def record(self, state: np.ndarray, reward: float,
               rank: int, total_hypotheses: int, step: int):
        """
        Record a dream evaluation result for this hypothesis.

        Args:
            state: The state context where this was evaluated
            reward: Total predicted reward from the dream
            rank: 0=best, n-1=worst
            total_hypotheses: How many hypotheses were compared
            step: Current training step
        """
        sig = self._state_signature(state)
        self.total_evaluations += 1
        self._all_rewards.append(reward)

        snapshot = ContextSnapshot(
            state_signature=sig,
            reward=reward,
            was_best=(rank == 0),
            step=step,
        )

        if rank == 0:
            # This hypothesis was the best choice in this context
            self.times_ranked_first += 1
            self.total_reward_when_best += reward
            self._success_states.append(snapshot)
            self._update_centroid('success', sig)
        elif rank == total_hypotheses - 1:
            # This hypothesis was the worst choice
            self.times_ranked_last += 1
            self._failure_states.append(snapshot)
            self._update_centroid('failure', sig)

    def should_activate(self, state: np.ndarray) -> float:
        """
        How likely is this hypothesis to be a good choice in the given state?

        Returns 0-1 activation probability based on similarity to past
        success/failure contexts.

        If we have no context data yet, returns 0.5 (uncertain).
        """
        if self._success_count < 3 and self._failure_count < 3:
            return 0.5  # Not enough data

        sig = self._state_signature(state)

        # Distance to success centroid
        if self._success_count > 0:
            d_success = np.linalg.norm(sig - self._success_centroid)
        else:
            d_success = float('inf')

        # Distance to failure centroid
        if self._failure_count > 0:
            d_failure = np.linalg.norm(sig - self._failure_centroid)
        else:
            d_failure = float('inf')

        # Convert distances to activation via softmax-style
        if d_success == float('inf') and d_failure == float('inf'):
            return 0.5

        # Higher activation when closer to success, farther from failure
        # Use inverse distance ratio
        total = d_success + d_failure
        if total < 1e-8:
            return 0.5

        activation = d_failure / total  # 0-1, higher = closer to success
        return float(np.clip(activation, 0.0, 1.0))

    @property
    def win_rate(self) -> float:
        """Fraction of evaluations where this hypothesis was best."""
        if self.total_evaluations == 0:
            return 0.0
        return self.times_ranked_first / self.total_evaluations

    @property
    def avg_reward(self) -> float:
        if not self._all_rewards:
            return 0.0
        return float(np.mean(self._all_rewards))

    @property
    def specialization_score(self) -> float:
        """
        How specialized is this hypothesis?

        High = works well in specific contexts, poorly in others (specialist)
        Low = works equally well/poorly everywhere (generalist or useless)
        """
        if self.total_evaluations < 10:
            return 0.0
        if not self._all_rewards:
            return 0.0
        reward_std = float(np.std(self._all_rewards))
        # Normalize by mean absolute reward
        mean_abs = float(np.mean(np.abs(list(self._all_rewards)))) + 1e-8
        return min(1.0, reward_std / mean_abs)

    def _state_signature(self, state: np.ndarray) -> np.ndarray:
        """Compress state to fixed-size signature for context matching."""
        flat = state.flatten().astype(np.float32)
        if flat.size == self.state_dim:
            return flat
        elif flat.size > self.state_dim:
            # Downsample
            idx = np.linspace(0, flat.size - 1, self.state_dim, dtype=int)
            return flat[idx]
        else:
            padded = np.zeros(self.state_dim, dtype=np.float32)
            padded[:flat.size] = flat
            return padded

    def _update_centroid(self, kind: str, sig: np.ndarray):
        """Online centroid update (running mean)."""
        if kind == 'success':
            self._success_count += 1
            alpha = 1.0 / self._success_count
            self._success_centroid = (
                (1 - alpha) * self._success_centroid + alpha * sig
            )
        else:
            self._failure_count += 1
            alpha = 1.0 / self._failure_count
            self._failure_centroid = (
                (1 - alpha) * self._failure_centroid + alpha * sig
            )

    def summary(self) -> str:
        return (
            f"H{self.hypothesis_id} ({self.hypothesis_name}): "
            f"win_rate={self.win_rate:.1%}, "
            f"avg_r={self.avg_reward:+.2f}, "
            f"spec={self.specialization_score:.2f}, "
            f"evals={self.total_evaluations}"
        )


# ══════════════════════════════════════════════════════════════
# 2. OPTIONS LIBRARY — Discovered behavioral sub-policies
# ══════════════════════════════════════════════════════════════


class OptionStatus(Enum):
    CANDIDATE = "candidate"      # Newly discovered, not yet validated
    VALIDATED = "validated"      # Proven useful in at least one context
    PROMOTED = "promoted"        # Consistently good, part of main repertoire
    RETIRED = "retired"          # No longer useful (context changed)


@dataclass
class BehavioralOption:
    """
    A discovered behavioral sub-policy (from the Options framework).

    An option is:
      - A policy (action selector) that works in specific contexts
      - Initiation set: states where this option should be considered
      - Termination condition: when to stop using this option
    """
    option_id: int
    name: str
    source_hypothesis_id: int         # Which hypothesis spawned this
    action_bias: np.ndarray           # Learned action preference weights
    initiation_centroid: np.ndarray   # Center of states where this works
    initiation_radius: float          # How far from centroid to activate
    avg_reward: float                 # Expected reward when active
    status: OptionStatus = OptionStatus.CANDIDATE
    times_used: int = 0
    times_successful: int = 0         # Led to positive reward
    discovery_step: int = 0

    @property
    def success_rate(self) -> float:
        if self.times_used == 0:
            return 0.0
        return self.times_successful / self.times_used

    def should_initiate(self, state: np.ndarray) -> float:
        """Probability that this option should activate in the given state."""
        flat = state.flatten()
        if flat.size != self.initiation_centroid.size:
            # Resize
            if flat.size > self.initiation_centroid.size:
                idx = np.linspace(
                    0, flat.size - 1,
                    self.initiation_centroid.size, dtype=int
                )
                flat = flat[idx]
            else:
                padded = np.zeros_like(self.initiation_centroid)
                padded[:flat.size] = flat
                flat = padded

        dist = np.linalg.norm(flat - self.initiation_centroid)
        if self.initiation_radius < 1e-8:
            return 0.0
        # Gaussian activation based on distance
        activation = np.exp(-0.5 * (dist / self.initiation_radius) ** 2)
        return float(activation)

    def select_action(self, state: np.ndarray, n_actions: int) -> int:
        """Select action based on learned bias + state features."""
        bias = self.action_bias[:n_actions]
        # Add small state-dependent noise
        flat = state.flatten()[:n_actions] if state.flatten().size >= n_actions else np.zeros(n_actions)
        logits = bias + flat * 0.1
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (exp_logits.sum() + 1e-8)
        return int(np.random.choice(n_actions, p=probs))

    def summary(self) -> str:
        return (
            f"Option-{self.option_id} ({self.name}): "
            f"status={self.status.value}, "
            f"success={self.success_rate:.1%}, "
            f"used={self.times_used}, "
            f"avg_r={self.avg_reward:+.2f}"
        )


class OptionsLibrary:
    """
    Library of discovered behavioral options.

    Grows over time as the dreamer identifies distinct strategy contexts.
    Options are promoted/retired based on real performance.

    This is the "strategy vocabulary" — the set of behavioral building
    blocks the agent has discovered it can combine.
    """

    MAX_OPTIONS = 20  # Cap to prevent bloat
    PROMOTE_THRESHOLD = 10   # Uses before considering promotion
    RETIRE_THRESHOLD = 0.2   # Success rate below this → retire

    def __init__(self, state_dim: int = 16, n_actions: int = 4):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.options: Dict[int, BehavioralOption] = {}
        self._next_id = 0

    def discover_option(self, profile: HypothesisProfile,
                        current_step: int) -> Optional[BehavioralOption]:
        """
        Try to discover a new option from a hypothesis profile.

        An option is discovered when:
        1. The hypothesis has been evaluated enough times
        2. It has a clear specialization (works in some contexts, not others)
        3. We haven't already discovered this option
        """
        if profile.total_evaluations < 15:
            return None  # Not enough data

        if profile.specialization_score < 0.2:
            return None  # Too generalist or too consistently bad

        if profile.win_rate < 0.1:
            return None  # Almost never the best choice

        # Check if we already have an option from this hypothesis
        for opt in self.options.values():
            if (opt.source_hypothesis_id == profile.hypothesis_id
                    and opt.status != OptionStatus.RETIRED):
                return None  # Already have one

        if len(self.active_options) >= self.MAX_OPTIONS:
            # Try to retire the worst option to make room
            self._retire_worst()
            if len(self.active_options) >= self.MAX_OPTIONS:
                return None

        # Create new option from hypothesis profile
        option = BehavioralOption(
            option_id=self._next_id,
            name=f"{profile.hypothesis_name}_opt",
            source_hypothesis_id=profile.hypothesis_id,
            action_bias=np.zeros(self.n_actions),  # Start neutral
            initiation_centroid=profile._success_centroid.copy(),
            initiation_radius=max(1.0, float(np.std(
                [s.state_signature for s in profile._success_states],
                axis=0
            ).mean()) if len(profile._success_states) > 1 else 1.0),
            avg_reward=profile.avg_reward,
            status=OptionStatus.CANDIDATE,
            discovery_step=current_step,
        )

        self.options[self._next_id] = option
        self._next_id += 1
        return option

    def get_active_options(self, state: np.ndarray,
                           threshold: float = 0.3) -> List[BehavioralOption]:
        """Get all options that should activate in the given state."""
        activated = []
        for opt in self.active_options:
            activation = opt.should_initiate(state)
            if activation >= threshold:
                activated.append(opt)
        return activated

    def record_outcome(self, option_id: int, reward: float):
        """Record the outcome of using an option."""
        if option_id not in self.options:
            return
        opt = self.options[option_id]
        opt.times_used += 1
        if reward > 0:
            opt.times_successful += 1
        # Update running average reward
        opt.avg_reward = (
            opt.avg_reward * 0.95 + reward * 0.05
        )
        # Check for promotion
        if (opt.status == OptionStatus.CANDIDATE
                and opt.times_used >= self.PROMOTE_THRESHOLD):
            if opt.success_rate >= 0.3:
                opt.status = OptionStatus.VALIDATED
            elif opt.success_rate < self.RETIRE_THRESHOLD:
                opt.status = OptionStatus.RETIRED

        if (opt.status == OptionStatus.VALIDATED
                and opt.times_used >= self.PROMOTE_THRESHOLD * 3
                and opt.success_rate >= 0.5):
            opt.status = OptionStatus.PROMOTED

    @property
    def active_options(self) -> List[BehavioralOption]:
        return [o for o in self.options.values()
                if o.status != OptionStatus.RETIRED]

    @property
    def promoted_options(self) -> List[BehavioralOption]:
        return [o for o in self.options.values()
                if o.status == OptionStatus.PROMOTED]

    def _retire_worst(self):
        """Retire the worst-performing active option."""
        active = self.active_options
        if not active:
            return
        worst = min(active, key=lambda o: o.success_rate)
        if worst.success_rate < self.RETIRE_THRESHOLD:
            worst.status = OptionStatus.RETIRED

    def summary(self) -> str:
        lines = [
            f"OptionsLibrary: {len(self.active_options)} active, "
            f"{len(self.promoted_options)} promoted, "
            f"{sum(1 for o in self.options.values() if o.status == OptionStatus.RETIRED)} retired"
        ]
        for opt in self.active_options:
            lines.append(f"  {opt.summary()}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 3. DREAMER TEACHER — Training signal back to main policy
# ══════════════════════════════════════════════════════════════


@dataclass
class TeachingSignal:
    """
    A training signal from the dreamer to the main policy.

    The dreamer produces these to teach the main policy what it learned
    from dream simulations. Over time, this should make the dreamer
    less necessary as the policy absorbs its wisdom.
    """
    action_recommended: int        # Best action from dream results
    confidence: float              # 0-1, how sure the dreamer is
    context_state: np.ndarray      # State where this signal applies
    source: str                    # "dream", "option", "amygdala"
    reward_estimate: float         # Expected reward from this action
    active_option_id: Optional[int] = None  # If triggered by an option


class DreamerTeacher:
    """
    Training signal pipeline: dreamer → main policy.

    Three teaching modes:
    1. Action advisory: "in this state, action X would be better"
       (Used during training to nudge Q-values / policy gradients)

    2. Option injection: "I discovered a behavioral option, learn it"
       (Adds option-derived training examples to the replay buffer)

    3. Curriculum signal: "the dreamer is becoming less useful"
       (Signals that the main policy has absorbed enough and the
       dreamer can reduce its frequency)

    The key metric is dreamer_reliance: how much the main policy
    diverges from dreamer recommendations. When this reaches < 0.2,
    the main policy has learned enough to act autonomously.
    """

    def __init__(self, n_actions: int = 4, state_dim: int = 16,
                 max_signals_per_step: int = 3):
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.max_signals = max_signals_per_step

        # Hypothesis profiles (one per hypothesis)
        self.profiles: Dict[int, HypothesisProfile] = {}

        # Options library
        self.options = OptionsLibrary(state_dim, n_actions)

        # Teaching state
        self._pending_signals: deque = deque(maxlen=100)
        self._total_signals_generated = 0
        self._total_signals_followed = 0   # Main policy agreed
        self._total_signals_rejected = 0   # Main policy disagreed

        # Reliance tracking — seed with neutral prior (0.5) so reliance
        # starts at 0.5 rather than 1.0 before any signals are generated.
        # This breaks the circular dependency where reliance=1.0 prevents
        # signals from being generated, which prevents reliance from updating.
        self._agreement_history: deque = deque(maxlen=200)
        self._agreement_history.extend([True, False])  # neutral 0.5 prior

    def process_dream_results(self, dream_results: list,
                              state: np.ndarray,
                              current_step: int) -> List[TeachingSignal]:
        """
        Process dream results and generate teaching signals.

        This is the main entry point. Call after each dream cycle.

        Args:
            dream_results: List[DreamResult] from DreamerEngine.dream()
            state: The state where dreaming occurred
            current_step: Training step

        Returns:
            List of TeachingSignals for the main policy to learn from
        """
        if not dream_results:
            return []

        signals = []

        # Clear stale signals from the previous dream cycle before generating
        # new ones. This ensures get_best_action() always uses the most recent
        # dream results, not an accumulation of old signals.
        self._pending_signals.clear()

        # 1. Update hypothesis profiles
        self._update_profiles(dream_results, state, current_step)

        # 2. Generate action advisory signal from best hypothesis
        best = dream_results[0]  # Already sorted by reward
        if best.trajectory and best.confidence > 0.1:
            signal = TeachingSignal(
                action_recommended=best.trajectory[0],
                confidence=best.confidence * self._dream_quality(dream_results),
                context_state=state.copy(),
                source="dream",
                reward_estimate=best.avg_predicted_reward,
            )
            signals.append(signal)

        # 3. Check if any options should activate
        active_opts = self.options.get_active_options(state, threshold=0.3)
        for opt in active_opts[:2]:  # Max 2 option signals
            action = opt.select_action(state, self.n_actions)
            signal = TeachingSignal(
                action_recommended=action,
                confidence=opt.success_rate * opt.should_initiate(state),
                context_state=state.copy(),
                source="option",
                reward_estimate=opt.avg_reward,
                active_option_id=opt.option_id,
            )
            signals.append(signal)

        # 4. Try to discover new options
        for profile in self.profiles.values():
            new_opt = self.options.discover_option(profile, current_step)
            if new_opt:
                pass  # Discovery is its own reward; no signal needed

        # Cap signals
        signals = sorted(
            signals, key=lambda s: s.confidence, reverse=True
        )[:self.max_signals]

        self._total_signals_generated += len(signals)
        self._pending_signals.extend(signals)

        return signals

    def clear_episode_signals(self):
        """
        Clear stale pending signals.

        NOTE: This is now a no-op — signals are cleared inside
        process_dream_results() before each new dream cycle, so they
        persist from end-of-episode-N into the start of episode-N+1
        where they are actually used for action selection.

        Kept for API compatibility.
        """
        pass  # Intentionally empty — see process_dream_results()

    def get_best_action(self, state: np.ndarray) -> Optional[Tuple[int, float]]:
        """
        Quick query: what action does the dreamer recommend?

        Returns (action, confidence) or None if no recommendation.
        Used by the main policy for action selection enhancement.
        """
        # Check pending signals for matching context
        if not self._pending_signals:
            return None

        # Find most recent signal with high confidence
        best_signal = None
        best_conf = 0.0

        for signal in reversed(self._pending_signals):
            if signal.confidence > best_conf:
                best_signal = signal
                best_conf = signal.confidence

        if best_signal and best_signal.confidence > 0.05:  # lowered from 0.2
            return best_signal.action_recommended, best_signal.confidence

        return None

    def record_policy_action(self, state: np.ndarray, actual_action: int,
                             dreamer_action: Optional[int],
                             reward: float):
        """
        Record what the main policy actually did vs. what dreamer recommended.

        This tracks agreement and builds the dreamer_reliance metric.
        """
        if dreamer_action is not None:
            agreed = (actual_action == dreamer_action)
            self._agreement_history.append(agreed)
            if agreed:
                self._total_signals_followed += 1
            else:
                self._total_signals_rejected += 1

        # Record outcomes for active options
        for signal in self._pending_signals:
            if signal.active_option_id is not None:
                self.options.record_outcome(
                    signal.active_option_id, reward
                )

    @property
    def dreamer_reliance(self) -> float:
        """
        How much does the main policy rely on the dreamer?

        1.0 = always follows dreamer (policy hasn't learned yet)
        0.0 = never follows dreamer (policy is fully autonomous)

        Goal: drive this toward 0.2 or below.
        """
        if not self._agreement_history:
            return 1.0  # No data → assume full reliance
        return float(np.mean(self._agreement_history))

    @property
    def dreamer_is_needed(self) -> bool:
        """
        Is the dreamer still providing value?

        Returns False when the main policy has absorbed enough
        that the dreamer can reduce frequency.
        """
        # Dreamer is still needed if:
        # 1. Policy still follows >20% of recommendations
        # 2. We haven't generated enough signals to judge
        if self._total_signals_generated < 50:
            return True
        return self.dreamer_reliance > 0.2

    @property
    def recommended_dream_interval(self) -> int:
        """
        How often should the dreamer run?

        Starts frequent (every step), backs off as policy improves.
        """
        reliance = self.dreamer_reliance
        if reliance > 0.7:
            return 1   # Every step — policy needs help
        elif reliance > 0.4:
            return 5   # Every 5 steps — learning
        elif reliance > 0.2:
            return 15  # Every 15 steps — mostly autonomous
        else:
            return 50  # Rarely — policy is strong

    def _update_profiles(self, dream_results: list,
                         state: np.ndarray, step: int):
        """Update hypothesis profiles from dream results."""
        n = len(dream_results)
        for rank, result in enumerate(dream_results):
            hid = result.hypothesis_id
            if hid not in self.profiles:
                self.profiles[hid] = HypothesisProfile(
                    hypothesis_id=hid,
                    hypothesis_name=result.hypothesis_name,
                    state_dim=self.state_dim,
                )
            self.profiles[hid].record(
                state=state,
                reward=result.total_predicted_reward,
                rank=rank,
                total_hypotheses=n,
                step=step,
            )

    def _dream_quality(self, dream_results: list) -> float:
        """
        How useful was this dream?

        High quality = clear winner among hypotheses.
        Low quality = all hypotheses perform similarly (dream was uninformative).
        """
        if len(dream_results) < 2:
            return 1.0
        rewards = [r.total_predicted_reward for r in dream_results]
        spread = max(rewards) - min(rewards)
        mean_abs = np.mean(np.abs(rewards)) + 1e-8
        quality = min(1.0, spread / mean_abs)
        return float(quality)

    def summary(self) -> str:
        lines = [
            "DreamerTeacher:",
            f"  Signals generated: {self._total_signals_generated}",
            f"  Followed: {self._total_signals_followed}, "
            f"Rejected: {self._total_signals_rejected}",
            f"  Dreamer reliance: {self.dreamer_reliance:.1%}",
            f"  Dreamer needed: {self.dreamer_is_needed}",
            f"  Recommended interval: every {self.recommended_dream_interval} steps",
            f"  Hypothesis profiles: {len(self.profiles)}",
        ]
        for p in self.profiles.values():
            lines.append(f"    {p.summary()}")
        lines.append(f"  {self.options.summary()}")
        return "\n".join(lines)


# ======================================================================
# 4. HYPOTHESIS EVOLVER -- Offline autonomous hypothesis mutation
# ======================================================================


class HypothesisEvolver:
    """
    Autonomous hypothesis mutation -- the offline alternative to Tetra.

    When a hypothesis consistently loses dream competitions (win_rate below
    WIN_RATE_THRESHOLD for MIN_EVALS evaluations), it gets replaced with a
    mutated variant that attends to a different part of the state vector or
    uses a different operator.

    Eight mutation strategies cycle through different ways of reading the board:
      0. argmin of first quarter  (minimize height)
      1. argmax of first quarter  (maximize lines)
      2. argmin of abs-diff       (build flat)
      3. argmin of second quarter (mid-board height)
      4. argmax of second quarter (mid-board density)
      5. argmin of last quarter   (column tops)
      6. weighted combo: min(height) + max(density)
      7. noisy argmin (tie-breaking variant)

    This gives the dreamer a self-repair loop: bad hypotheses get replaced,
    good ones survive. No LLM required, works fully offline.

    Usage:
        evolver = HypothesisEvolver(n_actions=40)
        new_hyp = evolver.maybe_evolve(profile, hypothesis, current_step)
        if new_hyp is not None:
            hypotheses[i] = new_hyp   # swap in-place
    """

    MIN_EVALS = 20             # evaluations before considering replacement
    WIN_RATE_THRESHOLD = 0.10  # win rate below this -> candidate
    PATIENCE = 3               # consecutive bad dream cycles before replacing

    def __init__(self, n_actions: int = 40):
        self.n_actions = n_actions
        # Per hypothesis-id: [generation, consecutive_bad_cycles]
        self._state: Dict[int, List[int]] = {}
        self._mutation_log: List[Dict] = []

    def maybe_evolve(self, profile: 'HypothesisProfile',
                     hypothesis,
                     current_step: int):
        """
        Check if this hypothesis should be mutated.

        Returns a new Hypothesis if mutation triggered, else None.
        Caller should swap the returned hypothesis into its slot.
        """
        hid = profile.hypothesis_id

        if hid not in self._state:
            self._state[hid] = [0, 0]  # [generation, bad_cycles]

        gen, bad_cycles = self._state[hid]

        if profile.total_evaluations < self.MIN_EVALS:
            return None

        if profile.win_rate >= self.WIN_RATE_THRESHOLD:
            self._state[hid][1] = 0
            return None

        # Still losing -- increment patience counter
        self._state[hid][1] += 1

        if bad_cycles < self.PATIENCE:
            return None

        # Trigger mutation
        new_gen = gen + 1
        self._state[hid] = [new_gen, 0]

        new_hyp = self._build_mutant(
            hypothesis.id, hypothesis.name, new_gen, current_step
        )

        self._mutation_log.append({
            'step': current_step,
            'hypothesis_id': hid,
            'old_name': hypothesis.name,
            'new_name': new_hyp.name,
            'generation': new_gen,
            'old_win_rate': profile.win_rate,
            'old_evals': profile.total_evaluations,
        })

        return new_hyp

    def _build_mutant(self, hid: int, base_name: str,
                      generation: int, step: int):
        """Build a mutated hypothesis using one of 8 strategies."""
        from throng4.basal_ganglia.dreamer_engine import Hypothesis

        strategy = generation % 8
        n = self.n_actions

        if strategy == 0:
            sel = lambda s, n=n: int(np.argmin(s[:max(4, n // 4)]))
            desc = "minimize first-quarter (height)"

        elif strategy == 1:
            sel = lambda s, n=n: int(np.argmax(s[:max(4, n // 4)]))
            desc = "maximize first-quarter (lines)"

        elif strategy == 2:
            def sel(s, n=n):
                q = s[:max(4, n // 4)]
                d = np.abs(np.diff(q))
                return int(np.argmin(d)) if d.size > 0 else 0
            desc = "minimize bumpiness (flat board)"

        elif strategy == 3:
            def sel(s, n=n):
                lo, hi = max(4, n // 4), max(8, n // 2)
                q = s[lo:hi] if s.size > lo else s
                return int(np.argmin(q)) if q.size > 0 else 0
            desc = "minimize second-quarter (mid-board)"

        elif strategy == 4:
            def sel(s, n=n):
                lo, hi = max(4, n // 4), max(8, n // 2)
                q = s[lo:hi] if s.size > lo else s
                return int(np.argmax(q)) if q.size > 0 else 0
            desc = "maximize second-quarter (mid-density)"

        elif strategy == 5:
            def sel(s, n=n):
                lo = max(0, s.size - max(4, n // 4))
                q = s[lo:]
                return int(np.argmin(q)) if q.size > 0 else 0
            desc = "minimize last-quarter (column tops)"

        elif strategy == 6:
            def sel(s, n=n):
                q = max(4, n // 4)
                height_score = -s[:q]
                density_score = s[q:2*q] if s.size > 2*q else np.zeros(q)
                if density_score.size < q:
                    density_score = np.pad(
                        density_score, (0, q - density_score.size)
                    )
                combined = 0.6 * height_score + 0.4 * density_score
                return int(np.argmax(combined))
            desc = "weighted height+density combo"

        else:  # strategy == 7
            rng_seed = step % 1000
            def sel(s, n=n, seed=rng_seed):
                rng = np.random.default_rng(
                    seed + int(np.sum(np.abs(s)) * 100) % 997
                )
                noise = rng.uniform(
                    -0.05, 0.05, size=min(max(4, n // 4), s.size)
                )
                q = s[:noise.size] + noise
                return int(np.argmin(q))
            desc = "noisy argmin (tie-breaking)"

        # Strip old version suffix from base name
        clean_base = base_name.split('_v')[0]
        return Hypothesis(
            id=hid,
            name=f"{clean_base}_v{generation}",
            action_selector=sel,
            description=f"Gen {generation}: {desc}",
        )

    @property
    def mutation_count(self) -> int:
        return len(self._mutation_log)

    def summary(self) -> str:
        if not self._mutation_log:
            return "HypothesisEvolver: no mutations yet"
        lines = [f"HypothesisEvolver: {self.mutation_count} mutations"]
        for m in self._mutation_log[-5:]:
            lines.append(
                f"  step={m['step']}: {m['old_name']} -> {m['new_name']} "
                f"(was win={m['old_win_rate']:.0%} "
                f"over {m['old_evals']} evals)"
            )
        return "\n".join(lines)
