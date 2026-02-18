"""
Tetris Curriculum Training with Full Basal Ganglia Dreamer
==========================================================

Wraps the existing PortableNNAgent + TetrisAdapter with the full
dreamer stack (DreamerEngine, Amygdala, DreamerTeacher, OptionsLibrary).

The dreamer:
  1. Learns a world model from every real step
  2. Runs Tetris-specific hypotheses (minimize-height, maximize-lines, build-flat)
  3. Generates teaching signals that nudge (not override) action selection
  4. Tracks dreamer_reliance and backs off as the agent improves
  5. Discovers behavioral options and tracks their performance

Usage:
    python run_tetris_dreamer.py --level 2 --episodes 100
    python run_tetris_dreamer.py --level 2 --episodes 50 --no-dreamer  # baseline
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.basal_ganglia.dreamer_engine import DreamerEngine, Hypothesis
from throng4.basal_ganglia.amygdala import Amygdala
from throng4.basal_ganglia.hypothesis_profiler import DreamerTeacher, HypothesisEvolver
from throng4.basal_ganglia.bootstrap_hypotheses import bootstrap_hypotheses
from throng4.basal_ganglia.execution_profiler import ExecutionProfiler
from throng4.llm_policy.openclaw_bridge import OpenClawBridge
from throng4.storage.experiment_db import ExperimentDB
from throng4.storage.telemetry_logger import TelemetryLogger


# ═══════════════════════════════════════════════════════════════
# Curriculum Configuration
# ═══════════════════════════════════════════════════════════════

MAX_PIECES_PER_LEVEL = {
    1: 50,     # O-block only, very easy
    2: 100,    # O + I blocks
    3: 150,    # + T blocks
    4: 200,    # + S, Z blocks
    5: 300,    # All blocks, 8-wide
    6: 400,    # All blocks, 10-wide
    7: 500,    # Standard Tetris
}

EPISODES_PER_LEVEL = {
    1: 10,     # Quick sanity check
    2: 200,   # User request: 200 for 2-6
    3: 200,
    4: 200,
    5: 200,
    6: 200,
    7: 1000,  # User request: 1000 at end
}

# Win condition: mean lines cleared over last 20 episodes
ADVANCE_THRESHOLD_PER_LEVEL = {
    1: 1.0,    # Very low bar for level 1
    2: 2.0,   # Modest start
    3: 5.0,
    4: 10.0,
    5: 15.0,
    6: 20.0,
    7: 40.0,  # Mastery
}


# ═══════════════════════════════════════════════════════════════
# Tetris-specific Hypotheses
# ═══════════════════════════════════════════════════════════════

def create_tetris_hypotheses(n_actions: int) -> List[Hypothesis]:
    """
    Create Tetris-specific strategy hypotheses.

    These operate on the compressed state vector from the dreamer's
    world model — they don't call back into the environment.

    Three strategies:
      1. minimize_height: Prefer low aggregate height
      2. maximize_lines: Prefer actions that clear lines
      3. build_flat: Minimize bumpiness for future clears
    """
    return [
        Hypothesis(
            id=0,
            name="minimize_height",
            action_selector=lambda s: int(np.argmin(s[:max(4, s.size // 4)])),
            description="Prefer placements that minimize aggregate height",
        ),
        Hypothesis(
            id=1,
            name="maximize_lines",
            action_selector=lambda s: int(np.argmax(s[:max(4, s.size // 4)])),
            description="Prefer placements that maximize line-clear potential",
        ),
        Hypothesis(
            id=2,
            name="build_flat",
            action_selector=lambda s: int(
                np.argmin(np.abs(np.diff(s[:max(4, s.size // 4)])))
            ) if s.size >= 4 else 0,
            description="Minimize bumpiness to prepare for future line clears",
        ),
    ]


# ═══════════════════════════════════════════════════════════════
# Dreamer Tetris Runner
# ═══════════════════════════════════════════════════════════════

class DreamerTetrisRunner:
    """
    Runs Tetris training with the full basal ganglia dreamer active.

    The dreamer:
      - Learns the board transition dynamics from every real step
      - Runs 3 hypothesis dreams per episode (when calibrated)
      - Nudges action selection toward dreamer-recommended placements
      - Tracks which strategies work in which contexts
      - Automatically reduces its influence as the agent improves
    """

    def __init__(self, level: int = 2, max_pieces: int = 200,
                 dreamer_enabled: bool = True,
                 tetra_enabled: bool = False,
                 dreamer_state_size: int = 32,
                 dream_steps: int = 20,
                 nudge_strength: float = 0.3,
                 tetra_observation_interval: int = 50,
                 db_path: str = "experiments/experiments.db",
                 game: str = "tetris"):
        """
        Args:
            level: Tetris curriculum level
            max_pieces: Max pieces per episode
            dreamer_enabled: Whether to use the dreamer
            dreamer_state_size: Compressed state dimension for world model
            dream_steps: Lookahead depth per dream
            nudge_strength: Max influence of dreamer on action selection (0-1)
        """
        self.level = level
        self.max_pieces = max_pieces
        self.dreamer_enabled = dreamer_enabled
        self.tetra_enabled = tetra_enabled
        self.dream_steps = dream_steps
        self.nudge_strength = nudge_strength
        self.tetra_observation_interval = tetra_observation_interval
        self.game = game

        # Persistent storage — open once, keep open for the session
        import uuid as _uuid
        self.db = ExperimentDB(db_path)
        self.telemetry = TelemetryLogger()
        self.session_id = str(_uuid.uuid4())[:8]

        # Environment
        self.adapter = TetrisAdapter(level=level, max_pieces=max_pieces)
        self.adapter.reset()

        # Compute feature size
        valid_actions = self.adapter.get_valid_actions()
        if valid_actions:
            sample_features = self.adapter.make_features(valid_actions[0])
            n_features = len(sample_features)
        else:
            n_features = 24  # Fallback

        # Agent
        self.agent = PortableNNAgent(
            n_features=n_features,
            config=AgentConfig(
                n_hidden=128,
                epsilon=0.2,
                gamma=0.95,
                learning_rate=0.005,
            )
        )

        # Dreamer stack
        if dreamer_enabled:
            # Map placement action space to integer for dreamer
            # Max valid placements in Tetris ~= 40 (4 rotations × 10 columns)
            self.dreamer_n_actions = 40  # Upper bound on placement count
            self.dreamer = DreamerEngine(
                n_hypotheses=3,
                network_size='micro',
                state_size=dreamer_state_size,
                n_actions=self.dreamer_n_actions,
                dream_interval=1,  # Teacher will adjust dynamically
            )
            self.amygdala = Amygdala()
            self.teacher = DreamerTeacher(
                n_actions=self.dreamer_n_actions,
                state_dim=min(16, dreamer_state_size),
            )
            # Bootstrap hypotheses: tier-1 priors always, tier-2 from DB if
            # available, tier-3 from Tetra if bridge is set later.
            self.hypotheses = bootstrap_hypotheses(
                game=self.game,
                n_actions=self.dreamer_n_actions,
                db=self.db,
                bridge=None,  # Tetra bridge not ready yet at init time
                verbose=True,
            )
            # If bootstrap returned fewer than 3, pad with Tetris-specific ones
            if len(self.hypotheses) < 3:
                self.hypotheses = create_tetris_hypotheses(self.dreamer_n_actions)
            self.evolver = HypothesisEvolver(n_actions=self.dreamer_n_actions)
            # ExecutionProfiler: sub-ganglia for execution-level learning.
            # Uses the same state_dim as the dreamer teacher.
            self.exec_profiler = ExecutionProfiler(
                state_dim=min(16, dreamer_state_size)
            )
        else:
            self.dreamer = None
            self.amygdala = None
            self.teacher = None
            self.hypotheses = None
            self.evolver = None
            self.exec_profiler = None
        
        # Tetra bridge
        if tetra_enabled:
            self.bridge = OpenClawBridge(
                game=f"Tetris_Level_{level}",
                agent_id="main"  # Use the default 'main' agent
            )
            # Check if gateway is available
            if not self.bridge.is_available:
                print("⚠️  OpenClaw gateway not available — Tetra disabled")
                self.bridge = None
                self.tetra_enabled = False
        else:
            self.bridge = None
        
        # Tracking
        self.episode_results: List[Dict[str, Any]] = []
        self.dreamer_events: List[Dict[str, Any]] = []
        self.tetra_responses: List[Dict[str, Any]] = []
        self._saved_states: List[Dict[str, Any]] = []  # Hippocampus: saved board states
        self._last_state: Optional[np.ndarray] = None
        self._step_count = 0

    def run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run a single episode with dreamer integration."""
        self.agent.reset_episode()
        self.adapter = TetrisAdapter(level=self.level, max_pieces=self.max_pieces)
        state = self.adapter.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        dreamer_nudges = 0
        dreamer_agrees = 0
        exec_nudges = 0
        # Track current best hypothesis name for ExecutionProfiler goal labelling
        _current_goal = "explore"

        # Clear stale signals from previous episode so get_best_action()
        # doesn't return recommendations for a different board state.
        if self.dreamer_enabled and self.teacher:
            self.teacher.clear_episode_signals()

        while not done:
            valid_actions = self.adapter.get_valid_actions()
            if not valid_actions:
                break

            # ── Agent selects action (with optional dreamer nudge) ──
            dreamer_bias = None
            if self.dreamer_enabled and self.teacher:
                dreamer_bias = self._get_dreamer_bias(
                    state, valid_actions, episode_num
                )

            action = self._select_action_with_nudge(
                valid_actions, dreamer_bias
            )

            # ── ExecutionProfiler nudge (second soft layer) ──
            exec_nudge = None
            if self.exec_profiler and self.dreamer_enabled:
                # Candidate methods = action indices as strings
                candidate_methods = [str(i) for i in range(len(valid_actions))]
                exec_nudge = self.exec_profiler.get_execution_nudge(
                    goal=_current_goal,
                    state=(
                        state if isinstance(state, np.ndarray)
                        else np.zeros(self.exec_profiler.state_dim)
                    ),
                    candidate_methods=candidate_methods,
                )
                if exec_nudge.strength > 0.01:
                    exec_nudges += 1
                    # Re-score with exec nudge if it recommends a different action
                    exec_idx = exec_nudge.method_index
                    if exec_idx < len(valid_actions):
                        exec_action = valid_actions[exec_idx]
                        if exec_action != action:
                            # Only switch if exec nudge is confident enough
                            # to overcome the main policy's preference
                            action = exec_action

            # Determine action index for dreamer world model
            action_idx = valid_actions.index(action) if action in valid_actions else 0
            action_idx = min(action_idx, self.dreamer_n_actions - 1) if self.dreamer_enabled else 0

            # ── Step ──
            next_state, reward, done, info = self.adapter.step(action)
            features = self.adapter.make_features(action)
            self.agent.record_step(features, reward)

            # ── Feed dreamer world model ──
            if self.dreamer_enabled and self.dreamer:
                compressed_state = self.dreamer._ensure_compressed(
                    state if isinstance(state, np.ndarray)
                    else np.zeros(self.dreamer.state_size)
                )
                compressed_next = self.dreamer._ensure_compressed(
                    next_state if isinstance(next_state, np.ndarray)
                    else np.zeros(self.dreamer.state_size)
                )
                self.dreamer.learn(compressed_state, action_idx,
                                   compressed_next, reward)
                self._last_state = compressed_state

                # Track agreement
                if dreamer_bias is not None:
                    dreamer_nudges += 1
                    dreamer_action = dreamer_bias.get('action')
                    if dreamer_action is not None:
                        actual_idx = valid_actions.index(action)
                        if actual_idx == dreamer_action:
                            dreamer_agrees += 1
                        self.teacher.record_policy_action(
                            compressed_state, actual_idx,
                            dreamer_action, reward
                        )

                # ── ExecutionProfiler: record this attempt ──
                if self.exec_profiler:
                    actual_idx = valid_actions.index(action) if action in valid_actions else 0
                    # Goal = current best hypothesis (what the dreamer is trying to do)
                    # Method = which action index was chosen
                    _current_goal = getattr(
                        self, '_last_best_hypothesis', 'explore'
                    )
                    self.exec_profiler.record_attempt(
                        goal=_current_goal,
                        method=str(actual_idx),
                        state=compressed_state,
                        success=reward > 0,
                        outcome_value=float(reward),
                    )

            episode_reward += reward
            steps += 1
            state = next_state
            self._step_count += 1

        # ── End episode ──
        info = self.adapter.get_info()
        lines = info['lines_cleared']
        self.agent.end_episode(final_score=float(lines))

        # ── Dream cycle (post-episode) ──
        dream_info = {}
        if (self.dreamer_enabled and self.dreamer
                and self.dreamer.is_calibrated
                and self._last_state is not None):
            dream_info = self._run_dream_cycle(episode_num)

        result = {
            'episode': episode_num,
            'lines': lines,
            'pieces': info['pieces_placed'],
            'reward': episode_reward,
            'steps': steps,
            'dreamer_nudges': dreamer_nudges,
            'dreamer_agrees': dreamer_agrees,
            'exec_nudges': exec_nudges,
            **dream_info,
        }

        self.episode_results.append(result)

        # ── Persist to ExperimentDB ──────────────────────────────────────────
        # Grab board features for DB (max_height, holes, bumpiness)
        try:
            _bf = self.adapter._compute_board_features(self.adapter.env.board)
            _max_h  = _bf['max_height']
            _holes  = _bf['holes']
            _bump   = float(_bf['bumpiness'])
        except Exception:
            _max_h = _holes = 0; _bump = 0.0

        # Hypothesis performance snapshot
        _hyp_perf = {}
        if self.hypotheses:
            for h in self.hypotheses:
                _hyp_perf[h.name] = {
                    'win_rate': round(getattr(h, 'win_rate', 0.0), 3),
                    'avg_reward': round(getattr(h, 'avg_reward', 0.0), 3),
                }

        self.db.log_episode(
            game=self.game,
            level=self.level,
            episode_num=episode_num,
            score=episode_reward,
            lines_cleared=lines,
            pieces_placed=info['pieces_placed'],
            max_height=_max_h,
            holes=_holes,
            bumpiness=_bump,
            outcome_tags={
                'dreamer_nudges': dreamer_nudges,
                'exec_nudges': exec_nudges,
                'best_hypothesis': dream_info.get('best_hypothesis', ''),
            },
            hypothesis_performance=_hyp_perf if _hyp_perf else None,
            session_id=self.session_id,
        )

        # Compact JSONL trace (greppable, no SQLite overhead)
        self.telemetry.log_episode({
            'game': self.game,
            'level': self.level,
            'episode': episode_num,
            'session': self.session_id,
            'lines': lines,
            'score': round(episode_reward, 2),
            'pieces': info['pieces_placed'],
            'max_height': _max_h,
            'holes': _holes,
            'dreamer_nudges': dreamer_nudges,
            'exec_nudges': exec_nudges,
            'best_hypothesis': dream_info.get('best_hypothesis', ''),
        })
        
        # ── Send observation to Tetra (every N episodes) ──
        if (self.tetra_enabled and self.bridge
                and (episode_num + 1) % self.tetra_observation_interval == 0):
            self._send_tetra_observation(episode_num, result)
        
        return result

    def _get_dreamer_bias(self, state: np.ndarray,
                          valid_actions: list,
                          episode_num: int) -> Optional[Dict]:
        """Get dreamer's action recommendation.

        Returns a bias dict with 'action' as an index into valid_actions
        (not the dreamer's abstract 40-action space) and a nudge strength.
        """
        rec = self.teacher.get_best_action(
            state if isinstance(state, np.ndarray)
            else np.zeros(16)
        )
        if rec is None:
            return None

        dreamer_action_idx, confidence = rec
        if confidence < 0.05:
            return None

        # Scale nudge by reliance (more nudge early, less as policy improves)
        reliance = self.teacher.dreamer_reliance
        effective_nudge = self.nudge_strength * confidence * reliance

        # Map dreamer's abstract action index to the closest valid_actions index.
        # The dreamer operates in a fixed 40-action space; valid_actions is the
        # actual list of placements available this step (variable length).
        # We use modulo as a rough mapping — imperfect but consistent.
        # The nudge is small (0.3 * conf * reliance) so a wrong mapping just
        # adds a small random bias rather than catastrophically misdirecting.
        n_valid = len(valid_actions)
        mapped_action = dreamer_action_idx % n_valid if n_valid > 0 else 0

        return {
            'action': mapped_action,
            'confidence': confidence,
            'nudge': effective_nudge,
        }

    def _select_action_with_nudge(self, valid_actions: list,
                                   dreamer_bias: Optional[Dict]) -> Any:
        """Select action from agent, optionally nudged by dreamer."""
        # Epsilon-greedy exploration (handled by agent internally)
        if np.random.rand() < self.agent.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]

        # Score all actions
        scores = []
        for i, action in enumerate(valid_actions):
            features = self.adapter.make_features(action)
            score = float(self.agent.forward(features))

            # Apply dreamer nudge (additive bias to recommended action)
            if (dreamer_bias is not None
                    and i == dreamer_bias['action']
                    and dreamer_bias['nudge'] > 0.01):
                score += dreamer_bias['nudge']

            scores.append(score)

        best_idx = int(np.argmax(scores))
        return valid_actions[best_idx]

    def _run_dream_cycle(self, episode_num: int) -> Dict:
        """Run dream cycle post-episode."""
        dream_state = self._last_state
        if dream_state is None:
            return {}

        # Run dreams
        dream_results = self.dreamer.dream(
            dream_state, self.hypotheses, n_steps=self.dream_steps
        )

        if not dream_results:
            return {}

        # Feed teacher
        signals = self.teacher.process_dream_results(
            dream_results, dream_state, self._step_count
        )

        # Dynamically adjust dream interval
        self.dreamer.dream_interval = (
            self.teacher.recommended_dream_interval
        )

        # Amygdala check
        danger = self.amygdala.assess_danger(
            dream_results, current_step=self._step_count
        )
        danger_level = danger.level.value if danger else "safe"

        if self.amygdala.should_override(danger, self._step_count):
            self.amygdala.record_override(self._step_count)
            self.dreamer_events.append({
                'episode': episode_num,
                'event': 'amygdala_override',
                'danger': danger_level,
            })

        # Evolver: check each hypothesis profile for mutation
        if self.evolver and self.teacher.profiles:
            for i, hyp in enumerate(self.hypotheses):
                profile = self.teacher.profiles.get(hyp.id)
                if profile is None:
                    continue
                new_hyp = self.evolver.maybe_evolve(
                    profile, hyp, self._step_count
                )
                if new_hyp is not None:
                    self.hypotheses[i] = new_hyp
                    # Reset the profile so the new variant starts fresh
                    del self.teacher.profiles[hyp.id]
                    print(f"  ⚙️  Evolved H{hyp.id}: {hyp.name} → {new_hyp.name} "
                          f"(was win={profile.win_rate:.0%})")
                    # Persist both old (retired) and new (candidate) to DB
                    self.db.upsert_hypothesis(
                        name=hyp.name, game=self.game, level=self.level,
                        confidence=profile.win_rate,
                        win_rate=profile.win_rate,
                        evidence_count=profile.total_evaluations,
                        status='retired',
                        metadata={'evolved_into': new_hyp.name},
                    )
                    self.db.upsert_hypothesis(
                        name=new_hyp.name, game=self.game, level=self.level,
                        confidence=0.5, win_rate=0.0, evidence_count=0,
                        status='candidate',
                        metadata={'evolved_from': hyp.name},
                    )

        return {
            'dreamer_reliance': self.teacher.dreamer_reliance,
            'dream_interval': self.dreamer.dream_interval,
            'active_options': len(self.teacher.options.active_options),
            'danger_level': danger_level,
            'best_hypothesis': dream_results[0].hypothesis_name,
            'best_h_reward': dream_results[0].total_predicted_reward,
            'teaching_signals': len(signals),
            'mutations': self.evolver.mutation_count if self.evolver else 0,
        }
    
    def _send_tetra_observation(self, episode_num: int, result: Dict[str, Any]):
        """Send rich observation to Tetra with board states, failure analysis, and specific questions."""
        if not self.bridge:
            return
        
        # Build rich context
        recent = self.episode_results[-10:]
        trend = self._compute_trend(recent)
        failure_analysis = self._analyze_failures(recent)
        board_snapshot = self._get_board_snapshot()
        hypothesis_detail = self._get_hypothesis_detail()
        
        # Construct a detailed, actionable message for Tetra
        observation_parts = [
            f"=== Tetris Level {self.level} - Episode {episode_num} Report ===",
            f"",
            f"CURRENT STATE:",
            f"  Lines cleared this episode: {result['lines']}",
            f"  Pieces placed: {result['pieces']}",
            f"  Board: {self.adapter.board_width}x{self.adapter.env.height}",
            f"  Pieces available: {', '.join(self.adapter.env.piece_types)}",
            f"",
            f"PERFORMANCE TREND (last 10 episodes):",
            f"  Mean lines: {trend['mean']:.1f}",
            f"  Best: {trend['best']}  Worst: {trend['worst']}",
            f"  Trend: {trend['direction']}",
            f"  Consistency: {trend['consistency']}",
            f"",
        ]
        
        if board_snapshot:
            observation_parts.extend([
                f"BOARD STATE (end of episode):",
                board_snapshot,
                f"",
            ])
        
        if failure_analysis:
            observation_parts.extend([
                f"FAILURE ANALYSIS:",
                failure_analysis,
                f"",
            ])
        
        observation_parts.extend([
            f"HYPOTHESIS PERFORMANCE:",
            hypothesis_detail,
            f"",
            f"DREAMER STATUS:",
            f"  Reliance: {result.get('dreamer_reliance', 'N/A')}",
            f"  Dream interval: {result.get('dream_interval', 'N/A')}",
            f"  Active options: {result.get('active_options', 0)}",
            f"  Best hypothesis: {result.get('best_hypothesis', 'N/A')}",
            f"",
            f"QUESTIONS - Please respond with SPECIFIC actionable advice:",
            f"  1. Given the board state and failure pattern, what SPECIFIC",
            f"     placement strategy should we try? (e.g. 'prioritize filling",
            f"     column 3 gaps before building height elsewhere')",
            f"  2. Which hypothesis should we STOP using and why?",
            f"  3. Suggest a NEW micro-hypothesis in this format:",
            f"     'When [board condition], do [specific placement action]'",
        ])
        
        observation = "\n".join(observation_parts)
        
        # Save interesting state for replay
        self._save_interesting_state(episode_num, 'tetra_checkpoint')
        
        # Rich structured context
        context = {
            'level': self.level,
            'board_width': self.adapter.board_width,
            'board_height': self.adapter.env.height,
            'pieces_available': self.adapter.env.piece_types,
            'episode': episode_num,
            'lines_cleared': result['lines'],
            'pieces_placed': result['pieces'],
            'dreamer_reliance': result.get('dreamer_reliance', 0),
            'active_options': result.get('active_options', 0),
            'hypothesis_profiles': self._get_hypothesis_stats(),
            'trend': trend,
            'failure_modes': self._get_failure_modes(recent),
            'saved_states_count': len(self._saved_states),
            'recent_performance': [r['lines'] for r in recent],
        }
        
        try:
            response = self.bridge.send_observation(
                episode=episode_num,
                observation=observation,
                context=context
            )
            
            self.tetra_responses.append({
                'episode': episode_num,
                'response': response.raw[:500] if response.raw else '',
                'hypotheses': response.hypotheses,
                'success': response.success,
            })
            
            # Log Tetra's response
            if response.raw:
                preview = response.raw[:200].replace('\n', ' ')
                print(f"\n  Tetra: {preview}...")
            
            if response.hypotheses:
                print(f"  Tetra suggested {len(response.hypotheses)} hypotheses")
                self._add_tetra_hypotheses(response.hypotheses)
        
        except Exception as e:
            print(f"\n  Tetra observation failed: {e}")

    
    def _format_hypothesis_profiles(self) -> str:
        """Create human-readable summary of hypothesis performance."""
        if not self.teacher or not self.teacher.profiles:
            return "No profiles yet"
        
        lines = []
        for profile in self.teacher.profiles.values():
            lines.append(
                f"{profile.hypothesis_name}={profile.win_rate:.0%}"
            )
        return ", ".join(lines)
    
    def _get_hypothesis_stats(self) -> Dict[str, Dict[str, float]]:
        """Get detailed hypothesis statistics for Tetra."""
        if not self.teacher or not self.teacher.profiles:
            return {}
        
        stats = {}
        for profile in self.teacher.profiles.values():
            stats[profile.hypothesis_name] = {
                'win_rate': profile.win_rate,
                'avg_reward': profile.avg_reward,
                'specialization': profile.specialization_score,
                'total_evaluations': profile.total_evaluations,
            }
        return stats
    
    def _add_tetra_hypotheses(self, hypotheses: List[Dict[str, Any]]):
        """Add new hypotheses suggested by Tetra to the dreamer."""
        if not self.dreamer or not self.hypotheses:
            return
        for h_data in hypotheses:
            print(f"    -> {h_data.get('name', 'unnamed')}: {h_data.get('description', 'no description')}")
    
    def _compute_trend(self, recent: List[Dict]) -> Dict[str, Any]:
        """Compute performance trend from recent episodes."""
        if not recent:
            return {'mean': 0, 'best': 0, 'worst': 0, 'direction': 'N/A', 'consistency': 'N/A'}
        
        lines = [r['lines'] for r in recent]
        mean = sum(lines) / len(lines)
        
        if len(lines) >= 4:
            first_half = sum(lines[:len(lines)//2]) / (len(lines)//2)
            second_half = sum(lines[len(lines)//2:]) / (len(lines) - len(lines)//2)
            if first_half > 0 and second_half > first_half * 1.2:
                direction = 'IMPROVING (+{:.0f}%)'.format((second_half / first_half - 1) * 100)
            elif first_half > 0 and second_half < first_half * 0.8:
                direction = 'DECLINING ({:.0f}%)'.format((second_half / first_half - 1) * 100)
            else:
                direction = 'STABLE'
        else:
            direction = 'TOO FEW EPISODES'
        
        if len(lines) >= 3 and mean > 0:
            std = (sum((x - mean)**2 for x in lines) / len(lines)) ** 0.5
            cv = std / mean
            consistency = 'HIGH' if cv < 0.3 else 'MEDIUM' if cv < 0.7 else 'LOW (erratic)'
        else:
            consistency = 'N/A'
        
        return {
            'mean': mean, 'best': max(lines), 'worst': min(lines),
            'direction': direction, 'consistency': consistency,
        }
    
    def _analyze_failures(self, recent: List[Dict]) -> str:
        """Analyze common failure patterns from recent episodes."""
        if not recent:
            return "No data yet"
        
        lines = [r['lines'] for r in recent]
        pieces = [r['pieces'] for r in recent]
        parts = []
        
        short_eps = sum(1 for p in pieces if p < 10)
        if short_eps > 0:
            parts.append(f"  - {short_eps}/{len(recent)} episodes died within 10 pieces (board fills too fast)")
        
        zero_eps = sum(1 for l in lines if l == 0)
        if zero_eps > 0:
            parts.append(f"  - {zero_eps}/{len(recent)} episodes cleared ZERO lines")
        
        if len(lines) >= 3:
            mean_l = sum(lines) / len(lines)
            best = max(lines)
            if best > mean_l * 3 and mean_l > 0:
                parts.append(f"  - High variance: best={best} but mean={mean_l:.1f} (inconsistent)")
        
        if not parts:
            parts.append("  No obvious failure patterns detected")
        return "\n".join(parts)
    
    def _get_board_snapshot(self) -> str:
        """Get board state summary for Tetra."""
        try:
            board = self.adapter.env.board
            features = self.adapter._compute_board_features(board)
            heights = features['heights']
            max_h = max(heights) if heights else 0
            
            lines = []
            lines.append(f"  Column heights: {heights}")
            lines.append(f"  Aggregate height: {features['agg_height']}")
            lines.append(f"  Holes: {features['holes']}")
            lines.append(f"  Bumpiness: {features['bumpiness']}")
            lines.append(f"  Max height: {features['max_height']}/{self.adapter.env.height}")
            lines.append(f"  Avg row completeness: {features['avg_completeness']:.0%}")
            
            # Visual board top
            if max_h > 0:
                lines.append(f"  Board top (cols 0-{self.adapter.board_width - 1}):")
                start_row = max(0, self.adapter.env.height - max_h - 1)
                for r in range(start_row, min(start_row + 6, self.adapter.env.height)):
                    row_str = '  |'
                    for c in range(self.adapter.board_width):
                        row_str += '#' if board[r][c] is not None else '.'
                    row_str += '|'
                    lines.append(row_str)
            
            return "\n".join(lines)
        except Exception:
            return ""
    
    def _get_hypothesis_detail(self) -> str:
        """Get detailed hypothesis performance for Tetra."""
        if not self.teacher or not self.teacher.profiles:
            return "  No hypothesis profiles yet"
        
        lines = []
        for profile in self.teacher.profiles.values():
            lines.append(
                f"  {profile.hypothesis_name}: "
                f"win={profile.win_rate:.0%}, "
                f"avg_reward={profile.avg_reward:.2f}, "
                f"spec={profile.specialization_score:.2f}, "
                f"evals={profile.total_evaluations}"
            )
        return "\n".join(lines)
    
    def _get_failure_modes(self, recent: List[Dict]) -> List[str]:
        """Extract structured failure modes for context."""
        modes = []
        if not recent:
            return modes
        lines = [r['lines'] for r in recent]
        pieces = [r['pieces'] for r in recent]
        
        zero_rate = sum(1 for l in lines if l == 0) / len(lines)
        if zero_rate > 0.3:
            modes.append('frequent_zero_lines')
        
        short_rate = sum(1 for p in pieces if p < 10) / len(pieces)
        if short_rate > 0.3:
            modes.append('early_death')
        
        mean_l = sum(lines) / len(lines) if lines else 0
        if max(lines) > 3 * mean_l and mean_l > 0:
            modes.append('high_variance')
        
        return modes
    
    def _save_interesting_state(self, episode_num: int, reason: str):
        """Save a board state for future replay (hippocampus foundation)."""
        try:
            board_copy = [row[:] for row in self.adapter.env.board]
            features = self.adapter._compute_board_features(board_copy)
            
            self._saved_states.append({
                'episode': episode_num,
                'reason': reason,
                'level': self.level,
                'features': features,
                'pieces_placed': self.adapter.env.pieces_placed,
                'lines_cleared': self.adapter.env.lines_cleared,
                'timestamp': time.time(),
            })
            
            # Keep buffer bounded
            if len(self._saved_states) > 100:
                self._saved_states = self._saved_states[-100:]
        except Exception:
            pass

    def run_training(self, n_episodes: int = 100,
                     verbose: bool = True) -> Dict[str, Any]:
        """Run full training session."""
        level_config = TetrisCurriculumEnv.LEVELS[self.level]

        print(f"\n{'='*70}")
        print(f"TETRIS DREAMER TRAINING — Level {self.level}: {level_config['name']}")
        print(f"{'='*70}")
        print(f"  Board: {level_config['width']}×{level_config['height']}")
        print(f"  Pieces: {level_config['pieces']}")
        print(f"  Dreamer: {'ENABLED' if self.dreamer_enabled else 'DISABLED (baseline)'}")
        if self.dreamer_enabled:
            print(f"  Dream steps: {self.dream_steps}")
            print(f"  Nudge strength: {self.nudge_strength}")
        print(f"  Episodes: {n_episodes}")
        print(f"{'='*70}\n")

        t0 = time.time()

        for ep in range(n_episodes):
            result = self.run_episode(ep)

            if verbose and (ep + 1) % 10 == 0:
                recent = self.episode_results[-10:]
                avg_lines = np.mean([r['lines'] for r in recent])
                avg_reward = np.mean([r['reward'] for r in recent])
                max_lines = max(r['lines'] for r in recent)

                line = (
                    f"  Ep {ep+1:3d}/{n_episodes}: "
                    f"Lines={avg_lines:.2f} (max={max_lines}), "
                    f"R={avg_reward:.2f}"
                )

                if self.dreamer_enabled and 'dreamer_reliance' in result:
                    line += (
                        f" | dream: reliance={result['dreamer_reliance']:.0%}"
                        f" int={result.get('dream_interval', '?')}"
                        f" opts={result.get('active_options', 0)}"
                        f" best_h={result.get('best_hypothesis', '?')}"
                    )

                print(line)

        elapsed = time.time() - t0

        # ── Summary ──
        all_lines = [r['lines'] for r in self.episode_results]

        summary = {
            'level': self.level,
            'level_name': level_config['name'],
            'dreamer_enabled': self.dreamer_enabled,
            'episodes': n_episodes,
            'elapsed_s': elapsed,
            'mean_lines': float(np.mean(all_lines)),
            'std_lines': float(np.std(all_lines)),
            'max_lines': int(np.max(all_lines)),
            'mean_reward': float(np.mean([r['reward'] for r in self.episode_results])),
            # Last 20 episodes
            'final_mean_lines': float(np.mean(all_lines[-20:])),
            'final_max_lines': int(np.max(all_lines[-20:])),
        }

        if self.dreamer_enabled and self.teacher:
            summary['dreamer'] = {
                'final_reliance': self.teacher.dreamer_reliance,
                'dreamer_needed': self.teacher.dreamer_is_needed,
                'signals_generated': self.teacher._total_signals_generated,
                'signals_followed': self.teacher._total_signals_followed,
                'active_options': len(self.teacher.options.active_options),
                'promoted_options': len(self.teacher.options.promoted_options),
                'hypothesis_profiles': {
                    p.hypothesis_name: {
                        'win_rate': p.win_rate,
                        'avg_reward': p.avg_reward,
                        'specialization': p.specialization_score,
                    }
                    for p in self.teacher.profiles.values()
                },
                'amygdala_overrides': len(self.dreamer_events),
            }

        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE ({elapsed:.1f}s)")
        print(f"{'='*70}")
        print(f"  Mean lines: {summary['mean_lines']:.2f} ± {summary['std_lines']:.2f}")
        print(f"  Max lines: {summary['max_lines']}")
        print(f"  Final 20 avg: {summary['final_mean_lines']:.2f}")

        if self.dreamer_enabled and self.teacher:
            d = summary['dreamer']
            print(f"\n  Dreamer Stats:")
            print(f"    Reliance: {d['final_reliance']:.0%}")
            print(f"    Still needed: {d['dreamer_needed']}")
            print(f"    Signals: {d['signals_generated']} generated, "
                  f"{d['signals_followed']} followed")
            print(f"    Options: {d['active_options']} active, "
                  f"{d['promoted_options']} promoted")
            if d['hypothesis_profiles']:
                print(f"    Hypothesis profiles:")
                for name, stats in d['hypothesis_profiles'].items():
                    print(f"      {name}: win={stats['win_rate']:.0%}, "
                          f"r={stats['avg_reward']:+.2f}, "
                          f"spec={stats['specialization']:.2f}")
            if d['amygdala_overrides'] > 0:
                print(f"    Amygdala overrides: {d['amygdala_overrides']}")

        print(f"{'='*70}\n")

        return summary

    def run_curriculum(self, start_level: int = 1, max_level: int = 7,
                       auto_advance: bool = True, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run full curriculum progression with auto-advance.
        
        The agent trains on each level until it meets the threshold,
        then advances to the next level. The dreamer stays active throughout,
        learning and adapting to each new level's dynamics.
        """
        all_stats = []
        
        print(f"\n{'='*70}")
        print(f"TETRIS CURRICULUM WITH DREAMER")
        print(f"{'='*70}")
        print(f"  Levels: {start_level} → {max_level}")
        print(f"  Dreamer: {'ENABLED' if self.dreamer_enabled else 'DISABLED'}")
        print(f"  Auto-advance: {auto_advance}")
        print(f"{'='*70}\n")
        
        for level in range(start_level, max_level + 1):
            # Update runner to new level
            self.level = level
            self.max_pieces = MAX_PIECES_PER_LEVEL.get(level, 200)
            episodes = EPISODES_PER_LEVEL.get(level, 100)
            
            # Reset episode tracking for this level
            self.episode_results = []
            
            # Train on this level
            stats = self.run_training(n_episodes=episodes, verbose=verbose)
            all_stats.append(stats)
            
            # Check if should advance (informational only — always continues)
            if auto_advance and level < max_level:
                threshold = ADVANCE_THRESHOLD_PER_LEVEL.get(level, 2.0)
                if stats['mean_lines'] >= threshold:
                    print(f"\n  ✅ L{level} threshold met ({stats['mean_lines']:.2f} >= {threshold:.2f}) — advancing to L{level+1}\n")
                else:
                    print(f"\n  ⚠️  L{level} threshold not met ({stats['mean_lines']:.2f} < {threshold:.2f}) — advancing anyway\n")
        
        # Final summary
        print(f"\n{'='*70}")
        print(f"CURRICULUM COMPLETE")
        print(f"{'='*70}")
        print(f"  Levels completed: {len(all_stats)}")
        print(f"  Final level: {all_stats[-1]['level']}")
        print(f"\n  Level-by-level performance:")
        print(f"  {'Level':<8} {'Episodes':<12} {'Mean Lines':<15} {'Max Lines':<12}")
        print(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12}")
        for s in all_stats:
            print(f"  {s['level']:<8} {s['episodes']:<12} {s['mean_lines']:<15.2f} {s['max_lines']:<12}")
        print(f"{'='*70}\n")
        
        return all_stats


def run_comparison(level: int = 2, episodes: int = 50):
    """Run dreamer vs baseline comparison."""
    print("\n" + "=" * 70)
    print("DREAMER vs BASELINE COMPARISON")
    print("=" * 70)

    # Baseline (no dreamer)
    print("\n▸ Running BASELINE (no dreamer)...")
    baseline = DreamerTetrisRunner(
        level=level, dreamer_enabled=False
    )
    baseline_stats = baseline.run_training(n_episodes=episodes)

    # Dreamer
    print("\n▸ Running WITH DREAMER...")
    dreamer = DreamerTetrisRunner(
        level=level, dreamer_enabled=True
    )
    dreamer_stats = dreamer.run_training(n_episodes=episodes)

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Dreamer':>12} {'Delta':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for key, label in [
        ('mean_lines', 'Mean lines'),
        ('max_lines', 'Max lines'),
        ('final_mean_lines', 'Final 20 avg'),
        ('mean_reward', 'Mean reward'),
    ]:
        b = baseline_stats[key]
        d = dreamer_stats[key]
        delta = d - b
        sign = "+" if delta > 0 else ""
        print(f"  {label:<23} {b:>10.2f}   {d:>10.2f}   {sign}{delta:>8.2f}")

    print(f"\n{'='*70}\n")

    return {'baseline': baseline_stats, 'dreamer': dreamer_stats}


def main():
    parser = argparse.ArgumentParser(description="Tetris with Dreamer")
    parser.add_argument('--level', type=int, default=2, help="Curriculum level")
    parser.add_argument('--episodes', type=int, default=50, help="Episodes")
    parser.add_argument('--max-pieces', type=int, default=200, help="Max pieces/ep")
    parser.add_argument('--no-dreamer', action='store_true', help="Disable dreamer")
    parser.add_argument('--tetra', action='store_true', help="Enable Tetra (LLM strategic layer)")
    parser.add_argument('--compare', action='store_true', help="Run A/B comparison")
    parser.add_argument('--curriculum', action='store_true', help="Run full curriculum (levels 1-7)")
    parser.add_argument('--start-level', type=int, default=1, help="Curriculum start level")
    parser.add_argument('--max-level', type=int, default=7, help="Curriculum max level")
    parser.add_argument('--nudge', type=float, default=0.3, help="Nudge strength")
    parser.add_argument('--dream-steps', type=int, default=20, help="Dream lookahead")
    parser.add_argument('--output', type=str, default=None, help="Save results JSON")
    parser.add_argument('--db', type=str, default='experiments/experiments.db',
                        help="Path to ExperimentDB SQLite file")
    args = parser.parse_args()

    if args.compare:
        results = run_comparison(level=args.level, episodes=args.episodes)
    elif args.curriculum:
        runner = DreamerTetrisRunner(
            level=args.start_level,
            dreamer_enabled=not args.no_dreamer,
            tetra_enabled=args.tetra,
            dream_steps=args.dream_steps,
            nudge_strength=args.nudge,
            db_path=args.db,
        )
        results = runner.run_curriculum(
            start_level=args.start_level,
            max_level=args.max_level,
            auto_advance=True,
            verbose=True
        )
        # Print DB summary after curriculum
        stats = runner.db.summary_stats()
        print(f"\nDB: {stats['total_episodes']} episodes, "
              f"{stats['total_hypotheses']} hypotheses persisted")
        for row in stats['episodes_by_level']:
            print(f"  L{row['level']}: {row['n']} eps, "
                  f"avg_lines={row['avg_lines']:.1f}, max={row['max_lines']}")
    else:
        runner = DreamerTetrisRunner(
            level=args.level,
            max_pieces=args.max_pieces,
            dreamer_enabled=not args.no_dreamer,
            tetra_enabled=args.tetra,
            dream_steps=args.dream_steps,
            nudge_strength=args.nudge,
            db_path=args.db,
        )
        results = runner.run_training(
            n_episodes=args.episodes, verbose=True
        )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
