"""
fast_loop.py — High-speed simulation runner, zero LLM calls.

The FastLoop runs Tetris episodes as fast as possible. It:
  1. Loads a frozen PolicyPack at startup (or every pack_refresh_interval eps)
  2. Uses PolicyPack biases to nudge action selection (non-blocking)
  3. Logs every episode to ExperimentDB + TelemetryLogger
  4. Flags novelty events (unusual board states) to the DB for SlowLoop pickup
  5. Never calls Tetra, the evolver, or the dreamer — those live in SlowLoop

Why separate?
  The original DreamerTetrisRunner blocks on hypothesis evaluation and dream
  cycles. For bulk data collection (overnight runs, curriculum), you want
  200 eps in ~8s, not ~80s. FastLoop achieves this by freezing the policy
  and removing all per-episode LLM/evolver overhead.

The policy is refreshed by SlowLoop writing new PolicyPacks to the DB.
FastLoop simply reloads the latest pack every pack_refresh_interval episodes —
no coordination, no locks, no shared state.

Usage
-----
    from throng4.runners.fast_loop import FastLoop
    from throng4.storage import ExperimentDB, TelemetryLogger

    db = ExperimentDB('experiments/experiments.db')
    logger = TelemetryLogger()

    loop = FastLoop(game='tetris', level=2, db=db, logger=logger)
    stats = loop.run(n_episodes=500, verbose=True)
    print(stats)

    # CLI:
    python -m throng4.runners.fast_loop --level 2 --episodes 500
"""

from __future__ import annotations

import time
import sys
import os
import argparse
import numpy as np
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
from throng4.storage.experiment_db import ExperimentDB
from throng4.storage.telemetry_logger import TelemetryLogger
from throng4.storage.policy_pack import PolicyPack, PromotionGates
from throng4.storage.auto_metric import AutoMetric
from throng4.cognition.threat_estimator import ThreatEstimator
from throng4.cognition.mode_controller import ModeController


# ── Novelty detector ───────────────────────────────────────────────────────

class _NoveltyDetector:
    """
    Lightweight online novelty detector using exponential moving statistics.

    Flags episodes where the outcome is surprising (very high or very low
    relative to the running EMA). These flagged episodes are written to the
    DB events table for SlowLoop analysis.
    """

    def __init__(self, window: float = 0.01):
        self._ema = None
        self._ema_sq = None
        self._window = window   # EMA smoothing factor

    def update(self, value: float) -> Optional[float]:
        """
        Update with a new value. Returns z-score-like surprise, or None
        if not yet calibrated (< ~100 updates).
        """
        if self._ema is None:
            self._ema = value
            self._ema_sq = value ** 2
            return None

        self._ema += self._window * (value - self._ema)
        self._ema_sq += self._window * (value ** 2 - self._ema_sq)
        variance = max(self._ema_sq - self._ema ** 2, 1e-6)
        std = variance ** 0.5
        return (value - self._ema) / std

    def is_novel(self, value: float, threshold: float = 2.5) -> tuple:
        """Returns (is_novel: bool, surprise: float)."""
        surprise = self.update(value)
        if surprise is None:
            return False, 0.0
        return abs(surprise) >= threshold, float(surprise)


# ── FastLoop ───────────────────────────────────────────────────────────────

class FastLoop:
    """
    High-speed episode runner with frozen PolicyPack.

    No dreamer, no LLM, no evolver. Pure simulation + logging.
    """

    def __init__(self,
                 game: str = 'tetris',
                 level: int = 2,
                 db: Optional[ExperimentDB] = None,
                 logger: Optional[TelemetryLogger] = None,
                 pack: Optional[PolicyPack] = None,
                 pack_refresh_interval: int = 100,
                 db_path: str = 'experiments/experiments.db',
                 log_dir: str = 'experiments/logs',
                 novelty_threshold: float = 2.5):
        """
        Args:
            game:                   Game name tag for DB records.
            level:                  Tetris curriculum level.
            db:                     Open ExperimentDB (created if None).
            logger:                 TelemetryLogger (created if None).
            pack:                   Initial PolicyPack (loaded from DB if None).
            pack_refresh_interval:  Reload PolicyPack every N episodes.
            db_path:                Path for auto-created ExperimentDB.
            log_dir:                Path for auto-created TelemetryLogger.
            novelty_threshold:      Z-score threshold for novelty flagging.
        """
        self.game = game
        self.level = level
        self.pack_refresh_interval = pack_refresh_interval
        self.novelty_threshold = novelty_threshold

        # Storage
        self.db     = db     or ExperimentDB(db_path)
        self.logger = logger or TelemetryLogger(log_dir)

        # PolicyPack — load latest from DB if not provided
        self.pack = pack or PolicyPack.load_latest(self.db, game=game)

        # Agent — fresh per FastLoop instance, not transferred from DreamerRunner
        max_pieces = {1:50, 2:100, 3:150, 4:200, 5:300, 6:400, 7:500}.get(level, 200)
        _tmp_adapter = TetrisAdapter(level=level, max_pieces=max_pieces)
        _tmp_adapter.reset()
        valid = _tmp_adapter.get_valid_actions()
        n_features = len(_tmp_adapter.make_features(valid[0])) if valid else 24

        self.agent = PortableNNAgent(
            n_features=n_features,
            config=AgentConfig(n_hidden=128, epsilon=0.15,
                               gamma=0.95, learning_rate=0.005),
        )
        self.max_pieces = max_pieces

        # Novelty detector on lines_cleared
        self._novelty = _NoveltyDetector()

        # AutoMetric — records raw board observations passively
        board_h = {1:8, 2:10, 3:12, 4:14, 5:16, 6:18, 7:20}.get(level, 12)
        self._auto_metric = AutoMetric(
            db=self.db,
            game=game,
            obs_shape=(board_h, _tmp_adapter.board_width),
            min_correlation=0.35,
            min_episodes=100,
            storage_path=f'{log_dir}/auto_metric_{game}_L{level}.jsonl',
        )
        self._auto_metric_interval = 500


        # ThreatEstimator + ModeController
        # Try level-specific estimator first, then universal, then None
        self._threat: Optional[ThreatEstimator] = None
        for path in [f'experiments/threat_estimator_L{level}.npz',
                     'experiments/threat_estimator_all.npz']:
            try:
                self._threat = ThreatEstimator.load(path)
                if verbose:
                    print(f"  [amygdala] Loaded {path}  "
                          f"threshold={self._threat.threshold}")
                break
            except Exception:
                pass
        if self._threat is None and verbose:
            print("  [amygdala] No estimator found — mode will stay EXECUTE")

        self._mode_ctrl = ModeController(
            enter_survive=0.60, exit_survive=0.35,
            enter_explore=0.20, hysteresis_steps=5,
        )

        # Session ID for this FastLoop run
        import uuid
        self.session_id = str(uuid.uuid4())[:8]

    # ── Core episode ───────────────────────────────────────────────────────

    def _run_episode(self, episode_num: int) -> Dict[str, Any]:
        """Run one episode. Returns stats dict."""
        self.agent.reset_episode()
        self._mode_ctrl.reset_episode()
        adapter = TetrisAdapter(level=self.level, max_pieces=self.max_pieces)
        state = adapter.reset()

        done           = False
        episode_reward = 0.0
        steps          = 0
        pack_nudges    = 0
        survive_steps  = 0
        peak_threat    = 0.0

        while not done:
            valid_actions = adapter.get_valid_actions()
            if not valid_actions:
                break

            feature_fn = adapter.make_features

            # ── Amygdala: threat evaluation from current board state ───────
            mode = 'EXECUTE'
            if self._threat is not None:
                try:
                    bf = adapter._compute_board_features(adapter.env.board)
                    threat_feat = ThreatEstimator._episode_to_features({
                        'level':         self.level,
                        'max_height':    bf['max_height'],
                        'holes':         bf['holes'],
                        'bumpiness':     bf['bumpiness'],
                        'lines_cleared': adapter.env.lines_cleared,
                        'pieces_placed': adapter.env.pieces_placed,
                    })
                    if threat_feat is not None:
                        threat     = self._threat.predict(threat_feat)
                        mode       = self._mode_ctrl.update(threat, step=steps)
                        peak_threat = max(peak_threat, threat)
                        if mode == 'SURVIVE':
                            survive_steps += 1
                except Exception:
                    pass

            explore = (mode != 'SURVIVE')  # no random exploration in SURVIVE

            # ── Action selection ──────────────────────────────────────────
            if self.pack and np.random.random() < 0.3:
                bias = self.pack.get_action_bias(state, valid_actions)
                if bias and bias['confidence'] > 0.6:
                    action = self.agent.select_action(
                        valid_actions, feature_fn, explore=False
                    )
                    pack_nudges += 1
                else:
                    action = self.agent.select_action(
                        valid_actions, feature_fn, explore=explore
                    )
            else:
                action = self.agent.select_action(
                    valid_actions, feature_fn, explore=explore
                )

            features = feature_fn(action)
            next_state, reward, done, info = adapter.step(action)
            self.agent.record_step(features, reward)

            episode_reward += reward
            steps          += 1
            state           = next_state

        ep_info = adapter.get_info()
        lines   = ep_info['lines_cleared']
        pieces  = ep_info['pieces_placed']
        self.agent.end_episode(final_score=float(lines))

        # Board features for DB
        try:
            bf        = adapter._compute_board_features(adapter.env.board)
            max_h     = bf['max_height']
            holes     = bf['holes']
            bumpiness = float(bf['bumpiness'])
        except Exception:
            max_h = holes = 0; bumpiness = 0.0

        # AutoMetric: record raw board snapshot passively
        try:
            raw_board = np.array(adapter.env.board, dtype=np.float32)
            self._auto_metric.record(
                raw_obs=raw_board,
                outcome=float(lines),
                episode_id=f'{self.session_id}_{episode_num}',
                extra={
                    'known_holes':     float(holes),
                    'known_max_height': float(max_h),
                    'known_bumpiness': float(bumpiness),
                    'pieces_placed':   float(pieces),
                }
            )
        except Exception:
            pass

        return {
            'episode':      episode_num,
            'lines':        lines,
            'pieces':       pieces,
            'score':        round(episode_reward, 2),
            'steps':        steps,
            'pack_nudges':  pack_nudges,
            'max_height':   max_h,
            'holes':        holes,
            'bumpiness':    bumpiness,
            'survive_steps': survive_steps,
            'peak_threat':  round(peak_threat, 3),
            'mode':         self._mode_ctrl.mode,
        }

    # ── Main run loop ──────────────────────────────────────────────────────

    def run(self, n_episodes: int = 500, verbose: bool = True) -> Dict[str, Any]:
        """
        Run n_episodes at full speed.

        Args:
            n_episodes: Number of episodes to run.
            verbose:    Print progress every 50 episodes.

        Returns:
            Summary stats dict.
        """
        t_start = time.time()
        all_lines: List[int] = []

        pack_version  = self.pack.version if self.pack else 0
        pack_label    = f"v{pack_version}" if self.pack else "none"

        if verbose:
            print(f"\n{'='*60}")
            print(f"FastLoop  game={self.game}  level={self.level}  "
                  f"episodes={n_episodes}  pack={pack_label}")
            print(f"{'='*60}")

        for ep in range(n_episodes):
            # Periodic PolicyPack reload
            if ep > 0 and ep % self.pack_refresh_interval == 0:
                new_pack = PolicyPack.load_latest(self.db, game=self.game)
                if new_pack and (not self.pack or new_pack.version > self.pack.version):
                    self.pack = new_pack
                    pack_version = new_pack.version
                    if verbose:
                        print(f"  [pack reload] v{pack_version} at ep {ep}")

            result = self._run_episode(ep)
            lines  = result['lines']
            all_lines.append(lines)

            # Log to both stores
            self.db.log_episode(
                game=self.game, level=self.level, episode_num=ep,
                score=result['score'], lines_cleared=lines,
                pieces_placed=result['pieces'],
                max_height=result['max_height'],
                holes=result['holes'],
                bumpiness=result['bumpiness'],
                outcome_tags={'pack_nudges': result['pack_nudges'],
                              'pack_version': pack_version},
                policy_pack_version=pack_version,
                session_id=self.session_id,
            )
            self.logger.log_episode({
                'game': self.game, 'level': self.level,
                'episode': ep, 'session': self.session_id,
                'lines': lines, 'score': result['score'],
                'pieces': result['pieces'],
                'pack_version': pack_version,
                'pack_nudges': result['pack_nudges'],
            })

            # Novelty check
            is_novel, surprise = self._novelty.is_novel(
                lines, self.novelty_threshold
            )
            if is_novel:
                self.db.log_event(
                    event_type='novelty',
                    data={'lines': lines, 'surprise': round(surprise, 2),
                          'episode': ep, 'level': self.level,
                          'session': self.session_id},
                )

            # Periodic AutoMetric analysis
            if ep > 0 and ep % self._auto_metric_interval == 0:
                discoveries = self._auto_metric.analyze()
                if discoveries and verbose:
                    print(f"  [AutoMetric] {len(discoveries)} features discovered:")
                    for d in discoveries[:5]:
                        print(f"    {d.name:<30} r={d.correlation:+.3f}  {d.description[:60]}")

            # Progress print
            if verbose and (ep + 1) % 50 == 0:
                recent = all_lines[-50:]
                avg        = sum(recent) / len(recent)
                window_best = max(recent)          # best in this 50-ep window
                session_best = max(all_lines)      # all-time best this run
                elapsed = time.time() - t_start
                eps_per_s = (ep + 1) / elapsed
                print(f"  Ep {ep+1:>5}/{n_episodes}  "
                      f"avg={avg:6.1f}  best={window_best:4d}  "
                      f"(session={session_best})  "
                      f"{eps_per_s:.1f} ep/s")

        elapsed = time.time() - t_start
        mean_lines = sum(all_lines) / len(all_lines) if all_lines else 0
        final_20   = all_lines[-20:] if len(all_lines) >= 20 else all_lines
        final_mean = sum(final_20) / len(final_20) if final_20 else 0

        summary = {
            'game': self.game,
            'level': self.level,
            'episodes': n_episodes,
            'mean_lines': round(mean_lines, 2),
            'final_mean_lines': round(final_mean, 2),
            'max_lines': max(all_lines) if all_lines else 0,
            'elapsed_s': round(elapsed, 1),
            'eps_per_s': round(n_episodes / elapsed, 1),
            'pack_version': pack_version,
            'session_id': self.session_id,
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"DONE  mean={mean_lines:.1f}  max={summary['max_lines']}  "
                  f"{summary['eps_per_s']} ep/s  ({elapsed:.1f}s)")
            print(f"{'='*60}\n")

        return summary


# ── CLI entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FastLoop — high-speed Tetris runner')
    parser.add_argument('--game',     default='tetris')
    parser.add_argument('--level',    type=int, default=2)
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--db',       default='experiments/experiments.db')
    parser.add_argument('--log-dir',  default='experiments/logs')
    parser.add_argument('--refresh',  type=int, default=100,
                        help='PolicyPack reload interval (episodes)')
    args = parser.parse_args()

    db     = ExperimentDB(args.db)
    logger = TelemetryLogger(args.log_dir)
    loop   = FastLoop(game=args.game, level=args.level,
                      db=db, logger=logger,
                      pack_refresh_interval=args.refresh)
    stats  = loop.run(n_episodes=args.episodes, verbose=True)

    import json
    print(json.dumps(stats, indent=2))
    db.close()


if __name__ == '__main__':
    main()
