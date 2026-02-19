"""
threat_estimator.py — Learned threat detection for novel environments.

The ThreatEstimator answers: "Given the current state, how likely is it
that this episode ends (badly) within the next k steps?"

It is NOT a simulator. It does NOT use future information. It learns from
completed past episodes: states that occurred near the end of bad episodes
get labeled as high-threat. The neural net then learns to recognize those
patterns from current-state features alone.

Key design constraints (anti-save-scum):
  - Input: current state features only
  - Labels: retrospective from completed episodes (not simulated futures)
  - Output is noisy by design — false positives are expected and correct
  - Survival mode does NOT guarantee survival

Usage
-----
    # Training (in SlowLoop or offline):
    te = ThreatEstimator(n_features=18, k_steps=5)
    te.train_from_db(db, game='tetris', n_episodes=500)
    te.save('experiments/threat_estimator_L3.npz')

    # Inference (in FastLoop, per step):
    te = ThreatEstimator.load('experiments/threat_estimator_L3.npz')
    threat = te.predict(state_features)   # float in [0, 1]
    if threat > 0.6:
        # switch to SURVIVE mode
"""

from __future__ import annotations

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


class ThreatEstimator:
    """
    Small 2-layer neural net mapping state features → P(death within k steps).

    Architecture: Linear(n_features → 32) → ReLU → Linear(32 → 1) → Sigmoid
    Loss:         Binary cross-entropy
    Training:     Mini-batch SGD, batches of 64, up to 200 epochs

    Developmental calibration:
        - Starts knowing nothing (random weights, baseline threat ≈ 0.5)
        - Becomes usefully calibrated after ~500 labeled transitions
        - Threshold lowers as estimator matures (more hair-trigger with exp)
    """

    def __init__(self,
                 n_features: int = 18,
                 hidden_size: int = 32,
                 k_steps: int = 5,
                 threshold: float = 0.60,
                 lr: float = 0.01):
        """
        Args:
            n_features:   Size of state feature vector (matches TetrisAdapter).
            hidden_size:  Hidden layer width.
            k_steps:      Episodes/steps within which death = "high threat".
            threshold:    threat_level above this → SURVIVE mode.
            lr:           Learning rate.
        """
        self.n_features  = n_features
        self.hidden_size = hidden_size
        self.k_steps     = k_steps
        self.threshold   = threshold
        self.lr          = lr

        # Xavier init — starts knowing nothing
        scale_w1 = np.sqrt(2.0 / n_features)
        scale_w2 = np.sqrt(2.0 / hidden_size)
        self.W1 = np.random.randn(n_features, hidden_size).astype(np.float32) * scale_w1
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, 1).astype(np.float32) * scale_w2
        self.b2 = np.zeros(1, dtype=np.float32)

        # Training history
        self.n_trained     = 0
        self.train_history: List[Dict] = []
        self.created_at    = time.strftime('%Y-%m-%d %H:%M')

    # ── Forward pass ────────────────────────────────────────────────────────

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass. Returns (hidden, output_prob)."""
        h = np.maximum(0, X @ self.W1 + self.b1)           # ReLU
        logit = h @ self.W2 + self.b2                       # Linear
        prob  = 1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10)))  # Sigmoid
        return h, prob

    def predict(self, features: np.ndarray) -> float:
        """
        Predict threat level for a single state.

        Args:
            features: State feature vector (same format as TetrisAdapter).

        Returns:
            float in [0, 1]. Values > self.threshold indicate high threat.
        """
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        _, prob = self._forward(x)
        return float(prob[0, 0])

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch prediction. features shape: (N, n_features)."""
        x = np.asarray(features, dtype=np.float32)
        _, prob = self._forward(x)
        return prob.ravel()

    # ── Label generation ────────────────────────────────────────────────────

    @staticmethod
    def label_episodes(episodes: List[Dict],
                       k_steps: int = 5,
                       min_outcome: float = 0.0,
                       outcome_key: str = 'lines_cleared') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (features, labels) from a list of episode dicts.

        Labeling rule (retrospective — NOT future-peeking):
          - Sort episodes by outcome (lines_cleared ascending)
          - Bottom 30% = "bad" episodes → label their terminal board state as 1.0
          - Top 30% = "good" episodes → label their terminal board state as 0.0
          - Middle 40% = ambiguous → excluded from training

        This creates high-threat labels for board states that, in past experience,
        coincided with poor outcomes. The estimator learns the current-state
        signature of danger.

        Args:
            episodes:    List of episode dicts from ExperimentDB.get_episodes().
            k_steps:     (Not used directly here — kept for API symmetry with
                          step-level labeling variant.)
            min_outcome: Ignore episodes with outcome below this (e.g. 0 lines).
            outcome_key: Which field to rank by.

        Returns:
            X: np.ndarray of shape (N, n_features)
            y: np.ndarray of shape (N,), values in {0.0, 1.0}
        """
        # Filter episodes that have board feature data
        valid = []
        for ep in episodes:
            outcome = ep.get(outcome_key, 0) or 0
            holes    = ep.get('holes')
            max_h    = ep.get('max_height')
            bumpiness = ep.get('bumpiness')
            if holes is None or max_h is None:
                continue
            valid.append((outcome, ep))

        if len(valid) < 20:
            return np.array([]), np.array([])

        valid.sort(key=lambda x: x[0])
        n = len(valid)
        cutoff_bad  = n // 3       # bottom third
        cutoff_good = n - n // 3   # top third

        X_list, y_list = [], []

        for i, (outcome, ep) in enumerate(valid):
            if outcome < min_outcome:
                continue

            # Build feature vector from episode board stats
            feat = ThreatEstimator._episode_to_features(ep)
            if feat is None:
                continue

            if i < cutoff_bad:
                X_list.append(feat)
                y_list.append(1.0)   # high threat
            elif i >= cutoff_good:
                X_list.append(feat)
                y_list.append(0.0)   # low threat
            # else: excluded (ambiguous middle)

        if not X_list:
            return np.array([]), np.array([])

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    @staticmethod
    def _episode_to_features(ep: Dict) -> Optional[np.ndarray]:
        """
        Build a level-aware feature vector from episode DB record.

        At high levels (L6-L7), absolute board height is useless (always near max).
        Instead we use: holes relative to board volume, pieces_placed rate,
        and lines_per_piece efficiency as the threat discriminators.

        Features that scale by level:
          - hole density: holes / (board_volume * expected_fill_rate)
          - expected holes: level-specific baseline
          - pieces_survival: how long did we survive (key at L7 where pieces ~= 5)
        """
        try:
            level     = float(ep.get('level') or 3)
            max_h     = float(ep.get('max_height') or 0)
            holes     = float(ep.get('holes') or 0)
            bumpiness = float(ep.get('bumpiness') or 0)
            lines     = float(ep.get('lines_cleared') or 0)
            pieces    = float(ep.get('pieces_placed') or 1)

            board_h = {1:8, 2:10, 3:12, 4:14, 5:16, 6:18, 7:20}.get(int(level), 12)
            board_w = 6.0

            volume   = board_h * board_w
            # Expected holes at this level (learned from data: ~4*level^1.5)
            expected_holes = min(4.0 * (level ** 1.5), volume * 0.7)
            # Holes relative to level-specific expectation
            hole_excess    = max(0.0, holes - expected_holes) / (volume * 0.3 + 1.0)
            # Pieces-per-line efficiency: higher = safer (able to clear while alive)
            lines_per_piece = lines / max(pieces, 1)
            # Survival rate: pieces relative to a "good" episode for this level
            good_pieces = max(10.0, 200.0 / (level ** 1.5))  # decreases with level
            survival_norm = min(pieces / good_pieces, 1.0)

            feat = np.array([
                max_h    / board_h,                        # height fraction
                holes    / volume,                         # hole density (0-1)
                hole_excess,                               # holes above level expectation
                bumpiness / (board_h * (board_w - 1)),    # bumpiness norm
                min(lines / 50.0, 1.0),                    # lines cleared (capped)
                survival_norm,                             # pieces survived (level-scaled)
                min(lines_per_piece * 5.0, 1.0),          # lines/piece efficiency
                (max_h / board_h) ** 2,                   # height danger (quadratic)
                float(max_h > 0.9 * board_h),             # critical height (90%)
                float(hole_excess > 0.5),                 # holes way above expectation
                level / 7.0,                              # curriculum level
            ], dtype=np.float32)
            return feat
        except Exception:
            return None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              epochs: int = 150,
              batch_size: int = 64,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train on (X, y) pairs.

        Args:
            X: Feature matrix (N, n_features).
            y: Labels (N,), values in {0.0, 1.0}.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            verbose: Print loss curve.

        Returns:
            Training summary dict.
        """
        n = len(X)
        if n == 0:
            return {'error': 'no training data'}

        # Resize input layer if needed
        if X.shape[1] != self.n_features:
            self.n_features = X.shape[1]
            scale = np.sqrt(2.0 / self.n_features)
            self.W1 = np.random.randn(self.n_features, self.hidden_size).astype(np.float32) * scale

        losses = []
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X[batch_idx]
                yb = y[batch_idx].reshape(-1, 1)

                # Forward
                h, prob = self._forward(Xb)
                eps   = 1e-7
                loss  = -np.mean(yb * np.log(prob + eps) + (1 - yb) * np.log(1 - prob + eps))

                # Backward
                dL_dprob = -(yb / (prob + eps) - (1 - yb) / (1 - prob + eps)) / len(Xb)
                dL_dlogit = dL_dprob * prob * (1 - prob)

                dL_dW2 = h.T @ dL_dlogit
                dL_db2 = dL_dlogit.sum(axis=0)
                dL_dh  = dL_dlogit @ self.W2.T
                dL_dh_pre = dL_dh * (h > 0)  # ReLU gradient

                dL_dW1 = Xb.T @ dL_dh_pre
                dL_db1 = dL_dh_pre.sum(axis=0)

                # Update
                self.W2 -= self.lr * dL_dW2
                self.b2 -= self.lr * dL_db2
                self.W1 -= self.lr * dL_dW1
                self.b1 -= self.lr * dL_db1

                epoch_loss += loss
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 50 == 0:
                # Accuracy on full set
                _, preds = self._forward(X)
                preds_binary = (preds.ravel() > 0.5).astype(float)
                acc = np.mean(preds_binary == y)
                pos_rate = float(np.mean(preds.ravel()[y == 1.0]))
                neg_rate = float(np.mean(preds.ravel()[y == 0.0]))
                print(f"  Epoch {epoch+1:>3}/{epochs}  loss={avg_loss:.4f}  "
                      f"acc={acc:.1%}  "
                      f"P(threat|bad)={pos_rate:.2f}  P(threat|good)={neg_rate:.2f}")

        self.n_trained += n

        # Final stats
        _, preds = self._forward(X)
        preds_binary = (preds.ravel() > 0.5).astype(float)
        final_acc = float(np.mean(preds_binary == y))

        summary = {
            'n_samples':   n,
            'epochs':      epochs,
            'final_loss':  float(losses[-1]),
            'final_acc':   final_acc,
            'n_positive':  int(np.sum(y == 1.0)),
            'n_negative':  int(np.sum(y == 0.0)),
            'trained_at':  time.strftime('%Y-%m-%d %H:%M'),
        }
        self.train_history.append(summary)
        return summary

    def train_from_db(self, db, game: str = 'tetris',
                      n_episodes: int = 2000,
                      level: Optional[int] = None,
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Train directly from ExperimentDB.

        Args:
            db:          Open ExperimentDB instance.
            game:        Game tag to query.
            n_episodes:  How many recent episodes to use for labeling.
            level:       If set, train only on episodes from this level.
            verbose:     Print training progress.

        Returns:
            Training summary dict.
        """
        episodes = db.get_episodes(game=game, limit=n_episodes)
        if level is not None:
            episodes = [e for e in episodes if e.get('level') == level]

        if verbose:
            print(f"[ThreatEstimator] Labeling {len(episodes)} episodes "
                  f"(game={game}, level={level or 'all'}, k={self.k_steps})...")

        X, y = self.label_episodes(episodes, k_steps=self.k_steps)

        if len(X) == 0:
            print("[ThreatEstimator] No valid training data found.")
            return {'error': 'no data'}

        if verbose:
            print(f"[ThreatEstimator] Training on {len(X)} samples "
                  f"({int(np.sum(y))} high-threat, {int(np.sum(1-y))} low-threat)...")

        return self.train(X, y, verbose=verbose)

    # ── Mode decision ────────────────────────────────────────────────────────

    def mode(self, features: np.ndarray,
             hysteresis_state: Optional[str] = None) -> str:
        """
        Return recommended operating mode for these features.

        Args:
            features:         Current state features.
            hysteresis_state: Current mode (for hysteresis — SURVIVE is sticky).

        Returns:
            One of: 'EXPLORE', 'EXECUTE', 'SURVIVE'
        """
        threat = self.predict(features)

        # Hysteresis: once in SURVIVE, need threat < 0.35 to exit
        exit_threshold = 0.35 if hysteresis_state == 'SURVIVE' else self.threshold

        if threat >= self.threshold:
            return 'SURVIVE'
        elif threat >= 0.20:
            return 'EXECUTE'
        else:
            return 'EXPLORE'

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save weights and metadata to .npz file."""
        np.savez(path,
                 W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 metadata=json.dumps({
                     'n_features':  self.n_features,
                     'hidden_size': self.hidden_size,
                     'k_steps':     self.k_steps,
                     'threshold':   self.threshold,
                     'n_trained':   self.n_trained,
                     'created_at':  self.created_at,
                     'saved_at':    time.strftime('%Y-%m-%d %H:%M'),
                     'train_history': self.train_history,
                 }))
        print(f"[ThreatEstimator] Saved to {path} "
              f"(n_trained={self.n_trained}, threshold={self.threshold})")

    @classmethod
    def load(cls, path: str) -> 'ThreatEstimator':
        """Load from .npz file."""
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data['metadata']))

        te = cls(
            n_features=meta['n_features'],
            hidden_size=meta['hidden_size'],
            k_steps=meta['k_steps'],
            threshold=meta['threshold'],
        )
        te.W1 = data['W1']
        te.b1 = data['b1']
        te.W2 = data['W2']
        te.b2 = data['b2']
        te.n_trained     = meta.get('n_trained', 0)
        te.created_at    = meta.get('created_at', '')
        te.train_history = meta.get('train_history', [])
        return te

    def __repr__(self):
        return (f"ThreatEstimator(n_features={self.n_features}, "
                f"threshold={self.threshold}, n_trained={self.n_trained})")
