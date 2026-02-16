"""
Portable NN Agent — Environment-agnostic neural network agent.

This agent can work with any environment that provides:
- A feature extraction function: action → float[]
- A list of valid actions per step

Key features:
- Save/load weights for transfer learning
- 3-move lookahead (adaptive depth)
- Heuristic-biased initialization
- Replay buffer training
"""

import numpy as np
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for portable agent."""
    n_hidden: int = 48
    epsilon: float = 0.20
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.02
    gamma: float = 0.95
    learning_rate: float = 0.005
    replay_buffer_size: int = 5000
    batch_size: int = 128
    lookahead_depth: int = 3
    lookahead_threshold: int = 30  # Use depth-2 if >30 actions


class PortableNNAgent:
    """
    Environment-agnostic NN agent with transfer learning support.
    
    Architecture:
        Input → Hidden (ReLU) → Output (scalar value)
    
    The agent doesn't know about specific environments — it only knows:
    - How many features it expects (n_features)
    - How to score actions via a feature function
    """
    
    def __init__(self, n_features: int, config: Optional[AgentConfig] = None):
        """
        Initialize agent.
        
        Args:
            n_features: Size of feature vector
            config: Agent configuration
        """
        self.n_features = n_features
        self.config = config or AgentConfig()
        
        # Network weights (He initialization)
        n_hidden = self.config.n_hidden
        self.W1 = np.random.randn(n_hidden, n_features) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(1, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(1)
        
        # Apply heuristic bias (warm start)
        self._apply_heuristic_bias()
        
        # Training state
        self.epsilon = self.config.epsilon
        self.episode_count = 0
        self.best_score = 0
        self.recent_scores = []
        
        # Replay buffer: list of {x: features, target: G}
        self.replay_buffer = []
        
        # Episode buffer: list of {x: features, reward: r}
        self.episode_buffer = []
        
        # Metrics
        self.total_updates = 0
        self.recent_loss = []
        
        # Cache for forward pass
        self._last_x = None
        self._last_h = None
    
    def _apply_heuristic_bias(self):
        """Add small heuristic-aligned bias to hidden layer."""
        bias_strength = 0.3
        for i in range(self.config.n_hidden):
            self.b1[i] += bias_strength * (np.random.rand() * 0.1)
    
    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass through network.
        
        Args:
            x: Feature vector (n_features,) or larger
            
        Returns:
            Scalar value prediction
        """
        # Handle variable input size via padding/truncation
        if len(x) > self.n_features:
            # Pad weights with zeros for extra features (smooth adaptation)
            x = x[:self.n_features]  # Truncate for now (simple approach)
        elif len(x) < self.n_features:
            # Pad input with zeros
            x = np.pad(x, (0, self.n_features - len(x)), mode='constant')
        
        self._last_x = x
        
        # Hidden layer with ReLU
        z1 = self.W1 @ x + self.b1
        self._last_h = np.maximum(0, z1)
        
        # Output layer
        out = self.W2 @ self._last_h + self.b2
        return float(out[0])
    
    def select_action(
        self,
        valid_actions: List[Any],
        feature_fn: Callable[[Any], np.ndarray],
        explore: bool = True,
        lookahead_fn: Optional[Callable[[Any], List[Any]]] = None
    ) -> Any:
        """
        Select best action using epsilon-greedy + lookahead.
        
        Args:
            valid_actions: List of valid actions for current state
            feature_fn: Function mapping action → feature vector
            explore: Whether to use epsilon-greedy
            lookahead_fn: Optional function for lookahead: action → next_valid_actions
            
        Returns:
            Selected action
        """
        if not valid_actions:
            return None
        
        # Epsilon-greedy exploration
        if explore and np.random.rand() < self.epsilon:
            return valid_actions[np.random.randint(len(valid_actions))]
        
        # Determine lookahead depth
        depth = self.config.lookahead_depth
        if len(valid_actions) > self.config.lookahead_threshold:
            depth = 2
        
        # Score all actions
        best_action = valid_actions[0]
        best_value = -np.inf
        
        for action in valid_actions:
            value = self._score_action(action, feature_fn, depth, lookahead_fn)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action
    
    def _score_action(
        self,
        action: Any,
        feature_fn: Callable[[Any], np.ndarray],
        depth: int,
        lookahead_fn: Optional[Callable[[Any], List[Any]]]
    ) -> float:
        """Score an action with optional lookahead."""
        features = feature_fn(action)
        score = self.forward(features)
        
        if depth <= 1 or lookahead_fn is None:
            return score
        
        # Lookahead: get next valid actions after this action
        next_actions = lookahead_fn(action)
        if not next_actions:
            return score - 10  # Penalize dead-end states
        
        if depth == 2:
            # Depth 2: best next action
            best_next = max(
                (self.forward(feature_fn(a)) for a in next_actions),
                default=0
            )
            return score + 0.5 * best_next
        
        # Depth 3: sample random third moves
        best_next = -np.inf
        for next_action in next_actions:
            next_score = self.forward(feature_fn(next_action))
            
            # Sample 3 random third actions
            third_actions = lookahead_fn(next_action)
            if not third_actions:
                third_avg = -10
            else:
                third_samples = np.random.choice(
                    len(third_actions),
                    size=min(3, len(third_actions)),
                    replace=False
                )
                third_avg = np.mean([
                    self.forward(feature_fn(third_actions[i]))
                    for i in third_samples
                ])
            
            combined = next_score + 0.25 * third_avg
            if combined > best_next:
                best_next = combined
        
        return score + 0.5 * best_next
    
    def record_step(self, features: np.ndarray, reward: float):
        """
        Record a step in the current episode.
        
        Args:
            features: Feature vector for the action taken
            reward: Immediate reward received
        """
        self.episode_buffer.append({'x': features.copy(), 'reward': reward})
    
    def end_episode(self, final_score: float):
        """
        End episode and train on accumulated experience.
        
        Args:
            final_score: Final score/lines for this episode
        """
        self.episode_count += 1
        self.recent_scores.append(final_score)
        if len(self.recent_scores) > 20:
            self.recent_scores.pop(0)
        
        if final_score > self.best_score:
            self.best_score = final_score
        
        # Compute discounted returns (G)
        if self.episode_buffer:
            G = 0
            for i in range(len(self.episode_buffer) - 1, -1, -1):
                G = self.episode_buffer[i]['reward'] + self.config.gamma * G
                self.replay_buffer.append({
                    'x': self.episode_buffer[i]['x'],
                    'target': G
                })
            
            # Limit buffer size
            if len(self.replay_buffer) > self.config.replay_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.config.replay_buffer_size:]
            
            # Train on batch
            self._train_batch()
        
        # Clear episode buffer
        self.episode_buffer = []
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
    
    def _train_batch(self):
        """Train network on a batch from replay buffer."""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(
            len(self.replay_buffer),
            size=self.config.batch_size,
            replace=False
        )
        batch = [self.replay_buffer[i] for i in indices]
        
        total_loss = 0.0
        
        for sample in batch:
            # Forward pass
            pred = self.forward(sample['x'])
            target = sample['target']
            
            # Compute error (clipped for stability)
            error = pred - target
            clipped_error = np.clip(error, -5, 5)
            total_loss += error ** 2
            
            # Backprop output layer
            self.W2 -= self.config.learning_rate * clipped_error * self._last_h
            self.b2 -= self.config.learning_rate * clipped_error
            
            # Backprop hidden layer (only active neurons)
            for i in range(self.config.n_hidden):
                if self._last_h[i] > 0:
                    dh = clipped_error * self.W2[0, i]
                    self.W1[i] -= self.config.learning_rate * dh * self._last_x
                    self.b1[i] -= self.config.learning_rate * dh
        
        self.total_updates += 1
        avg_loss = total_loss / self.config.batch_size
        self.recent_loss.append(avg_loss)
        if len(self.recent_loss) > 50:
            self.recent_loss.pop(0)
    
    def save_weights(self, path: str):
        """
        Save agent weights to file.
        
        Args:
            path: Path to save weights (will add .npz extension)
        """
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            n_features=self.n_features,
            config_dict={
                'n_hidden': self.config.n_hidden,
                'epsilon': self.epsilon,
                'episode_count': self.episode_count,
                'best_score': self.best_score
            }
        )
    
    def load_weights(self, path: str):
        """
        Load agent weights from file.
        
        Args:
            path: Path to weights file
        """
        data = np.load(path, allow_pickle=True)
        
        # Verify feature count matches
        if data['n_features'] != self.n_features:
            raise ValueError(
                f"Feature mismatch: saved={data['n_features']}, "
                f"current={self.n_features}"
            )
        
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        
        # Restore training state
        config_dict = data['config_dict'].item()
        self.epsilon = config_dict.get('epsilon', self.epsilon)
        self.episode_count = config_dict.get('episode_count', 0)
        self.best_score = config_dict.get('best_score', 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        avg_score = (
            np.mean(self.recent_scores) if self.recent_scores else 0
        )
        avg_loss = (
            np.mean(self.recent_loss) if self.recent_loss else 0
        )
        
        return {
            'episode': self.episode_count,
            'epsilon': self.epsilon,
            'best_score': self.best_score,
            'avg_score': avg_score,
            'avg_loss': avg_loss,
            'total_updates': self.total_updates,
            'buffer_size': len(self.replay_buffer),
            'n_params': (
                self.W1.size + self.b1.size +
                self.W2.size + self.b2.size
            )
        }
    
    def reset_episode(self):
        """Reset episode-specific state (called at episode start)."""
        self.episode_buffer = []
