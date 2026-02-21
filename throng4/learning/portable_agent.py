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
    n_hidden: int = 256       # First hidden layer size
    n_hidden2: int = 128      # Second hidden layer size (0 = single-layer mode)
    n_step: int = 5           # N-step return horizon (1 = REINFORCE)
    epsilon: float = 0.20
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.02
    gamma: float = 0.95
    learning_rate: float = 0.005
    replay_buffer_size: int = 50000
    batch_size: int = 64
    lookahead_depth: int = 3
    lookahead_threshold: int = 30  # Use depth-2 if >30 actions
    target_update_freq: int = 1000 # network sync freq
    train_freq: int = 4            # train every N steps
    # Abstract feature layer options
    use_abstract_features: bool = False  # Set True to use 84-dim portable feature space
    ext_noise_std: float = 0.02          # Gaussian noise on ext block during training
    #   (prevents over-reliance on any one adapter-specific ext slot)
    #   Set to 0.0 to disable (e.g. during eval or ablation comparison)


class ReplayBuffer:
    """Experience replay buffer for off-policy Q-learning."""
    def __init__(self, capacity: int, rng: np.random.RandomState):
        self.capacity = capacity
        self.rng = rng
        self.buffer = []
        self.pos = 0

    def push(self, x: np.ndarray, reward: float, next_x_list: List[np.ndarray], done: bool):
        transition = (x, reward, next_x_list, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Any]:
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class PortableNNAgent:
    """
    Environment-agnostic NN agent with transfer learning support.
    
    Architecture:
        Input → Hidden (ReLU) → Output (scalar value)
    
    The agent doesn't know about specific environments — it only knows:
    - How many features it expects (n_features)
    - How to score actions via a feature function
    """
    
    def __init__(self, n_features: int, config: Optional[AgentConfig] = None, seed: Optional[int] = None):
        """
        Initialize agent.
        
        Args:
            n_features: Size of feature vector
            config: Agent configuration
            seed: Random seed for internal operations
        """
        self.n_features = n_features
        self.config = config or AgentConfig()
        
        # Isolated random state
        self.rng = np.random.RandomState(seed)
        
        # Network weights — two hidden layers (He initialization)
        n_hidden = self.config.n_hidden
        n_hidden2 = self.config.n_hidden2
        self.W1 = self.rng.randn(n_hidden, n_features) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(n_hidden)
        # Second hidden layer (256 → 128)
        self.W2 = self.rng.randn(n_hidden2, n_hidden) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_hidden2)
        # Output layer (128 → 1)
        self.W3 = self.rng.randn(1, n_hidden2) * np.sqrt(2.0 / n_hidden2)
        self.b3 = np.zeros(1)
        
        # Apply heuristic bias (warm start)
        self._apply_heuristic_bias()
        
        # Training state
        self.epsilon = self.config.epsilon
        self.episode_count = 0
        self.best_score = 0
        self.recent_scores = []
        
        # Replay buffer (Q-learning)
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size, self.rng)
        
        # Metrics
        self.total_updates = 0
        self.step_counter = 0
        self.recent_loss = []
        
        # Target network
        self.sync_target_network()
        
        # Cache for forward pass
        self._last_x = None
        self._last_h1 = None
        self._last_h2 = None
    
    def sync_target_network(self):
        """Synchronize target network with live weights."""
        self.target_W1 = self.W1.copy()
        self.target_b1 = self.b1.copy()
        self.target_W2 = self.W2.copy()
        self.target_b2 = self.b2.copy()
        self.target_W3 = self.W3.copy()
        self.target_b3 = self.b3.copy()
    
    def _apply_heuristic_bias(self):
        """Add small heuristic-aligned bias to first hidden layer."""
        bias_strength = 0.3
        for i in range(self.config.n_hidden):
            self.b1[i] += bias_strength * (self.rng.rand() * 0.1)
    
    def forward_target(self, x: np.ndarray) -> float:
        """Forward pass using target network."""
        if len(x) > self.n_features:
            x = x[:self.n_features]
        elif len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)), mode='constant')

        z1 = self.target_W1 @ x + self.target_b1
        h1 = np.maximum(0, z1)

        z2 = self.target_W2 @ h1 + self.target_b2
        h2 = np.maximum(0, z2)

        out = self.target_W3 @ h2 + self.target_b3
        return float(out[0])

    def forward(self, x: np.ndarray) -> float:
        """
        Forward pass: Input → 256 ReLU → 128 ReLU → scalar.

        Args:
            x: Feature vector (n_features,) or larger

        Returns:
            Scalar value prediction
        """
        # Handle variable input size via padding/truncation
        if len(x) > self.n_features:
            x = x[:self.n_features]
        elif len(x) < self.n_features:
            x = np.pad(x, (0, self.n_features - len(x)), mode='constant')

        self._last_x = x

        # First hidden layer
        z1 = self.W1 @ x + self.b1
        self._last_h1 = np.maximum(0, z1)

        # Second hidden layer
        z2 = self.W2 @ self._last_h1 + self.b2
        self._last_h2 = np.maximum(0, z2)

        # Output layer
        out = self.W3 @ self._last_h2 + self.b3
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
        if explore and self.rng.rand() < self.epsilon:
            return valid_actions[self.rng.randint(len(valid_actions))]
        
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
                third_samples = self.rng.choice(
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
    
    def record_step(self, features: np.ndarray, reward: float, next_features_list: List[np.ndarray], done: bool):
        """
        Record a step in the replay buffer.
        
        Args:
            features: Feature vector for the action taken
            reward: Immediate reward received after transition
            next_features_list: List of feature vectors for all available next actions
            done: Whether the episode ended
        """
        self.replay_buffer.push(features.copy(), reward, next_features_list, done)
        self.step_counter += 1
        
        # Train periodically
        if self.step_counter % self.config.train_freq == 0:
            self._train_batch()
            
        # Update target network
        if self.step_counter % self.config.target_update_freq == 0:
            self.sync_target_network()
    
    def end_episode(self, final_score: float):
        """End episode updates (only house-keeping, training happens during record_step)."""
        self.episode_count += 1
        self.recent_scores.append(final_score)
        if len(self.recent_scores) > 20:
            self.recent_scores.pop(0)

        if final_score > self.best_score:
            self.best_score = final_score

        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
    
    def _train_batch(self):
        """Train 2-layer network on a batch from replay buffer doing Q-learning."""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        batch = self.replay_buffer.sample(self.config.batch_size)
        total_loss = 0.0

        for x, r, next_x_list, done in batch:
            # Calculate target via Bellman equation
            if done:
                target = r
            else:
                if not next_x_list:
                    target = r - 10.0  # Penalize terminal dead ends
                else:
                    max_q = max(self.forward_target(nx) for nx in next_x_list)
                    target = r + self.config.gamma * max_q

            # Forward pass to cache activations (_last_x, _last_h1, etc)
            pred = self.forward(x)

            error = pred - target
            clipped_error = np.clip(error, -5, 5)
            total_loss += error ** 2

            # Backprop through output layer (W3, b3)
            self.W3 -= self.config.learning_rate * clipped_error * self._last_h2
            self.b3 -= self.config.learning_rate * clipped_error

            # Gradient into h2 (second hidden layer)
            dh2 = clipped_error * self.W3[0]  # shape (n_hidden2,)
            dh2 *= (self._last_h2 > 0)        # ReLU derivative

            # Backprop second hidden layer (W2, b2)
            self.W2 -= self.config.learning_rate * np.outer(dh2, self._last_h1)
            self.b2 -= self.config.learning_rate * dh2

            # Gradient into h1 (first hidden layer)
            dh1 = self.W2.T @ dh2             # shape (n_hidden,)
            dh1 *= (self._last_h1 > 0)        # ReLU derivative

            # Backprop first hidden layer (W1, b1)
            self.W1 -= self.config.learning_rate * np.outer(dh1, self._last_x)
            self.b1 -= self.config.learning_rate * dh1

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
            W3=self.W3,
            b3=self.b3,
            n_features=self.n_features,
            config_dict={
                'n_hidden': self.config.n_hidden,
                'n_hidden2': self.config.n_hidden2,
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

        if data['n_features'] != self.n_features:
            raise ValueError(
                f"Feature mismatch: saved={data['n_features']}, "
                f"current={self.n_features}"
            )

        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']

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
                self.W2.size + self.b2.size +
                self.W3.size + self.b3.size
            )
        }
    def reset_episode(self):
        """Reset episode-specific state (called at episode start)."""
        pass
