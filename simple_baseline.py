"""
Simple policy gradient baseline to verify environment works.
Uses basic 2-layer network with actual gradient descent.
"""

import numpy as np
from typing import Tuple, List
from collections import deque


class SimplePolicyNetwork:
    """Simple 2-layer feedforward network with policy gradient learning."""
    
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, lr: float = 0.01):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.lr = lr
        
        # Xavier initialization
        self.W1 = np.random.randn(n_inputs, n_hidden) / np.sqrt(n_inputs)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_outputs) / np.sqrt(n_hidden)
        self.b2 = np.zeros(n_outputs)
        
        # Episode memory for policy gradient
        self.states = []
        self.actions = []
        self.rewards = []
        self.hidden_states = []
        
    def forward(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, int]:
        """
        Forward pass through network.
        
        Returns:
            action_probs: Probability distribution over actions
            action: Sampled action (if training) or argmax (if not)
        """
        # Hidden layer
        h = np.tanh(state @ self.W1 + self.b1)
        
        # Output layer (logits)
        logits = h @ self.W2 + self.b2
        
        # Softmax for action probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        action_probs = exp_logits / np.sum(exp_logits)
        
        # Sample action (exploration) or take best (exploitation)
        if training:
            action = np.random.choice(self.n_outputs, p=action_probs)
            # Store for gradient computation
            self.states.append(state)
            self.actions.append(action)
            self.hidden_states.append(h)
        else:
            action = np.argmax(action_probs)
        
        return action_probs, action
    
    def store_reward(self, reward: float):
        """Store reward for current timestep."""
        self.rewards.append(reward)
    
    def update(self):
        """
        Update weights using policy gradient (REINFORCE).
        Called at end of episode.
        """
        if len(self.rewards) == 0:
            return
        
        # Compute discounted returns
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(self.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Normalize returns (reduces variance)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        
        # Compute gradients
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        
        for t in range(len(self.states)):
            state = self.states[t]
            action = self.actions[t]
            G = returns[t]
            h = self.hidden_states[t]
            
            # Forward pass to get action probs
            logits = h @ self.W2 + self.b2
            exp_logits = np.exp(logits - np.max(logits))
            action_probs = exp_logits / np.sum(exp_logits)
            
            # Gradient of log policy
            dlogits = action_probs.copy()
            dlogits[action] -= 1  # Gradient of log softmax
            
            # Backprop through output layer
            dW2 += np.outer(h, dlogits) * G
            db2 += dlogits * G
            
            # Backprop through hidden layer
            dh = (dlogits @ self.W2.T) * G
            dh = dh * (1 - h**2)  # tanh derivative
            
            dW1 += np.outer(state, dh)
            db1 += dh
        
        # Update weights
        self.W1 -= self.lr * dW1 / len(self.states)
        self.b1 -= self.lr * db1 / len(self.states)
        self.W2 -= self.lr * dW2 / len(self.states)
        self.b2 -= self.lr * db2 / len(self.states)
        
        # Clear episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.hidden_states = []
    
    def reset_episode(self):
        """Clear episode memory without updating."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.hidden_states = []


def train_simple_baseline(env, n_episodes: int = 100, verbose: bool = True):
    """
    Train simple baseline on environment.
    
    Args:
        env: Environment adapter (GridWorld or CartPole)
        n_episodes: Number of episodes to train
        verbose: Print progress
    
    Returns:
        episode_returns: List of episode returns
        network: Trained network
    """
    # Determine input/output sizes
    obs = env.reset()
    n_inputs = len(obs)
    
    # Assume discrete action space
    if hasattr(env, 'size'):  # GridWorld
        n_outputs = 4
    else:  # CartPole
        n_outputs = 2
    
    # Create network
    network = SimplePolicyNetwork(
        n_inputs=n_inputs,
        n_hidden=128,  # Larger hidden layer
        n_outputs=n_outputs,
        lr=0.001  # Learning rate
    )
    
    episode_returns = []
    recent_returns = deque(maxlen=10)
    
    if verbose:
        print(f"\nTraining simple baseline:")
        print(f"  Input dim: {n_inputs}")
        print(f"  Hidden dim: 128")
        print(f"  Output dim: {n_outputs}")
        print(f"  Episodes: {n_episodes}\n")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 500
        
        while not done and steps < max_steps:
            # Get action
            _, action = network.forward(obs, training=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Store reward
            network.store_reward(reward)
        
        # Update network at end of episode
        network.update()
        
        episode_returns.append(episode_reward)
        recent_returns.append(episode_reward)
        
        if verbose and (episode + 1) % 10 == 0:
            avg_return = np.mean(recent_returns)
            print(f"  Episode {episode+1}/{n_episodes}: "
                  f"return={episode_reward:.1f}, "
                  f"avg_return={avg_return:.1f}, "
                  f"steps={steps}")
    
    return episode_returns, network


if __name__ == "__main__":
    from throng3.environments import GridWorldAdapter, CartPoleAdapter
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("Simple Baseline Test")
    print("="*60)
    
    # Test on GridWorld
    print("\n--- GridWorld ---")
    env = GridWorldAdapter()
    returns_gw, net_gw = train_simple_baseline(env, n_episodes=50)
    
    print(f"\nGridWorld Results:")
    print(f"  Early returns (0-10): {np.mean(returns_gw[:10]):.2f}")
    print(f"  Late returns (40-50): {np.mean(returns_gw[40:]):.2f}")
    print(f"  Improvement: {np.mean(returns_gw[40:]) - np.mean(returns_gw[:10]):.2f}")
    
    # Test on CartPole
    print("\n--- CartPole ---")
    env = CartPoleAdapter()
    returns_cp, net_cp = train_simple_baseline(env, n_episodes=50)
    
    print(f"\nCartPole Results:")
    print(f"  Early returns (0-10): {np.mean(returns_cp[:10]):.2f}")
    print(f"  Late returns (40-50): {np.mean(returns_cp[40:]):.2f}")
    print(f"  Improvement: {np.mean(returns_cp[40:]) - np.mean(returns_cp[:10]):.2f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(returns_gw)
    plt.title('GridWorld Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(returns_cp)
    plt.title('CartPole Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/baseline_learning_curves.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved learning curves to results/baseline_learning_curves.png")
    
    print("\n" + "="*60)
    print("✓ Baseline test complete!")
    print("="*60)
