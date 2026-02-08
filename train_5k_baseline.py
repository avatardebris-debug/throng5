"""
Improved baseline with 5000 neurons and epsilon-greedy exploration.
"""

import numpy as np
from typing import Tuple
from collections import deque
from simple_baseline import SimplePolicyNetwork


class ImprovedPolicyNetwork(SimplePolicyNetwork):
    """Enhanced policy network with exploration."""
    
    def __init__(self, n_inputs: int, n_hidden: int, n_outputs: int, 
                 lr: float = 0.001, epsilon_start: float = 0.3, epsilon_end: float = 0.01):
        super().__init__(n_inputs, n_hidden, n_outputs, lr)
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = 0.995  # Decay per episode
        
    def forward(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, int]:
        """Forward pass with epsilon-greedy exploration."""
        # Get action probabilities from network
        h = np.tanh(state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits))
        action_probs = exp_logits / np.sum(exp_logits)
        
        if training:
            # Epsilon-greedy: explore with probability epsilon
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.n_outputs)
            else:
                action = np.random.choice(self.n_outputs, p=action_probs)
            
            # Store for gradient computation
            self.states.append(state)
            self.actions.append(action)
            self.hidden_states.append(h)
        else:
            # Exploitation only
            action = np.argmax(action_probs)
        
        return action_probs, action
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_improved_baseline(env, n_episodes: int = 500, n_hidden: int = 5000, verbose: bool = True):
    """
    Train improved baseline with exploration.
    
    Args:
        env: Environment adapter
        n_episodes: Number of episodes to train
        n_hidden: Number of hidden neurons
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
    network = ImprovedPolicyNetwork(
        n_inputs=n_inputs,
        n_hidden=n_hidden,
        n_outputs=n_outputs,
        lr=0.001,
        epsilon_start=0.3,
        epsilon_end=0.01
    )
    
    episode_returns = []
    recent_returns = deque(maxlen=20)
    
    if verbose:
        print(f"\nTraining improved baseline:")
        print(f"  Input dim: {n_inputs}")
        print(f"  Hidden dim: {n_hidden}")
        print(f"  Output dim: {n_outputs}")
        print(f"  Episodes: {n_episodes}")
        print(f"  Exploration: ε={network.epsilon_start:.2f} → {network.epsilon_end:.2f}\n")
    
    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            # Get action with exploration
            _, action = network.forward(obs, training=True)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Store reward
            network.store_reward(reward)
        
        # Update network
        network.update()
        network.decay_epsilon()
        
        episode_returns.append(episode_reward)
        recent_returns.append(episode_reward)
        
        if verbose and (episode + 1) % 50 == 0:
            avg_return = np.mean(recent_returns)
            print(f"  Episode {episode+1}/{n_episodes}: "
                  f"return={episode_reward:.2f}, "
                  f"avg_return={avg_return:.2f}, "
                  f"ε={network.epsilon:.3f}, "
                  f"steps={steps}")
    
    return episode_returns, network


if __name__ == "__main__":
    from throng3.environments import GridWorldAdapter
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("5000 Neuron Baseline Test")
    print("="*60)
    
    # Train on GridWorld
    env = GridWorldAdapter()
    returns, network = train_improved_baseline(env, n_episodes=500, n_hidden=5000)
    
    # Evaluate
    print(f"\n{'='*60}")
    print("Evaluation (50 episodes, no exploration):")
    print(f"{'='*60}")
    
    successes = 0
    eval_returns = []
    
    for ep in range(50):
        obs = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 200:
            _, action = network.forward(obs, training=False)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
        
        eval_returns.append(episode_reward)
        if env.pos == env.goal:
            successes += 1
    
    print(f"\nEvaluation results:")
    print(f"  Success rate: {successes}/50 = {successes*2}%")
    print(f"  Avg return: {np.mean(eval_returns):.3f}")
    
    # Training progress
    early = np.mean(returns[:50])
    late = np.mean(returns[-50:])
    
    print(f"\nTraining progress:")
    print(f"  Early (1-50): {early:.3f}")
    print(f"  Late (451-500): {late:.3f}")
    print(f"  Improvement: {late - early:+.3f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(returns, alpha=0.3)
    
    # Moving average
    window = 20
    moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(returns)), moving_avg, linewidth=2, label='Moving avg (20 ep)')
    
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('GridWorld Learning (5000 neurons)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/gridworld_5k_learning.png', dpi=150)
    print(f"\n✓ Saved learning curve to results/gridworld_5k_learning.png")
    
    if successes >= 30:
        print(f"\n✓ SUCCESS: 5K neuron network learned GridWorld!")
    elif successes >= 10:
        print(f"\n⚠ PARTIAL: Some learning, needs more training")
    else:
        print(f"\n✗ FAILURE: Network did not learn sufficiently")
    
    print("="*60)
