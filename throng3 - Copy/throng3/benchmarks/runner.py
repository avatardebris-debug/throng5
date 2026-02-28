"""Benchmark runner for training and evaluating transfer learning."""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple
from collections import deque

from throng3.pipeline import MetaNPipeline
from throng3.environments.adapter import EnvironmentAdapter


class TrainResult:
    """Results from a training run."""
    
    def __init__(self):
        self.steps_to_convergence: Optional[int] = None
        self.final_loss: float = float('inf')
        self.converged: bool = False
        self.loss_history: list = []
        self.reward_history: list = []
        self.episode_returns: list = []
        
    def __repr__(self):
        return (f"TrainResult(converged={self.converged}, "
                f"steps={self.steps_to_convergence}, "
                f"final_loss={self.final_loss:.4f})")


class BenchmarkRunner:
    """
    Runs training experiments and measures convergence.
    
    Handles:
    - Training until convergence or timeout
    - Checkpoint save/load for transfer learning
    - Convergence detection with rolling average
    """
    
    def __init__(self, pipeline: MetaNPipeline, env: EnvironmentAdapter):
        """
        Initialize benchmark runner.
        
        Args:
            pipeline: MetaNPipeline to train
            env: Environment adapter to train on
        """
        self.pipeline = pipeline
        self.env = env
        
    def train_until_convergence(
        self,
        max_steps: int = 5000,
        convergence_threshold: float = 0.1,
        convergence_window: int = 10,
        verbose: bool = False
    ) -> TrainResult:
        """
        Train until loss converges or max_steps reached.
        
        Args:
            max_steps: Maximum training steps before timeout
            convergence_threshold: Loss threshold for convergence
            convergence_window: Number of steps to average for convergence check
            verbose: Whether to print progress
            
        Returns:
            TrainResult with convergence info and metrics
        """
        result = TrainResult()
        loss_window = deque(maxlen=convergence_window)
        
        obs = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        recent_episode_returns = deque(maxlen=10)  # Track recent episodes
        
        for step in range(max_steps):
            # Get action from pipeline
            pipeline_result = self.pipeline.step(obs)
            action_probs = pipeline_result.get('output', np.zeros(4))
            
            # Select action (argmax for now, could add exploration)
            action = int(np.argmax(action_probs))
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Compute average episode return for loss calculation
            avg_episode_return = np.mean(recent_episode_returns) if recent_episode_returns else -100.0
            
            # Train pipeline with reward signal AND episode return context
            train_result = self.pipeline.step(
                next_obs, 
                reward=reward,
                episode_return=avg_episode_return
            )
            loss = train_result.get('loss', 1.0)
            
            # Track metrics
            result.loss_history.append(loss)
            result.reward_history.append(reward)
            loss_window.append(loss)
            
            # Handle episode end
            if done:
                result.episode_returns.append(episode_reward)
                recent_episode_returns.append(episode_reward)
                obs = self.env.reset()
                episode_reward = 0.0
                episode_steps = 0
            else:
                obs = next_obs
            
            # Check convergence (only after filling window)
            if len(loss_window) == convergence_window:
                avg_loss = np.mean(loss_window)
                
                if avg_loss < convergence_threshold:
                    result.converged = True
                    result.steps_to_convergence = step + 1
                    result.final_loss = avg_loss
                    
                    if verbose:
                        print(f"✓ Converged at step {step + 1} "
                              f"(avg loss: {avg_loss:.4f})")
                    break
            
            # Progress logging
            if verbose and (step + 1) % 500 == 0:
                avg_loss = np.mean(loss_window) if loss_window else float('inf')
                avg_return = np.mean(recent_episode_returns) if recent_episode_returns else 0
                print(f"  Step {step + 1}/{max_steps}, "
                      f"avg loss: {avg_loss:.4f}, "
                      f"avg return: {avg_return:.2f}, "
                      f"episodes: {len(result.episode_returns)}")
        
        # If didn't converge, record final state
        if not result.converged:
            result.steps_to_convergence = max_steps
            result.final_loss = np.mean(loss_window) if loss_window else float('inf')
            
            if verbose:
                print(f"✗ Did not converge after {max_steps} steps "
                      f"(final avg loss: {result.final_loss:.4f})")
        
        return result
    
    def measure_steps_to_convergence(
        self,
        max_steps: int = 5000,
        convergence_threshold: float = 0.1,
        convergence_window: int = 10
    ) -> int:
        """
        Measure steps needed to reach convergence.
        
        Returns:
            Number of steps to convergence, or max_steps if didn't converge
        """
        result = self.train_until_convergence(
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            convergence_window=convergence_window,
            verbose=False
        )
        return result.steps_to_convergence or max_steps
    
    def save_checkpoint(self, path: str) -> None:
        """
        Save pipeline state for transfer learning.
        
        Args:
            path: File path to save checkpoint
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Get pipeline state
        state = self.pipeline.save_state()
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_checkpoint(self, path: str) -> None:
        """
        Restore pipeline state from checkpoint.
        
        Args:
            path: File path to load checkpoint from
            
        Note: Full restoration not yet implemented - this is a placeholder
        for future transfer learning functionality.
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        # TODO: Implement full state restoration when FractalStack
        # has restore_state() method
        pass
    
    def validate_checkpoint(self, path: str) -> bool:
        """
        Validate checkpoint can be saved and loaded correctly.
        
        Args:
            path: Temporary path for validation
            
        Returns:
            True if round-trip successful
        """
        try:
            # Save current state
            self.save_checkpoint(path)
            
            # Verify file exists and can be loaded
            if not os.path.exists(path):
                return False
            
            # Verify it's valid pickle data
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Basic validation
            if 'stack' not in state:
                return False
            
            # Clean up temp file
            if os.path.exists(path):
                os.remove(path)
            
            return True
            
        except Exception as e:
            print(f"Checkpoint validation failed: {e}")
            return False
