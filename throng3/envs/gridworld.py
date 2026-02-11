"""
GridWorld Environment for Testing Compound Transfer

Simple 2D grid navigation with compositional structure:
- Spatial reasoning (navigation)
- Obstacle avoidance
- Goal-seeking

Variants test transfer of shared abstractions.
"""

import numpy as np
from typing import Tuple, List, Optional


class GridWorld:
    """
    2D grid navigation environment.
    
    Agent starts at (0, 0) and must reach the goal.
    Actions: 0=up, 1=down, 2=left, 3=right
    """
    
    def __init__(self, size: int = 5, 
                 obstacles: Optional[List[Tuple[int, int]]] = None,
                 goal: Optional[Tuple[int, int]] = None):
        """
        Initialize GridWorld.
        
        Args:
            size: Grid size (size x size)
            obstacles: List of (row, col) obstacle positions
            goal: (row, col) goal position (default: bottom-right)
        """
        self.size = size
        self.obstacles = obstacles or []
        self.goal = goal or (size - 1, size - 1)
        self.pos = (0, 0)
        self.steps = 0
        self.max_steps = size * size * 2  # Prevent infinite loops
        self.reward_shaping = True  # Distance-based reward shaping
    
    def _manhattan_distance(self, pos: Tuple[int, int]) -> int:
        """Manhattan distance from pos to goal."""
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])
    
    def reset(self) -> np.ndarray:
        """Reset to starting position."""
        self.pos = (0, 0)
        self.steps = 0
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state as one-hot encoded position.
        
        Returns:
            1D array of length size*size with 1 at current position
        """
        state = np.zeros(self.size * self.size, dtype=np.float32)
        idx = self.pos[0] * self.size + self.pos[1]
        state[idx] = 1.0
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take action and return (state, reward, done).
        
        Args:
            action: 0=up, 1=down, 2=left, 3=right
            
        Returns:
            (next_state, reward, done)
        """
        self.steps += 1
        prev_distance = self._manhattan_distance(self.pos)
        
        # Compute new position
        row, col = self.pos
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        
        new_pos = (row, col)
        
        # Initialize done flag
        done = False
        
        # Check obstacles
        if new_pos in self.obstacles:
            reward = -0.1  # Penalty for hitting obstacle
            new_pos = self.pos  # Stay in place
        elif new_pos == self.goal:
            reward = 1.0  # Goal reached!
            done = True
        else:
            reward = -0.01  # Small step penalty (encourages efficiency)
        
        # Distance-based reward shaping (gives gradient toward goal)
        if self.reward_shaping and not done:
            curr_distance = self._manhattan_distance(new_pos)
            # +0.1 for getting closer, -0.1 for getting farther
            reward += 0.1 * (prev_distance - curr_distance)
        
        # Check max steps
        if self.steps >= self.max_steps:
            done = True
        
        self.pos = new_pos
        return self.get_state(), reward, done
    
    def render(self) -> str:
        """Render grid as ASCII art."""
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                if (r, c) == self.pos:
                    row.append('A')  # Agent
                elif (r, c) == self.goal:
                    row.append('G')  # Goal
                elif (r, c) in self.obstacles:
                    row.append('#')  # Obstacle
                else:
                    row.append('.')  # Empty
            lines.append(' '.join(row))
        return '\n'.join(lines)


def create_gridworld_variants():
    """Create standard GridWorld task variants for transfer learning."""
    
    # Task A: Empty 5x5 grid
    env_a = GridWorld(size=5, obstacles=[], goal=(4, 4))
    
    # Task B: 5x5 with obstacles (wall in middle)
    env_b = GridWorld(
        size=5, 
        obstacles=[(2, 1), (2, 2), (2, 3)],  # Horizontal wall
        goal=(4, 4)
    )
    
    # Task C: 7x7 empty (larger grid)
    env_c = GridWorld(size=7, obstacles=[], goal=(6, 6))
    
    # Task D: 5x5 with different obstacle pattern
    env_d = GridWorld(
        size=5,
        obstacles=[(1, 2), (2, 2), (3, 2)],  # Vertical wall
        goal=(4, 4)
    )
    
    return {
        'empty_5x5': env_a,
        'obstacles_5x5': env_b,
        'empty_7x7': env_c,
        'obstacles_v_5x5': env_d,
    }


if __name__ == '__main__':
    # Test GridWorld
    print("Testing GridWorld Environment\n")
    
    env = GridWorld(size=5, obstacles=[(2, 2)], goal=(4, 4))
    
    print("Initial state:")
    print(env.render())
    print(f"State vector shape: {env.get_state().shape}")
    print()
    
    # Random episode
    state = env.reset()
    done = False
    total_reward = 0
    
    print("Random episode:")
    for step in range(20):
        action = np.random.randint(4)
        state, reward, done = env.step(action)
        total_reward += reward
        
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        print(f"Step {step+1}: {action_names[action]}, Reward: {reward:.2f}")
        print(env.render())
        print()
        
        if done:
            print(f"Episode finished! Total reward: {total_reward:.2f}")
            break
    
    # Test variants
    print("\n" + "="*50)
    print("GridWorld Variants:")
    print("="*50)
    
    variants = create_gridworld_variants()
    for name, env in variants.items():
        print(f"\n{name}:")
        print(env.render())
