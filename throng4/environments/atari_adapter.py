"""
Atari Adapter — Bridges Gymnasium ALE environments to PortableNNAgent.

Provides:
- Wrapping for gym's ALE (Arcade Learning Environment)
- Raw RAM state extraction for fast NN training
- Action space mapping
"""

import numpy as np
import gymnasium as gym
from typing import List, Tuple, Dict, Any, Optional
from throng4.environments.adapter import EnvironmentAdapter
import ale_py

# Register ALE environments
gym.register_envs(ale_py)

class AtariAdapter(EnvironmentAdapter):
    """
    Adapter for Atari environments from Gymnasium.
    
    Default behavior uses obs_type="ram" which returns a 128-byte array.
    This is highly optimized for fast NN training while still being rich
    enough to establish baseline performance.
    """
    
    def __init__(self, game_id: str = "ALE/Breakout-v5", 
                 render_mode: Optional[str] = None,
                 seed: Optional[int] = None):
        """
        Initialize Atari adapter.
        
        Args:
            game_id: Gymnasium environment ID (e.g. ALE/Breakout-v5, ALE/Pong-v5)
            render_mode: Either None or "human"
            seed: Random seed
        """
        super().__init__()
        self.game_id = game_id
        
        # We use RAM observation type to get a 1D vector of 128 floats directly
        self.env = gym.make(self.game_id, obs_type="ram", render_mode=render_mode)
        
        # RAM state is always 128 bytes
        self.n_features = 128
        
        # Extract discrete action space
        self.num_actions = self.env.action_space.n
        self.valid_actions = list(range(self.num_actions))
        
        # Current state
        self._current_obs = None
        self.done = False
        
        if seed is not None:
            self._initial_seed = seed
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment.
        """
        obs, info = self.env.reset(seed=seed)
        self._raw_ram = obs
        self._current_obs = self.preprocess_obs(obs)
        self.done = False
        
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        return self._current_obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action index
            
        Returns:
            (state, reward, done, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._raw_ram = obs
        
        # Combine terminated and truncated into a single 'done'
        self.done = terminated or truncated
        self._current_obs = self.preprocess_obs(obs)
        
        self.episode_steps += 1
        self.episode_reward += reward
        
        if self.done:
            self.total_episodes += 1
            
        return self._current_obs, float(reward), self.done, info
    
    def preprocess_obs(self, obs: Any) -> np.ndarray:
        """
        Preprocess RAM observation (0-255) to normalized features (0.0-1.0).
        """
        # Gymnasium RAM obs is a uint8 array of length 128
        normalized = np.array(obs, dtype=np.float32) / 255.0
        return normalized
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices."""
        return self.valid_actions if not self.done else []
    
    def make_features(self, action: int) -> np.ndarray:
        """
        Create feature vector.
        
        Unlike Tetris where 'make_features' parameterizes the board state
        by action (Q(s,a) natively), in standard MDPs like Atari, the feature 
        is simply the current state. The NN handles action selection via 
        multiple outputs or we concatenate the action to the state.
        
        However, PortableNNAgent is a single-output value network V(phi(s,a)).
        To make it compatible with Atari without changing its architecture,
        we concatenate the normalized state and a one-hot action vector.
        """
        if self._current_obs is None:
            return np.zeros(self.n_features + self.num_actions, dtype=np.float32)
            
        state_features = self._current_obs
        
        # One-hot encode the action
        action_hot = np.zeros(self.num_actions, dtype=np.float32)
        if 0 <= action < self.num_actions:
            action_hot[action] = 1.0
            
        # phi(s, a) = [state, one_hot_action]
        return np.concatenate([state_features, action_hot])

    def get_semantic_obs(self, action: int, reward: float) -> str:
        """
        Extract a human/LLM-readable representation of the RAM state and action.
        Currently optimized for Breakout to allow LLM diagnosis.
        """
        if not hasattr(self, '_raw_ram') or self._raw_ram is None:
            return "State: Unknown"
            
        action_names = ["No-Op", "Fire", "Right", "Left"]
        act_str = action_names[action] if 0 <= action < len(action_names) else f"Action_{action}"
        
        # Extracted Breakout Semantics
        if "Breakout" in self.game_id:
            paddle_x = self._raw_ram[72]
            ball_x = self._raw_ram[99]
            ball_y = self._raw_ram[101]
            lives = self._raw_ram[57]
            blocks = self._raw_ram[14] # Sometimes score/blocks
            
            return (f"Step {self.episode_steps:03d} | Action: {act_str: <6} | "
                    f"Paddle_X: {paddle_x:03d}, Ball: ({ball_x:03d}, {ball_y:03d}) | "
                    f"Reward: {reward} | Lives: {lives}")
                    
        # Fallback for other games: chunk of changing RAM or just numbers
        return f"Step {self.episode_steps} | Action: {act_str} | Reward: {reward} | RAM Hex: {self._raw_ram[:10].hex()}"

    # ------------------------------------------------------------------
    # Abstract Feature Protocol
    # ------------------------------------------------------------------

    def get_core_features(self) -> np.ndarray:
        """
        Map current Atari RAM state to the universal 20-dim core vector.
        Breakout-specific RAM indices; other games fall back to safe defaults.
        """
        from throng4.learning.abstract_features import (
            empty_core, IDX_AGENT_X, IDX_AGENT_Y, IDX_TARGET_X, IDX_TARGET_Y,
            IDX_THREAT_X, IDX_THREAT_Y, IDX_THREAT_PROX, IDX_REWARD_PROX,
            IDX_TARGET_VX, IDX_TARGET_VY, IDX_RESOURCE, IDX_DENSITY, IDX_EPISODE_PROG
        )
        core = empty_core()

        if not hasattr(self, '_raw_ram') or self._raw_ram is None:
            return core

        ram = self._raw_ram

        if "Breakout" in self.game_id:
            # Screen is ~160x210 pixels; normalize to [0,1]
            W, H = 160.0, 210.0
            paddle_x = ram[72] / W
            ball_x   = ram[99] / W
            ball_y   = ram[101] / H
            lives    = ram[57]

            # Velocity approximation: diff from last obs (we store _prev_ram)
            prev = getattr(self, '_prev_ram', ram)
            tvx = np.clip((int(ram[99]) - int(prev[99])) / W, -1, 1)
            tvy = np.clip((int(ram[101]) - int(prev[101])) / H, -1, 1)

            core[IDX_AGENT_X]     = paddle_x
            core[IDX_AGENT_Y]     = 0.95          # paddle is at bottom
            core[IDX_TARGET_X]    = ball_x
            core[IDX_TARGET_Y]    = ball_y
            core[IDX_THREAT_X]    = ball_x         # ball is also the threat
            core[IDX_THREAT_Y]    = ball_y
            # threat_prox: how close ball is to bottom (paddle zone)
            core[IDX_THREAT_PROX] = ball_y         # 0=top (safe), 1=bottom (danger)
            core[IDX_REWARD_PROX] = 1.0 - ball_y  # bricks are at top
            core[IDX_TARGET_VX]   = tvx
            core[IDX_TARGET_VY]   = tvy
            core[IDX_RESOURCE]    = min(lives, 5) / 5.0
            core[IDX_EPISODE_PROG] = min(self.episode_steps / 1000.0, 1.0)

            # Density: use fraction of RAM bytes that changed (crude activity proxy)
            changed = np.sum(ram != prev) / 128.0
            core[IDX_DENSITY] = changed

        # Store previous RAM for velocity estimation
        self._prev_ram = ram.copy()
        return core

    def get_ext_features(self):
        """
        Breakout-specific extension block (up to EXT_MAX slots).
        Slots: [fine_ball_x, fine_ball_y, fine_paddle_x, brick_activity, ...]
        """
        from throng4.learning.abstract_features import make_ext
        if not hasattr(self, '_raw_ram') or self._raw_ram is None:
            from throng4.learning.abstract_features import EXT_MAX
            import numpy as np
            return np.zeros(EXT_MAX, np.float32), np.zeros(EXT_MAX, np.float32)

        ram = self._raw_ram
        if "Breakout" in self.game_id:
            W, H = 160.0, 210.0
            # Finer (unnormalized) positions that might help edge-case decisions
            fine_ball_x   = ram[99] / W
            fine_ball_y   = ram[101] / H
            fine_paddle_x = ram[72] / W
            # Paddle-ball horizontal offset (signed), useful alignment signal
            align_offset  = (fine_ball_x - fine_paddle_x + 1.0) / 2.0
            # RAM byte 14 loosely correlates with remaining bricks
            brick_proxy   = ram[14] / 255.0

            return make_ext([fine_ball_x, fine_ball_y, fine_paddle_x,
                             align_offset, brick_proxy])

        from throng4.learning.abstract_features import EXT_MAX
        import numpy as np
        return np.zeros(EXT_MAX, np.float32), np.zeros(EXT_MAX, np.float32)

    def get_lookahead_actions(self, action: int) -> List[int]:
        """
        Atari is not a perfect information board game, and we can't trivially 
        simulate RAM state branches without an emulator reset/save-state hook.
        So we disable lookahead.
        """
        return []

    def get_info(self) -> Dict[str, Any]:
        """Get episode summary info."""
        return {
            'episode_reward': self.episode_reward,
            'episode_steps': self.episode_steps
        }

