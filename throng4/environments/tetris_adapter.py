"""
Tetris Adapter — Bridges TetrisCurriculumEnv to PortableNNAgent.

Provides:
- Feature extraction matching the HTML version (boardW + 12 features)
- Action enumeration (valid placements)
- Lookahead support for multi-move planning
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv
from throng4.environments.adapter import EnvironmentAdapter


class TetrisAdapter(EnvironmentAdapter):
    """
    Adapter for Tetris environment compatible with PortableNNAgent.
    
    Feature vector (boardW + 12):
        [0-3]:   aggH, holes, bump, maxH (normalized)
        [4:4+W]: per-column heights (normalized)
        [4+W]:   lines cleared (normalized)
        [5+W]:   placement row (normalized)
        [6+W]:   placement col (normalized)
        [7+W]:   placement rotation (normalized)
        [8+W]:   piece type I (one-hot)
        [9+W]:   piece type O (one-hot)
        [10+W]:  piece type T/S/Z (one-hot)
        [11+W]:  row completeness (avg fill fraction)
    """
    
    def __init__(self, level: int = 1, max_pieces: int = 500):
        """
        Initialize Tetris adapter.
        
        Args:
            level: Curriculum level (1-7)
            max_pieces: Max pieces per episode
        """
        super().__init__()
        self.env = TetrisCurriculumEnv(level=level, max_pieces=max_pieces)
        self.level = level
        self.max_pieces = max_pieces
        
        # Feature dimensions
        self.board_width = self.env.width
        self.n_features = self.board_width + 12
        
        # Current state
        self.current_piece = None
        self.valid_actions = []
        self.done = False
    
    def reset(self) -> np.ndarray:
        """
        Reset environment.
        
        Returns:
            Initial state (not used by PortableNNAgent, which uses features)
        """
        state = self.env.reset()
        self.current_piece = self.env.current_piece
        self.valid_actions = self.env.get_valid_actions()
        self.done = False
        
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        return state
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: (rotation, column) placement
            
        Returns:
            (state, reward, done, info)
        """
        state, reward, done, info = self.env.step(action)
        
        self.episode_steps += 1
        self.episode_reward += reward
        self.done = done
        
        if not done:
            self.current_piece = self.env.current_piece
            self.valid_actions = self.env.get_valid_actions()
        else:
            self.total_episodes += 1
        
        return state, reward, done, info
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """Get list of valid (rotation, column) placements."""
        return self.valid_actions
    
    def make_features(self, action: Tuple[int, int]) -> np.ndarray:
        """
        Create feature vector for a placement action.
        
        This matches the HTML version's expanded feature set.
        
        Args:
            action: (rotation, column) placement
            
        Returns:
            Feature vector of size (boardW + 12,)
        """
        rotation, column = action
        
        # Simulate placement to get resulting board state
        board_copy = [row[:] for row in self.env.board]
        piece_cells = self.env._get_piece_cells(rotation)
        
        # Find drop row
        row = 0
        while self.env._can_place_at(row + 1, column, rotation):
            row += 1
        
        # Place piece on copy
        for dr, dc in piece_cells:
            r, c = row + dr, column + dc
            if 0 <= r < self.env.height and 0 <= c < self.env.width:
                board_copy[r][c] = 1.0  # TetrisCurriculumEnv uses float board
        
        # Count cleared lines
        cleared = sum(1 for r in board_copy if all(cell is not None for cell in r))
        
        # Compute board features
        features_dict = self._compute_board_features(board_copy)
        
        # Build feature vector
        features = []
        
        # [0-3]: Normalized heuristics
        features.append(features_dict['agg_height'] / (self.board_width * self.env.height))
        features.append(features_dict['holes'] / (self.board_width * self.env.height * 0.5))
        features.append(features_dict['bumpiness'] / (self.env.height * (self.board_width - 1)))
        features.append(features_dict['max_height'] / self.env.height)
        
        # [4:4+W]: Per-column heights
        for h in features_dict['heights']:
            features.append(h / self.env.height)
        
        # [4+W]: Lines cleared
        features.append(cleared / 4.0)
        
        # [5+W]: Placement row
        features.append(row / self.env.height)
        
        # [6+W]: Placement column
        features.append(column / self.board_width)
        
        # [7+W]: Rotation
        features.append(rotation / 4.0)
        
        # [8+W, 9+W, 10+W]: Piece type (one-hot-ish)
        piece_idx = ['I', 'O', 'T', 'S', 'Z', 'L', 'J'].index(self.current_piece)
        features.append(1.0 if piece_idx == 0 else 0.0)  # I
        features.append(1.0 if piece_idx == 1 else 0.0)  # O
        features.append(1.0 if 2 <= piece_idx <= 4 else 0.0)  # T/S/Z
        
        # [11+W]: Row completeness
        features.append(features_dict['avg_completeness'])
        
        return np.array(features, dtype=np.float32)
    
    def _compute_board_features(self, board: List[List]) -> Dict[str, Any]:
        """
        Compute Dellacherie-style board features.
        
        Args:
            board: Board state (height × width)
            
        Returns:
            Dict with agg_height, holes, bumpiness, max_height, heights, avg_completeness
        """
        # Column heights
        heights = []
        for c in range(self.board_width):
            h = 0
            for r in range(self.env.height):
                if board[r][c] is not None:
                    h = self.env.height - r
                    break
            heights.append(h)
        
        agg_height = sum(heights)
        max_height = max(heights)
        
        # Holes
        holes = 0
        for c in range(self.board_width):
            found_block = False
            for r in range(self.env.height):
                if board[r][c] is not None:
                    found_block = True
                elif found_block:
                    holes += 1
        
        # Bumpiness
        bumpiness = sum(abs(heights[i] - heights[i+1]) 
                       for i in range(self.board_width - 1))
        
        # Row completeness
        occupied_rows = 0
        total_fill = 0.0
        for r in range(self.env.height):
            filled = sum(1 for cell in board[r] if cell is not None)
            if filled > 0:
                occupied_rows += 1
                total_fill += filled / self.board_width
        
        avg_completeness = total_fill / occupied_rows if occupied_rows > 0 else 0.0
        
        return {
            'agg_height': agg_height,
            'holes': holes,
            'bumpiness': bumpiness,
            'max_height': max_height,
            'heights': heights,
            'avg_completeness': avg_completeness
        }
    
    def get_lookahead_actions(self, action: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid actions for the next piece after placing current action.
        
        This enables multi-move lookahead in PortableNNAgent.
        
        Args:
            action: Current (rotation, column) placement
            
        Returns:
            List of valid actions for next piece
        """
        # We'd need to simulate the placement and get next piece's valid actions
        # For now, return empty list (lookahead disabled)
        # TODO: Implement full simulation if lookahead is critical
        return []
    
    def preprocess_obs(self, obs: Any) -> np.ndarray:
        """
        Convert raw observation to normalized 1D array.
        
        Note: PortableNNAgent doesn't use this — it uses make_features instead.
        This is here for EnvironmentAdapter compatibility.
        """
        return self.flatten(obs)
    
    def get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        return {
            'level': self.level,
            'board_width': self.board_width,
            'board_height': self.env.height,
            'pieces_placed': self.env.pieces_placed,
            'lines_cleared': self.env.lines_cleared,
            'current_piece': self.current_piece,
            'n_valid_actions': len(self.valid_actions)
        }
