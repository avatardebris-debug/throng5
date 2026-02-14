"""
Tetris Curriculum — Placement-based action space with Dellacherie heuristics.

Architecture:
  - Action space: enumerate all valid (rotation, column) placements
  - Reward: R = a×(lines) - b×(agg_height) - c×(holes) - d×(bumpiness)
  - Curriculum: gradually shift reward from heuristic → raw lines cleared

Dellacherie weights (classic):
  Lines cleared:    +0.76
  Aggregate height: -0.51
  Holes:            -0.36
  Bumpiness:        -0.18
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field


# ─── Tetromino definitions ───────────────────────────────────────────

TETROMINOES = {
    'I': [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    'O': [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    'T': [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    'S': [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    'Z': [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    'L': [
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
    'J': [
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
}


@dataclass
class DellacherieWeights:
    """Heuristic reward weights."""
    lines_cleared: float = 0.76
    aggregate_height: float = -0.51
    holes: float = -0.36
    bumpiness: float = -0.18


class TetrisCurriculumEnv:
    """
    Tetris with placement-based action space and heuristic rewards.
    
    Action space:
        Each action is a (rotation, column) placement.
        The piece is dropped to the lowest valid row.
    
    State:
        Flattened board + piece one-hot + board features
    
    Reward:
        R = w_lines * lines + w_height * Δheight + w_holes * Δholes + w_bump * Δbump
    """
    
    LEVELS = {
        1: {
            'name': 'O-block only',
            'pieces': ['O'],
            'rotations': 1,
            'width': 6,
            'height': 10,
        },
        2: {
            'name': 'O + I blocks',
            'pieces': ['O', 'I'],
            'rotations': 2,
            'width': 6,
            'height': 12,
        },
        3: {
            'name': 'O + I + T blocks',
            'pieces': ['O', 'I', 'T'],
            'rotations': 4,
            'width': 6,
            'height': 12,
        },
        4: {
            'name': '+ S, Z blocks',
            'pieces': ['O', 'I', 'T', 'S', 'Z'],
            'rotations': 4,
            'width': 8,
            'height': 14,
        },
        5: {
            'name': 'All blocks, 8-wide',
            'pieces': ['I', 'O', 'T', 'S', 'Z', 'L', 'J'],
            'rotations': 4,
            'width': 8,
            'height': 16,
        },
        6: {
            'name': 'All blocks, 10-wide',
            'pieces': ['I', 'O', 'T', 'S', 'Z', 'L', 'J'],
            'rotations': 4,
            'width': 10,
            'height': 18,
        },
        7: {
            'name': 'Standard Tetris',
            'pieces': ['I', 'O', 'T', 'S', 'Z', 'L', 'J'],
            'rotations': 4,
            'width': 10,
            'height': 20,
        },
    }
    
    def __init__(self, level: int = 1, max_pieces: int = 500,
                 weights: Optional[DellacherieWeights] = None,
                 heuristic_blend: float = 1.0):
        """
        Args:
            level: Curriculum level (1-7)
            max_pieces: Max pieces per episode
            weights: Heuristic reward weights
            heuristic_blend: 1.0 = full heuristic, 0.0 = raw lines only
        """
        assert level in self.LEVELS, f"Level must be 1-7, got {level}"
        
        config = self.LEVELS[level]
        self.level = level
        self.level_name = config['name']
        self.piece_types = config['pieces']
        self.max_rotations = config['rotations']
        self.width = config['width']
        self.height = config['height']
        self.max_pieces = max_pieces
        self.weights = weights or DellacherieWeights()
        self.heuristic_blend = heuristic_blend  # Curriculum: reduce over time
        
        # State
        self.board = None
        self.current_piece = None
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.total_reward = 0.0
        self.game_over = False
        
        # State dimensions
        self.board_dim = self.width * self.height
        self.piece_dim = len(self.piece_types)
        self.feature_dim = 3  # agg_height, holes, bumpiness (normalized)
        self.state_dim = self.board_dim + self.piece_dim + self.feature_dim
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.board = np.zeros((self.height, self.width), dtype=np.float32)
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.total_reward = 0.0
        self.game_over = False
        self._spawn_piece()
        return self._get_state()
    
    def get_valid_actions(self) -> List[Tuple[int, int]]:
        """
        Enumerate all valid (rotation, column) placements.
        
        Returns:
            List of (rotation, column) tuples
        """
        actions = []
        for rot in range(self.max_rotations):
            cells = self._get_piece_cells(rot)
            min_c = min(dc for _, dc in cells)
            max_c = max(dc for _, dc in cells)
            
            for col in range(-min_c, self.width - max_c):
                # Find drop row
                row = 0
                while self._can_place_at(row + 1, col, rot):
                    row += 1
                
                if self._can_place_at(row, col, rot):
                    actions.append((rot, col))
        
        return actions
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Place piece at (rotation, column) and compute reward.
        
        Args:
            action: (rotation, column) tuple
        
        Returns:
            (state, reward, done, info)
        """
        rotation, col = action
        
        # Snapshot board before placement
        board_before = self.board.copy()
        h_before = self._compute_heuristics()
        
        # Find drop row
        row = 0
        while self._can_place_at(row + 1, col, rotation):
            row += 1
        
        # Place piece
        cells = self._get_piece_cells(rotation)
        for dr, dc in cells:
            r, c = row + dr, col + dc
            if 0 <= r < self.height and 0 <= c < self.width:
                self.board[r, c] = 1.0
        
        self.pieces_placed += 1
        
        # Clear lines
        lines_before = self.lines_cleared
        self._clear_lines()
        new_lines = self.lines_cleared - lines_before
        
        # Compute reward
        h_after = self._compute_heuristics()
        
        # Heuristic reward (delta-based)
        heuristic_reward = (
            self.weights.lines_cleared * new_lines
            + self.weights.aggregate_height * (h_after['agg_height'] - h_before['agg_height'])
            + self.weights.holes * (h_after['holes'] - h_before['holes'])
            + self.weights.bumpiness * (h_after['bumpiness'] - h_before['bumpiness'])
        )
        
        # Raw reward (just lines)
        raw_reward = new_lines
        
        # Blend
        reward = (self.heuristic_blend * heuristic_reward 
                 + (1.0 - self.heuristic_blend) * raw_reward)
        
        # Game over check
        done = False
        if np.any(self.board[0] > 0) or np.any(self.board[1] > 0):
            done = True
            reward -= 1.0  # Penalty for game over
        elif self.pieces_placed >= self.max_pieces:
            done = True
        
        self.game_over = done
        self.total_reward += reward
        
        # Spawn next piece
        if not done:
            self._spawn_piece()
            # Check if any placement is valid
            if len(self.get_valid_actions()) == 0:
                done = True
                self.game_over = True
        
        info = {
            'lines_cleared': self.lines_cleared,
            'new_lines': new_lines,
            'pieces_placed': self.pieces_placed,
            'level': self.level,
            'heuristics': h_after,
            'heuristic_reward': heuristic_reward,
            'raw_reward': raw_reward,
        }
        
        return self._get_state(), reward, done, info
    
    def _spawn_piece(self):
        """Spawn a random piece."""
        idx = np.random.randint(len(self.piece_types))
        self.current_piece = self.piece_types[idx]
    
    def _get_piece_cells(self, rotation: int) -> List[Tuple[int, int]]:
        """Get cell offsets for current piece."""
        rots = TETROMINOES[self.current_piece]
        return rots[rotation % len(rots)]
    
    def _can_place_at(self, row: int, col: int, rotation: int) -> bool:
        """Check if piece can be placed."""
        cells = self._get_piece_cells(rotation)
        for dr, dc in cells:
            r, c = row + dr, col + dc
            if r < 0 or r >= self.height or c < 0 or c >= self.width:
                return False
            if self.board[r, c] > 0:
                return False
        return True
    
    def _clear_lines(self) -> int:
        """Clear completed lines."""
        full_rows = np.where(np.all(self.board > 0, axis=1))[0]
        if len(full_rows) == 0:
            return 0
        
        mask = np.ones(self.height, dtype=bool)
        mask[full_rows] = False
        remaining = self.board[mask]
        empty_rows = np.zeros((len(full_rows), self.width), dtype=np.float32)
        self.board = np.vstack([empty_rows, remaining])
        
        self.lines_cleared += len(full_rows)
        return len(full_rows)
    
    def _compute_heuristics(self) -> Dict[str, float]:
        """Compute Dellacherie board heuristics."""
        # Column heights
        heights = np.zeros(self.width)
        for c in range(self.width):
            for r in range(self.height):
                if self.board[r, c] > 0:
                    heights[c] = self.height - r
                    break
        
        # Aggregate height
        agg_height = np.sum(heights)
        
        # Holes
        holes = 0
        for c in range(self.width):
            found_block = False
            for r in range(self.height):
                if self.board[r, c] > 0:
                    found_block = True
                elif found_block:
                    holes += 1
        
        # Bumpiness
        bumpiness = 0
        for c in range(self.width - 1):
            bumpiness += abs(heights[c] - heights[c + 1])
        
        return {
            'agg_height': agg_height,
            'holes': holes,
            'bumpiness': bumpiness,
            'max_height': np.max(heights),
            'column_heights': heights.tolist(),
        }
    
    def _get_state(self) -> np.ndarray:
        """
        State = [flattened_board, piece_one_hot, normalized_features]
        """
        board_flat = self.board.flatten()
        
        piece_oh = np.zeros(len(self.piece_types), dtype=np.float32)
        if self.current_piece in self.piece_types:
            piece_oh[self.piece_types.index(self.current_piece)] = 1.0
        
        h = self._compute_heuristics()
        max_height = self.height * self.width
        features = np.array([
            h['agg_height'] / max(max_height, 1),
            h['holes'] / max(self.height * self.width, 1),
            h['bumpiness'] / max(self.height * (self.width - 1), 1),
        ], dtype=np.float32)
        
        return np.concatenate([board_flat, piece_oh, features])
    
    def render(self) -> str:
        """Text rendering."""
        lines = [f"Level {self.level}: {self.level_name}"]
        lines.append(f"Lines: {self.lines_cleared} | Pieces: {self.pieces_placed}")
        lines.append("+" + "-" * self.width + "+")
        for row in self.board:
            line = "|"
            for cell in row:
                line += "█" if cell > 0 else " "
            lines.append(line + "|")
        lines.append("+" + "-" * self.width + "+")
        
        h = self._compute_heuristics()
        lines.append(f"Height: {h['agg_height']:.0f} | Holes: {h['holes']} | Bump: {h['bumpiness']:.0f}")
        return "\n".join(lines)
    
    def __repr__(self):
        return (f"TetrisCurriculumEnv(level={self.level}, "
                f"'{self.level_name}', "
                f"board={self.width}×{self.height}, "
                f"state_dim={self.state_dim})")


class TetrisCurriculum:
    """
    Manages progression through Tetris curriculum levels.
    
    As agent improves: reduce heuristic_blend to shift from
    heuristic reward → raw lines cleared reward.
    """
    
    def __init__(self, 
                 advance_threshold: float = 2.0,
                 eval_episodes: int = 20,
                 start_level: int = 1,
                 blend_decay: float = 0.995):
        self.advance_threshold = advance_threshold  # Avg lines to advance
        self.eval_episodes = eval_episodes
        self.current_level = start_level
        self.heuristic_blend = 1.0
        self.blend_decay = blend_decay
        self.level_history: Dict[int, List[float]] = {}
    
    def get_env(self, max_pieces: int = 500) -> TetrisCurriculumEnv:
        return TetrisCurriculumEnv(
            level=self.current_level,
            max_pieces=max_pieces,
            heuristic_blend=self.heuristic_blend,
        )
    
    def record_episode(self, lines: int, reward: float):
        if self.current_level not in self.level_history:
            self.level_history[self.current_level] = []
        self.level_history[self.current_level].append(lines)
        # Decay blend toward raw lines
        self.heuristic_blend = max(0.1, self.heuristic_blend * self.blend_decay)
    
    def should_advance(self) -> bool:
        if self.current_level >= 7:
            return False
        history = self.level_history.get(self.current_level, [])
        if len(history) < self.eval_episodes:
            return False
        recent = history[-self.eval_episodes:]
        return np.mean(recent) >= self.advance_threshold
    
    def advance(self) -> bool:
        if self.should_advance():
            self.current_level += 1
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'current_level': self.current_level,
            'heuristic_blend': self.heuristic_blend,
        }
        for level, history in self.level_history.items():
            config = TetrisCurriculumEnv.LEVELS[level]
            stats[f'level_{level}'] = {
                'name': config['name'],
                'episodes': len(history),
                'mean_lines': np.mean(history) if history else 0.0,
                'best_lines': max(history) if history else 0,
            }
        return stats
