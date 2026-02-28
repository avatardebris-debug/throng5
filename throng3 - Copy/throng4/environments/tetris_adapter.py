"""
Tetris Adapter — Bridges TetrisCurriculumEnv to PortableNNAgent.

Provides:
- Feature extraction matching the HTML version (boardW + 12 features)
- Action enumeration (valid placements)
- Lookahead support for multi-move planning
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv, TETROMINOES
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
    
    def __init__(self, level: int = 1, max_pieces: int = 500,
                 weights=None, seed: Optional[int] = None):
        """
        Initialize Tetris adapter.

        Args:
            level:      Curriculum level (1-7)
            max_pieces: Max pieces per episode
            weights:    Optional DellacherieWeights — pass enacted weights here
            seed:       Random seed for the environment
        """
        super().__init__()
        self.env = TetrisCurriculumEnv(level=level, max_pieces=max_pieces,
                                       weights=weights, seed=seed)
        self.level = level
        self.max_pieces = max_pieces

        
        # Feature dimensions
        self.board_width = self.env.width
        self.n_features = self.board_width + 12
        
        # Current state
        self.current_piece = None
        self.valid_actions = []
        self.done = False
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset environment.
        
        Returns:
            Initial state (not used by PortableNNAgent, which uses features)
        """
        state = self.env.reset(seed=seed)
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
        # IMPORTANT: use .copy() not [:] — numpy row slices are views, not copies.
        # Using [:] would mutate self.env.board through the view.
        board_copy = [row.copy() for row in self.env.board]
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
        
        # Count cleared lines (a row is full if every cell is truthy / filled)
        cleared = sum(1 for r in board_copy if all(cell for cell in r))

        
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
            board: Board state (height × width). Cells are truthy (filled)
                   or falsy (empty) — numpy.float32 1.0/0.0, or None/value.

        Returns:
            Dict with agg_height, holes, bumpiness, max_height, heights,
            avg_completeness, and column_heights.
        """
        board_h = len(board)
        board_w = len(board[0]) if board_h > 0 else self.board_width

        # Column heights: scan top-to-bottom, first filled cell defines height
        heights = []
        for c in range(board_w):
            h = 0
            for r in range(board_h):
                if board[r][c]:          # truthy = filled
                    h = board_h - r
                    break
            heights.append(h)

        agg_height = sum(heights)
        max_height = max(heights) if heights else 0

        # Holes: empty cell beneath a filled cell in the same column
        holes = 0
        for c in range(board_w):
            found_block = False
            for r in range(board_h):
                if board[r][c]:          # filled cell
                    found_block = True
                elif found_block:        # empty cell below a filled one
                    holes += 1

        # Bumpiness: sum of absolute height differences between adjacent cols
        bumpiness = sum(abs(heights[i] - heights[i + 1])
                        for i in range(len(heights) - 1))

        # Row completeness
        occupied_rows = 0
        total_fill    = 0.0
        for r in range(board_h):
            filled = sum(1 for c in range(board_w) if board[r][c])
            if filled > 0:
                occupied_rows += 1
                total_fill    += filled / board_w

        avg_completeness = total_fill / occupied_rows if occupied_rows > 0 else 0.0

        return {
            'agg_height':       agg_height,
            'holes':            holes,
            'bumpiness':        bumpiness,
            'max_height':       max_height,
            'heights':          heights,
            'avg_completeness': avg_completeness,
        }

    
    def get_lookahead_actions(self, action: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Enumerate valid placements for a sampled next piece, after placing `action`.

        Simulates the board state that results from placing the current piece,
        clears any completed lines on the copy, then samples one random next piece
        from the level's piece set and returns every valid (rotation, col) for it.

        Args:
            action: (rotation, column) of the current piece placement

        Returns:
            List of (rotation, col) tuples valid for the simulated next state.
        """
        rotation, column = action

        # — Simulate current placement on a board copy (no mutation) —
        board_copy = self.env.board.copy()
        piece_cells = self.env._get_piece_cells(rotation)

        row = 0
        while self.env._can_place_at(row + 1, column, rotation):
            row += 1
        for dr, dc in piece_cells:
            r, c = row + dr, column + dc
            if 0 <= r < self.env.height and 0 <= c < self.env.width:
                board_copy[r, c] = 1.0

        # — Clear completed lines on copy —
        full_rows = np.where(np.all(board_copy > 0, axis=1))[0]
        if len(full_rows):
            mask = np.ones(self.env.height, dtype=bool)
            mask[full_rows] = False
            remaining = board_copy[mask]
            empty = np.zeros((len(full_rows), self.env.width), dtype=np.float32)
            board_copy = np.vstack([empty, remaining])

        # — Enumerate valid placements for ALL level pieces (no rng sampling) —
        # We avoid calling self.env.rng here so that audit replays see the same
        # piece sequence as the original episode.
        lookahead_actions = []
        for piece in self.env.piece_types:
            lookahead_actions.extend(self._enumerate_actions_on_board(board_copy, piece))
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for a in lookahead_actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique

    def _enumerate_actions_on_board(
        self, board: np.ndarray, piece: str
    ) -> List[Tuple[int, int]]:
        """
        Return valid (rotation, col) placements for `piece` on `board`.

        Used by get_lookahead_actions to enumerate future valid actions without
        advancing the live environment state.
        """
        rotations_def = TETROMINOES[piece]
        actions = []
        for rot_idx, cells in enumerate(rotations_def):
            min_c = min(dc for _, dc in cells)
            max_c = max(dc for _, dc in cells)
            for col in range(-min_c, self.env.width - max_c):
                row = 0
                while self._can_place_on(board, row + 1, col, cells):
                    row += 1
                if self._can_place_on(board, row, col, cells):
                    actions.append((rot_idx, col))
        return actions

    def _can_place_on(
        self, board: np.ndarray, row: int, col: int,
        cells: List[Tuple[int, int]]
    ) -> bool:
        """
        Check whether `cells` (relative offsets) fit at (row, col) on `board`.

        Equivalent to TetrisCurriculumEnv._can_place_at but works on an arbitrary
        board array rather than self.env.board.
        """
        for dr, dc in cells:
            r, c = row + dr, col + dc
            if r < 0 or r >= self.env.height or c < 0 or c >= self.env.width:
                return False
            if board[r, c] > 0:
                return False
        return True

    def make_features_for(
        self, board: np.ndarray, piece: str, action: Tuple[int, int]
    ) -> np.ndarray:
        """
        Build a feature vector for (board, piece, action) without touching live state.

        Used during lookahead scoring so the agent can evaluate future placements
        with the correct piece type encoded, rather than the current live piece.

        Args:
            board:  Board state (height × width) to evaluate against.
            piece:  Piece type string, e.g. 'I', 'O', 'T'.
            action: (rotation, column) placement to evaluate.

        Returns:
            Feature vector of size (boardW + 12,)
        """
        rotation, column = action
        cells = TETROMINOES[piece][rotation % len(TETROMINOES[piece])]

        # Find drop row on this board
        row = 0
        while self._can_place_on(board, row + 1, column, cells):
            row += 1

        # Simulate placement on a copy
        board_copy = [r_.copy() for r_ in board]
        for dr, dc in cells:
            r_, c_ = row + dr, column + dc
            if 0 <= r_ < self.env.height and 0 <= c_ < self.env.width:
                board_copy[r_][c_] = 1.0

        cleared = sum(1 for r_ in board_copy if all(cell for cell in r_))
        features_dict = self._compute_board_features(board_copy)

        features = []
        features.append(features_dict['agg_height'] / (self.board_width * self.env.height))
        features.append(features_dict['holes'] / (self.board_width * self.env.height * 0.5))
        features.append(features_dict['bumpiness'] / (self.env.height * (self.board_width - 1)))
        features.append(features_dict['max_height'] / self.env.height)
        for h in features_dict['heights']:
            features.append(h / self.env.height)
        features.append(cleared / 4.0)
        features.append(row / self.env.height)
        features.append(column / self.board_width)
        features.append(rotation / 4.0)

        all_pieces = ['I', 'O', 'T', 'S', 'Z', 'L', 'J']
        piece_idx = all_pieces.index(piece)
        features.append(1.0 if piece_idx == 0 else 0.0)       # I
        features.append(1.0 if piece_idx == 1 else 0.0)       # O
        features.append(1.0 if 2 <= piece_idx <= 4 else 0.0)  # T/S/Z
        features.append(features_dict['avg_completeness'])

        return np.array(features, dtype=np.float32)
    
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

    # ------------------------------------------------------------------
    # Abstract Feature Protocol
    # ------------------------------------------------------------------

    def get_core_features(self) -> np.ndarray:
        """
        Map current Tetris board state to the universal 20-dim core vector.

        agent   = current piece position (col normalized, row from drop sim)
        target  = best column heuristically (centre of least-hole region)
        threat  = board saturation (max_height / board_height)
        density = hole density
        resource = remaining capacity (1 - saturation)
        """
        from throng4.learning.abstract_features import (
            empty_core, IDX_AGENT_X, IDX_AGENT_Y, IDX_TARGET_X, IDX_TARGET_Y,
            IDX_THREAT_PROX, IDX_REWARD_PROX, IDX_RESOURCE,
            IDX_DENSITY, IDX_EPISODE_PROG, IDX_CONTEXT_0, IDX_CONTEXT_1, IDX_CONTEXT_2
        )
        core = empty_core()

        board = self.env.board
        H = float(self.env.height)
        W = float(self.board_width)

        feats = self._compute_board_features(board)
        max_h    = feats['max_height'] / H
        holes    = feats['holes'] / max(W * H * 0.5, 1.0)
        bump     = feats['bumpiness'] / max(H * (W - 1), 1.0)
        heights  = feats['heights']

        # agent_x = current piece column (midpoint of valid actions if available)
        if self.valid_actions:
            cols = [a[1] for a in self.valid_actions]
            agent_x = (min(cols) + max(cols)) / 2.0 / W
        else:
            agent_x = 0.5

        # agent_y = normalised drop row (approx from max_height)
        agent_y  = max_h

        # target_x = column with lowest height (where pieces should go)
        if heights:
            best_col = int(np.argmin(heights))
            target_x = best_col / W
            target_y = heights[best_col] / H
        else:
            target_x, target_y = 0.5, 0.0

        # threat_prox: how close we are to the danger ceiling (0=safe, 1=full)
        threat_prox = max_h

        # reward_prox: proximity to clearing a line (avg row completeness)
        reward_prox = feats['avg_completeness']

        # resource: remaining column capacity (inverse of average fill)
        avg_fill    = feats['agg_height'] / max(H * W, 1.0)
        resource    = 1.0 - avg_fill

        core[IDX_AGENT_X]      = agent_x
        core[IDX_AGENT_Y]      = agent_y
        core[IDX_TARGET_X]     = target_x
        core[IDX_TARGET_Y]     = target_y
        core[IDX_THREAT_PROX]  = np.clip(threat_prox, 0, 1)
        core[IDX_REWARD_PROX]  = np.clip(reward_prox, 0, 1)
        core[IDX_RESOURCE]     = np.clip(resource, 0, 1)
        core[IDX_DENSITY]      = np.clip(holes, 0, 1)
        core[IDX_EPISODE_PROG] = min(getattr(self.env, 'pieces_placed', 0) / self.max_pieces, 1.0)
        core[IDX_CONTEXT_0]    = np.clip(bump, 0, 1)      # bumpiness
        core[IDX_CONTEXT_1]    = feats['avg_completeness'] # row fill fraction
        core[IDX_CONTEXT_2]    = min(getattr(self.env, 'lines_cleared', 0) / 10.0, 1.0)

        return core

    def get_ext_features(self):
        """
        Tetris extension block: per-column heights (normalized).

        For level 2 (4-wide) to level 7 (8-wide) boards this gives 4–8 slots
        of detailed column structure — the richest game-specific signal.
        """
        from throng4.learning.abstract_features import make_ext
        board  = self.env.board
        H      = float(self.env.height)
        feats  = self._compute_board_features(board)
        heights_norm = [h / H for h in feats['heights']]
        # Also append agg_height and bumpiness as the last two ext slots
        agg_norm  = feats['agg_height'] / max(H * self.board_width, 1.0)
        bump_norm = feats['bumpiness'] / max(H * (self.board_width - 1), 1.0)
        return make_ext(heights_norm + [agg_norm, bump_norm])

