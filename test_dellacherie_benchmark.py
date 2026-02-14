"""
Benchmark: Dellacherie Heuristic on Tetris Curriculum

Tests the hand-crafted Dellacherie heuristic on levels 1-7 to establish
a performance ceiling for comparison with our learned agents.

Dellacherie features:
- Landing height
- Eroded piece cells
- Row transitions
- Column transitions
- Holes
- Well sums
"""

import sys
sys.path.insert(0, 'c:/Users/avata/aicompete/throng3')

import numpy as np
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv


def dellacherie_features(board, piece, position):
    """
    Compute Dellacherie heuristic features for a board state.
    
    Returns weighted score (higher = better move).
    """
    height, width = board.shape
    
    # Feature 1: Landing height (negative = prefer lower)
    landing_height = -position[0]
    
    # Feature 2: Eroded piece cells (positive = prefer clearing lines with piece)
    eroded_cells = 0
    # (Simplified: would need to track which cells belong to current piece)
    
    # Feature 3: Row transitions (negative = prefer fewer transitions)
    row_transitions = 0
    for r in range(height):
        for c in range(width - 1):
            if board[r, c] != board[r, c + 1]:
                row_transitions += 1
    
    # Feature 4: Column transitions (negative = prefer fewer transitions)
    col_transitions = 0
    for c in range(width):
        for r in range(height - 1):
            if board[r, c] != board[r + 1, c]:
                col_transitions += 1
    
    # Feature 5: Holes (negative = avoid creating holes)
    holes = 0
    for c in range(width):
        found_block = False
        for r in range(height):
            if board[r, c] == 1:
                found_block = True
            elif found_block and board[r, c] == 0:
                holes += 1
    
    # Feature 6: Well sums (negative = avoid deep wells)
    well_sum = 0
    for c in range(width):
        well_depth = 0
        for r in range(height):
            if board[r, c] == 0:
                # Check if surrounded by blocks
                left_blocked = (c == 0 or board[r, c - 1] == 1)
                right_blocked = (c == width - 1 or board[r, c + 1] == 1)
                if left_blocked and right_blocked:
                    well_depth += 1
        well_sum += well_depth * (well_depth + 1) // 2  # Triangular number
    
    # Dellacherie weights (tuned for standard Tetris)
    score = (
        -1.0 * landing_height +
        1.0 * eroded_cells +
        -1.0 * row_transitions +
        -1.0 * col_transitions +
        -4.0 * holes +
        -1.0 * well_sum
    )
    
    return score


def select_best_action_heuristic(env):
    """
    Select best action using Dellacherie heuristic.
    Evaluates all valid actions and picks the one with highest score.
    """
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None
    
    best_action = valid_actions[0]
    best_score = float('-inf')
    
    current_board = env.board.copy()
    
    for action in valid_actions:
        # Simulate action
        env_copy = TetrisCurriculumEnv(level=env.level, max_pieces=env.max_pieces)
        env_copy.board = current_board.copy()
        env_copy.current_piece = env.current_piece
        
        # Apply action and get resulting board
        _, _, _, info = env_copy.step(action)
        
        # Score the resulting board
        score = dellacherie_features(env_copy.board, env.current_piece, action)
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action


def test_dellacherie_level(level, n_episodes=50, max_lines=100):
    """
    Test Dellacherie heuristic on a single level.
    Returns mean lines cleared per episode.
    
    Args:
        level: Tetris level (1-7)
        n_episodes: Number of episodes to run
        max_lines: Maximum lines to clear before stopping episode (prevents infinite games)
    """
    env = TetrisCurriculumEnv(level=level, max_pieces=100)
    lines_per_episode = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_lines = 0
        
        for step in range(100):
            action = select_best_action_heuristic(env)
            if action is None:
                break
            
            next_state, reward, done, info = env.step(action)
            total_lines += info.get('lines_cleared', 0)
            
            # Cap at max_lines to prevent infinite games
            if total_lines >= max_lines:
                break
            
            if done:
                break
        
        lines_per_episode.append(total_lines)
        
        if (episode + 1) % 10 == 0:
            mean_10 = np.mean(lines_per_episode[-10:])
            print(f"    Ep {episode + 1}: mean_10={mean_10:.2f} lines")
    
    mean_lines = np.mean(lines_per_episode)
    return mean_lines, lines_per_episode


if __name__ == '__main__':
    print("="*70)
    print("DELLACHERIE HEURISTIC BENCHMARK")
    print("="*70)
    print("\nTesting hand-crafted expert on Tetris curriculum (L1-L7)")
    print("Baseline for comparison with learned agents\n")
    
    results = {}
    
    for level in range(1, 8):
        print(f"\n{'='*70}")
        print(f"Level {level}")
        print(f"{'='*70}")
        
        mean_lines, history = test_dellacherie_level(level, n_episodes=50)
        results[level] = {
            'mean_lines': mean_lines,
            'history': history
        }
        
        print(f"\n  Final: {mean_lines:.2f} lines/episode (50 episodes)")
    
    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY: Dellacherie Heuristic Performance")
    print("="*70)
    
    print(f"\n{'Level':<8} {'Mean Lines':<15} {'vs Target (1.5)':<20}")
    print("-" * 50)
    
    for level in range(1, 8):
        mean = results[level]['mean_lines']
        vs_target = "PASS" if mean >= 1.5 else "FAIL"
        print(f"  {level:<6} {mean:<15.2f} {vs_target:<20}")
    
    print("\n" + "="*70)
    print("COMPARISON: Dellacherie vs Our Best Agent (128 units, progressive)")
    print("="*70)
    
    # Our agent's results from Phase F6
    our_results = {
        1: 1.50,
        2: 1.50,
        3: 1.50,
        4: 1.50,
        5: 1.50,
        6: 1.50,
        7: 0.00,
    }
    
    print(f"\n{'Level':<8} {'Dellacherie':<15} {'Our Agent':<15} {'Gap':<15}")
    print("-" * 60)
    
    for level in range(1, 8):
        dell = results[level]['mean_lines']
        ours = our_results[level]
        gap = dell - ours
        print(f"  {level:<6} {dell:<15.2f} {ours:<15.2f} {gap:+<15.2f}")
    
    print("\nNote: Positive gap = Dellacherie better, Negative gap = Our agent better")
