"""
replay_short_episode.py

Runs L7 episodes with full board printing when pieces_placed <= 2 at death.
If it can't reproduce the bug, prints stats and exits.

Run: python replay_short_episode.py
"""
import sys; sys.path.insert(0, '.')
from throng4.environments.tetris_curriculum import TetrisCurriculumEnv, TETROMINOES
from throng4.environments.tetris_adapter import TetrisAdapter
from throng4.cognition.enaction_engine import EnactionEngine
from throng4.environments.tetris_curriculum import DellacherieWeights
import numpy as np

def print_board(board, label=''):
    h, w = board.shape
    print(f'  {label}  ({h}x{w})')
    print('  +' + '-'*w + '+')
    for r, row in enumerate(board):
        print(f'  |{"".join("#" if c else "." for c in row)}|  row {r}')
    print('  +' + '-'*w + '+')

caught = 0
total = 0

for trial in range(50000):
    # Mirror what FastLoop._run_episode does exactly
    weights = DellacherieWeights()
    try:
        eng = EnactionEngine()
        cfg = eng.load()
        cfg.apply_to_weights(weights)
    except Exception:
        cfg = None

    adapter = TetrisAdapter(level=7, max_pieces=500, weights=weights)
    state = adapter.reset()
    done = False
    steps = 0

    while not done:
        valid_actions = adapter.get_valid_actions()
        if not valid_actions:
            print(f"  [trial {trial}] BREAK: no valid actions at step {steps}, "
                  f"pieces={adapter.env.pieces_placed}")
            break

        # Epsilon-greedy like FastLoop (15% random)
        if np.random.rand() < 0.15:
            action = valid_actions[np.random.randint(len(valid_actions))]
        else:
            action = valid_actions[0]  # greedy = first

        next_state, reward, done, info = adapter.step(action)

        if cfg is not None and cfg.piece_phases:
            holes_mult = cfg.get_phase_multiplier('holes', adapter.env.pieces_placed)
            if holes_mult != 1.0 and reward < 0:
                reward = reward * holes_mult

        steps += 1
        state = next_state

    total += 1
    ep_info = adapter.get_info()
    pieces = ep_info['pieces_placed']
    lines  = ep_info['lines_cleared']

    if pieces <= 2:
        caught += 1
        print(f'\n=== CAUGHT: pieces_placed={pieces} lines={lines} trial={trial} ===')
        print(f'  steps in loop: {steps}   done={done}')
        print(f'  current_piece: {adapter.env.current_piece}')
        print_board(adapter.env.board, label='Board at death:')
        print(f'  env.pieces_placed: {adapter.env.pieces_placed}')
        print(f'  valid_actions for next piece: {len(adapter.env.get_valid_actions())}')
        if caught >= 3:
            break

print(f'\nTotal: {caught}/{total} short episodes in {total} trials')
