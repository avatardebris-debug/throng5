"""Debug hole detection: inspect the actual board cell values after an episode."""
import sys
sys.path.insert(0, '.')
from throng4.environments.tetris_adapter import TetrisAdapter

adapter = TetrisAdapter(level=2, max_pieces=50)
adapter.reset()

# Play through one episode quickly
done = False
steps = 0
while not done and steps < 60:
    valid = adapter.get_valid_actions()
    if not valid:
        break
    # just pick first action always
    _, _, done, _ = adapter.step(valid[0])
    steps += 1

board = adapter.env.board
print(f"Board dimensions: {len(board)} rows x {len(board[0])} cols")
print(f"Board type of cell [0][0]: {type(board[0][0])!r}")
print(f"Sample cell values (bottom 3 rows):")
for r in range(max(0, len(board)-3), len(board)):
    row_vals = [repr(board[r][c]) for c in range(len(board[r]))]
    print(f"  row {r}: {row_vals}")

print()
# Show which cells the height calc considers "occupied"
print("Height detection (board[r][c] is not None):")
for c in range(len(board[0])):
    for r in range(len(board)):
        cell = board[r][c]
        if cell is not None:
            print(f"  col {c}: first block at row {r}, cell={repr(cell)}")
            break

print()
# Raw hole scan
holes = 0
for c in range(len(board[0])):
    found_block = False
    for r in range(len(board)):
        cell = board[r][c]
        if cell is not None:
            found_block = True
        elif found_block:
            holes += 1
print(f"Holes (None check): {holes}")

# Also try 0/falsy check
holes2 = 0
for c in range(len(board[0])):
    found_block = False
    for r in range(len(board)):
        cell = board[r][c]
        if cell:  # truthy
            found_block = True
        elif found_block:
            holes2 += 1
print(f"Holes (truthy check): {holes2}")

# Show features from adapter
bf = adapter._compute_board_features(board)
print(f"\n_compute_board_features result: {bf}")

# Also get info
print(f"get_info: {adapter.get_info()}")
