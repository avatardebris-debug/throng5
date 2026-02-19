import sys, random
sys.path.insert(0, '.')
from throng4.environments.tetris_adapter import TetrisAdapter

print("=== Verifying game-over fix ===")
print("Running 200 random L7 episodes, checking for 'cleared line then died'...")

impossible_deaths = 0
total_games = 200
piece_counts = []

for trial in range(total_games):
    adapter = TetrisAdapter(level=7, max_pieces=500)
    adapter.reset()
    done = False
    last_lines = 0
    while not done:
        valid = adapter.get_valid_actions()
        if not valid:
            break
        _, _, done, info = adapter.step(random.choice(valid))
        new_lines = info['lines_cleared'] - last_lines
        last_lines = info['lines_cleared']
        if done and new_lines > 0:
            impossible_deaths += 1
            print(
                f"  Trial {trial}: DIED after clearing {new_lines} lines! "
                f"pieces={info['pieces_placed']} total_lines={info['lines_cleared']}"
            )

    piece_counts.append(adapter.env.pieces_placed)

import statistics
print(f"\nResults ({total_games} episodes):")
print(f"  Impossible deaths (cleared line then died): {impossible_deaths}")
print(f"  Pieces placed: min={min(piece_counts)} mean={statistics.mean(piece_counts):.1f} max={max(piece_counts)}")
print(f"  Verdict: {'FIXED' if impossible_deaths == 0 else 'STILL BROKEN'}")
