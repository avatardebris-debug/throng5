"""Run full curriculum L2-L7, 1500 episodes per level."""
import sys, subprocess
sys.path.insert(0, '.')

levels = [2, 3, 4, 5, 6, 7]
for lvl in levels:
    print(f"\n=== L{lvl} ===", flush=True)
    subprocess.run(
        ['python', '-X', 'utf8', '-m', 'throng4.runners.fast_loop',
         '--level', str(lvl), '--episodes', '1500'],
        cwd=r'c:\Users\avata\aicompete\throng3'
    )

print("\n=== All levels done. Running SlowLoop nightly. ===")
subprocess.run(
    ['python', '-X', 'utf8', '-m', 'throng4.runners.slow_loop', '--mode', 'nightly'],
    cwd=r'c:\Users\avata\aicompete\throng3'
)
print("\nCurriculum complete.")
