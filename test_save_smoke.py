"""Quick integration test for SaveStateManager in MetaPolicyController."""
import sys
sys.path.insert(0, '.')

from throng4.meta_policy import MetaPolicyController, SaveStateManager

c = MetaPolicyController()
print(f"save_state_manager: {hasattr(c, 'save_state_manager')}")
assert hasattr(c, 'save_state_manager')

# Run a few episodes
for i in range(5):
    c.on_episode_complete(1.0)
print(f"Episodes: {c.episode_count}")
print(f"Save states: {len(c.save_state_manager.save_states)}")
print("Integration OK")
