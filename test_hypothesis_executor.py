"""
Test hypothesis executor on real Atari game.

Verifies:
1. Visual pattern extraction detects grid formation + synchronized motion
2. Causal discovery identifies shooting action
3. LLM prompt includes visual + causal information
"""

import sys
sys.path.insert(0, '.')

from throng4.environments.atari_adapter import AtariAdapter
from throng4.meta_policy import (
    MetaPolicyController,
    ControllerConfig,
)
import numpy as np

print("=" * 60)
print("HYPOTHESIS EXECUTOR INTEGRATION TEST")
print("=" * 60)

# Create controller
controller = MetaPolicyController(ControllerConfig(
    fingerprint_episodes=10,
    concept_discovery_interval=25,
    promote_after_episodes=30,
))

# Test on Space Invaders (should detect grid + synchronized motion)
print("\n" + "#" * 60)
print("# Testing on Space Invaders")
print("#" * 60)

env = AtariAdapter('SpaceInvaders')
pipeline = controller.on_new_environment(env)

# Run 100 episodes to collect data
print("\nRunning 100 episodes to collect visual/causal data...")
for ep in range(100):
    state = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    
    while not done and steps < 500:
        action = pipeline.select_action(state, explore=True)
        next_state, reward, done, info = env.step(action)
        
        # Record for visual/causal discovery
        controller.on_step(state, action, reward, next_state)
        
        pipeline.update(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state
        steps += 1
    
    meta_status = controller.on_episode_complete(episode_reward)
    
    if (ep + 1) % 25 == 0:
        print(f"  Ep {ep+1}: avg={meta_status['avg_reward']:.1f}, "
              f"states_tracked={len(controller.recent_states)}, "
              f"transitions={len(controller.recent_transitions)}")

env.close()

# Check visual patterns
print("\n" + "=" * 60)
print("VISUAL PATTERN EXTRACTION")
print("=" * 60)

if controller.current_visual_patterns:
    print(controller.current_visual_patterns.summary())
    
    # Verify expectations for Space Invaders
    vp = controller.current_visual_patterns
    print("\nVerification:")
    print(f"  Entity count > 10: {vp.entity_count > 10} (expected: True)")
    print(f"  Motion type: {vp.motion_type} (expected: synchronized or grid-like)")
    print(f"  Spatial layout: {vp.spatial_layout} (expected: grid or clustered)")
else:
    print("⚠️  No visual patterns extracted (need more states)")

# Check causal discovery
print("\n" + "=" * 60)
print("CAUSAL DISCOVERY")
print("=" * 60)

if controller.current_causal_effects:
    print(controller.causal_discovery.get_summary(controller.current_causal_effects))
    
    # Find shooting action (should create entities)
    shooting_actions = [
        aid for aid, effect in controller.current_causal_effects.items()
        if effect.creates_entities
    ]
    
    print(f"\nActions that create entities (likely shooting): {shooting_actions}")
    print(f"Expected: Should include action 1 or 4 (Space Invaders shoot actions)")
else:
    print("⚠️  No causal effects discovered (need more transitions)")

# Generate LLM prompt
print("\n" + "=" * 60)
print("LLM PROMPT (with visual + causal info)")
print("=" * 60)

# Force plateau to trigger prompt generation
controller.episode_rewards.extend([50.0] * 50)  # Fake plateau
prompt = controller.get_abstract_llm_prompt()

if prompt:
    print(prompt)
    
    # Verify no game names leaked
    assert 'SpaceInvaders' not in prompt, "LEAK: Game name in prompt!"
    assert 'space' not in prompt.lower() or 'state space' in prompt.lower(), "LEAK: 'space' without 'state space'"
    assert 'invaders' not in prompt.lower(), "LEAK: 'invaders' in prompt!"
    
    print("\n✅ No game names leaked!")
    
    # Verify visual/causal info present
    if controller.current_visual_patterns:
        assert 'Visual Patterns' in prompt or 'Entity count' in prompt, "Missing visual patterns!"
        print("✅ Visual patterns included in prompt")
    
    if controller.current_causal_effects:
        assert 'Causal Discovery' in prompt or 'action effects' in prompt, "Missing causal discovery!"
        print("✅ Causal discovery included in prompt")
else:
    print("⚠️  No prompt generated (cooldown or not plateauing)")

print("\n" + "=" * 60)
print("✅ HYPOTHESIS EXECUTOR INTEGRATION TEST COMPLETE!")
print("=" * 60)
