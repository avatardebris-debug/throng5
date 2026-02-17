"""
Test Tetra dialogue loop on Atari game.

Verifies:
1. Plateau detection triggers Tetra query
2. Tetra suggests strategy based on visual/causal patterns
3. Strategy is applied and tested
4. Results are reported back to Tetra
5. Tetra refines strategy
"""

import sys
sys.path.insert(0, '.')

from throng4.environments.atari_adapter import AtariAdapter
from throng4.meta_policy import MetaPolicyController, ControllerConfig
from throng4.meta_policy.tetra_client import TetraClient

print("=" * 60)
print("TETRA DIALOGUE LOOP TEST")
print("=" * 60)

# Create Tetra client
print("\nConnecting to Tetra...")
tetra = TetraClient(base_url="http://localhost:8000")

# Test connection
test_response = tetra.query("Hello, can you hear me?")
if "Error" in test_response:
    print(f"⚠️  {test_response}")
    print("\nMake sure Tetra is running on http://localhost:8000")
    print("Exiting test...")
    sys.exit(1)

print(f"✅ Tetra connected: {test_response[:100]}...")
tetra.reset_conversation()  # Clear test message

# Create controller with Tetra client
controller = MetaPolicyController(
    config=ControllerConfig(
        fingerprint_episodes=10,
        llm_cooldown=15,  # Allow LLM query after 15 episodes
    ),
    llm_client=tetra,
)

# Test on Space Invaders
print("\n" + "#" * 60)
print("# Testing on Space Invaders (game name hidden from Tetra)")
print("#" * 60)

env = AtariAdapter('SpaceInvaders')
pipeline = controller.on_new_environment(env)

print("\nRunning episodes until plateau...")
hypothesis_tested = False

for ep in range(100):
    state = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0
    
    while not done and steps < 500:
        action = pipeline.select_action(state, explore=True)
        next_state, reward, done, info = env.step(action)
        
        controller.on_step(state, action, reward, next_state)
        pipeline.update(state, action, reward, next_state, done)
        
        episode_reward += reward
        state = next_state
        steps += 1
    
    meta_status = controller.on_episode_complete(episode_reward)
    
    if (ep + 1) % 25 == 0:
        print(f"  Ep {ep+1}: avg={meta_status['avg_reward']:.1f}")
    
    # Check if we should test a hypothesis (after plateau)
    if not hypothesis_tested and ep > 30:
        result = controller.test_hypothesis_with_tetra(pipeline)
        
        if result['status'] == 'hypothesis_applied':
            print(f"\n{'='*60}")
            print(f"HYPOTHESIS TESTING")
            print(f"{'='*60}")
            print(f"Strategy: {result['strategy'].name}")
            print(f"Baseline: {result['baseline_reward']:.1f}")
            print(f"Modifications: {result['modifications']}")
            
            # Run test episodes
            test_rewards = []
            print(f"\nRunning {controller.hypothesis_test_episodes} test episodes...")
            
            for test_ep in range(controller.hypothesis_test_episodes):
                state = env.reset()
                done = False
                test_reward = 0.0
                steps = 0
                
                while not done and steps < 500:
                    action = pipeline.select_action(state, explore=True)
                    next_state, reward, done, info = env.step(action)
                    pipeline.update(state, action, reward, next_state, done)
                    
                    test_reward += reward
                    state = next_state
                    steps += 1
                
                test_rewards.append(test_reward)
                print(f"  Test ep {test_ep+1}: {test_reward:.1f}")
            
            # Report results to Tetra
            refinement = controller.report_hypothesis_results(
                test_rewards,
                result['baseline_reward'],
            )
            
            print(f"\n{'='*60}")
            print(f"TETRA REFINEMENT")
            print(f"{'='*60}")
            print(f"{refinement}")
            
            hypothesis_tested = True
            
            # Continue with refined strategy for a few more episodes
            print(f"\nContinuing with refined strategy...")

env.close()

print("\n" + "=" * 60)
print("DIALOGUE LOOP SUMMARY")
print("=" * 60)
print(f"Total hypotheses tested: {controller.total_hypotheses_tested}")
print(f"Hypothesis history:")
for i, h in enumerate(controller.hypothesis_history):
    print(f"  {i+1}. {h['strategy']}: {h['improvement']:+.1f} → {h['refinement'][:100]}...")

print("\n✅ Tetra dialogue loop test complete!")
