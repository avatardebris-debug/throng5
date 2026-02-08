"""
Test holographic fixes at increasing scales.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline
from throng3.environments import GridWorldAdapter

def test_scale(n_neurons, n_episodes=10):
    """Test Meta^N at specific scale."""
    print(f"\n{'='*60}")
    print(f"Testing {n_neurons} neurons ({n_episodes} episodes)")
    print(f"{'='*60}")
    
    try:
        pipeline = MetaNPipeline.create_minimal(
            n_neurons=n_neurons,
            n_inputs=2,
            n_outputs=4
        )
        
        env = GridWorldAdapter()
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 50:
                result = pipeline.step(input_data=obs, reward=0.0)
                output = result['output']
                
                # Check for NaN
                if np.any(np.isnan(output)):
                    print(f"  ✗ NaN detected at episode {episode+1}, step {steps+1}")
                    return False
                
                action = np.argmax(output)
                obs, reward, done, info = env.step(action)
                steps += 1
        
        print(f"  ✓ {n_episodes} episodes completed without NaN")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


print("="*60)
print("Holographic Scaling Test")
print("="*60)
print("Testing Meta^N at increasing scales to verify NaN fixes")

scales = [512, 1024, 2048, 4096]
results = {}

for scale in scales:
    success = test_scale(scale, n_episodes=10)
    results[scale] = success
    
    if not success:
        print(f"\n⚠ Failed at {scale} neurons, stopping test")
        break

print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")

for scale, success in results.items():
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"  {scale:5d} neurons: {status}")

if all(results.values()):
    print(f"\n✓ All scales passed! Holographic layer stable up to {max(scales)} neurons")
else:
    failed_scale = min(s for s, success in results.items() if not success)
    print(f"\n⚠ Failed at {failed_scale} neurons. May need further tuning.")

print("="*60)
