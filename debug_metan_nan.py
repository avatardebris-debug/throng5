"""
Debug Meta^N NaN issues with minimal test.
"""

import numpy as np
from throng3.pipeline import MetaNPipeline

print("="*60)
print("Meta^N Debug Test")
print("="*60)

# Create minimal pipeline
print("\nCreating minimal Meta^N pipeline (100 neurons)...")
pipeline = MetaNPipeline.create_minimal(
    n_neurons=100,
    n_inputs=2,
    n_outputs=4
)

print(f"✓ Pipeline created with {len(pipeline.stack.layers)} layers")

# Test a few steps
print("\nRunning 10 test steps...")
for step in range(10):
    obs = np.random.randn(2)
    
    result = pipeline.step(
        input_data=obs,
        reward=0.1,
        episode_return=0.5
    )
    
    output = result['output']
    loss = result.get('loss', 0.0)
    
    # Check for NaN
    if np.any(np.isnan(output)):
        print(f"\n✗ NaN detected in output at step {step+1}")
        print(f"  Input: {obs}")
        print(f"  Output: {output}")
        print(f"  Loss: {loss}")
        
        # Check holographic state
        if hasattr(pipeline.stack, 'holographic_state'):
            holo = pipeline.stack.holographic_state
            if holo._combined is not None:
                print(f"  Holographic state has NaN: {np.any(np.isnan(holo._combined))}")
        
        break
    else:
        print(f"  Step {step+1}: output={output[:2]}, loss={loss:.4f}")

print("\n" + "="*60)
print("Debug complete")
print("="*60)
