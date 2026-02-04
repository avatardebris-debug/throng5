"""
Basic Pipeline Example — Meta^N in Action

Demonstrates the full Meta^N pipeline:
1. Creates a default 6-layer stack
2. Runs optimization on a simple regression task
3. Shows cross-scale communication and holographic state
4. Prints a system report

Usage:
    python examples/basic_pipeline.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from throng3.pipeline import MetaNPipeline


def main():
    print("=" * 60)
    print("  throng3 — Meta^N Recursive Self-Optimization")
    print("  Basic Pipeline Example")
    print("=" * 60)
    
    # Create pipeline with all meta-layers
    print("\n🔧 Creating Meta^N pipeline...")
    pipeline = MetaNPipeline.create_default(
        n_neurons=500,
        n_inputs=32,
        n_outputs=16,
        include_llm=False,  # No LLM for basic demo
    )
    print(f"   {pipeline}")
    
    # Generate a simple regression task
    print("\n📊 Generating regression task...")
    np.random.seed(42)
    W_true = np.random.randn(16, 32) * 0.3
    noise_level = 0.05
    
    # Training loop
    print("\n🚀 Running optimization...")
    n_steps = 200
    losses = []
    
    for step in range(n_steps):
        # Generate sample
        x = np.random.randn(32)
        y = W_true @ x + np.random.randn(16) * noise_level
        
        # Compute reward based on improvement
        reward = 0.0
        if losses:
            reward = max(0, losses[-1] - 0.01)  # Reward for improving
        
        # Step the pipeline
        result = pipeline.step(x, target=y, reward=reward)
        losses.append(result['loss'])
        
        # Print progress
        if (step + 1) % 50 == 0:
            recent_loss = np.mean(losses[-50:])
            holo = result.get('holographic', {})
            coherence = holo.get('coherence', 0)
            signals = result.get('signals_routed', 0)
            
            print(f"   Step {step+1:4d}: loss={recent_loss:.4f}  "
                  f"coherence={coherence:.3f}  signals={signals}")
    
    # Final report
    print("\n" + pipeline.get_report())
    
    # Show layer-by-layer status
    print("\n📈 Layer Details:")
    state = pipeline.stack.get_system_state()
    for level, layer_state in sorted(state['layers'].items()):
        print(f"   Meta^{level} ({layer_state['name']}):")
        print(f"      Step: {layer_state['optimization_step']}")
        metrics = layer_state['metrics']
        print(f"      Loss: {metrics.get('loss', '?'):.4f}")
        print(f"      Accuracy: {metrics.get('accuracy', '?'):.4f}")
        print(f"      Stability: {metrics.get('stability', '?'):.3f}")
    
    # Show learning curve
    print("\n📉 Learning Curve:")
    window = 20
    for i in range(0, len(losses), window):
        chunk = losses[i:i+window]
        avg = np.mean(chunk)
        bar = "█" * int(avg * 50) + "░" * (50 - int(avg * 50))
        print(f"   Steps {i:3d}-{i+window:3d}: {avg:.4f} |{bar}|")
    
    print("\n✅ Pipeline example complete!")
    return pipeline


if __name__ == '__main__':
    main()
