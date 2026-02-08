"""
Benchmark network sizes and estimate training times.
"""

import numpy as np
import time
from simple_baseline import SimplePolicyNetwork

def benchmark_network_size(n_inputs, n_hidden, n_outputs, n_episodes=10):
    """Benchmark a specific network configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {n_hidden} hidden neurons")
    print(f"{'='*60}")
    
    # Create network
    start_create = time.time()
    network = SimplePolicyNetwork(n_inputs, n_hidden, n_outputs)
    create_time = time.time() - start_create
    
    # Calculate memory
    params = network.W1.size + network.b1.size + network.W2.size + network.b2.size
    memory_mb = (params * 8) / (1024 * 1024)  # 8 bytes per float64
    
    print(f"  Parameters: {params:,}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print(f"  Creation time: {create_time*1000:.2f} ms")
    
    # Benchmark forward pass
    obs = np.random.randn(n_inputs)
    forward_times = []
    for _ in range(100):
        start = time.time()
        _, _ = network.forward(obs, training=False)
        forward_times.append(time.time() - start)
    
    avg_forward = np.mean(forward_times) * 1000  # ms
    print(f"  Forward pass: {avg_forward:.3f} ms")
    
    # Benchmark episode (with gradient computation)
    episode_times = []
    for ep in range(n_episodes):
        start = time.time()
        
        # Simulate episode
        for step in range(50):  # 50 steps per episode
            obs = np.random.randn(n_inputs)
            _, action = network.forward(obs, training=True)
            reward = np.random.randn()
            network.store_reward(reward)
        
        # Update
        network.update()
        
        episode_times.append(time.time() - start)
    
    avg_episode = np.mean(episode_times)
    print(f"  Episode time (50 steps): {avg_episode:.3f} s")
    
    # Estimate training times
    episodes_100 = avg_episode * 100 / 60  # minutes
    episodes_1000 = avg_episode * 1000 / 60  # minutes
    
    print(f"\n  Estimated training time:")
    print(f"    100 episodes: {episodes_100:.1f} min")
    print(f"    1000 episodes: {episodes_1000:.1f} min ({episodes_1000/60:.1f} hours)")
    
    # N=30 experiment estimate
    n30_time = avg_episode * 100 * 30 * 2 / 3600  # hours (fresh + pretrained)
    print(f"    N=30 experiment (100 ep each): {n30_time:.1f} hours")
    
    return {
        'n_hidden': n_hidden,
        'params': params,
        'memory_mb': memory_mb,
        'forward_ms': avg_forward,
        'episode_s': avg_episode,
        'train_100ep_min': episodes_100,
        'train_1000ep_min': episodes_1000,
        'n30_experiment_hours': n30_time
    }


def main():
    print("="*60)
    print("Network Size Benchmark")
    print("="*60)
    print("\nTesting GridWorld configuration (2 inputs, 4 outputs)")
    
    # Test various sizes
    sizes = [32, 128, 512, 1024, 2000, 5000, 10000]
    results = []
    
    for size in sizes:
        try:
            result = benchmark_network_size(
                n_inputs=2,
                n_hidden=size,
                n_outputs=4,
                n_episodes=5  # Fewer episodes for large networks
            )
            results.append(result)
        except MemoryError:
            print(f"\n✗ {size} neurons: OUT OF MEMORY")
            break
        except KeyboardInterrupt:
            print(f"\n✗ Interrupted at {size} neurons")
            break
    
    # Summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Neurons':<10} {'Params':<12} {'Memory':<10} {'100ep':<10} {'1000ep':<12} {'N=30':<10}")
    print(f"{'':10} {'':12} {'(MB)':10} {'(min)':10} {'(min)':12} {'(hours)':10}")
    print("-"*60)
    
    for r in results:
        print(f"{r['n_hidden']:<10,} {r['params']:<12,} {r['memory_mb']:<10.2f} "
              f"{r['train_100ep_min']:<10.1f} {r['train_1000ep_min']:<12.1f} "
              f"{r['n30_experiment_hours']:<10.1f}")
    
    print(f"\n{'='*60}")
    print("Recommendations:")
    print(f"{'='*60}")
    
    # Find feasible sizes
    for r in results:
        if r['n30_experiment_hours'] < 1:
            print(f"✓ {r['n_hidden']:,} neurons: Quick validation ({r['n30_experiment_hours']:.1f}h for N=30)")
        elif r['n30_experiment_hours'] < 4:
            print(f"⚠ {r['n_hidden']:,} neurons: Overnight run ({r['n30_experiment_hours']:.1f}h for N=30)")
        elif r['n30_experiment_hours'] < 24:
            print(f"⚠ {r['n_hidden']:,} neurons: Full day ({r['n30_experiment_hours']:.1f}h for N=30)")
        else:
            print(f"✗ {r['n_hidden']:,} neurons: Too slow ({r['n30_experiment_hours']:.1f}h for N=30)")
    
    print(f"\n{'='*60}")
    print("Memory Limits:")
    print(f"{'='*60}")
    max_tested = results[-1]['n_hidden']
    max_memory = results[-1]['memory_mb']
    print(f"  Tested up to: {max_tested:,} neurons ({max_memory:.1f} MB)")
    print(f"  Estimated 100K neurons: ~{(100000**2 * 8 / 1024**2):.0f} MB")
    print(f"  Estimated 1M neurons: ~{(1000000**2 * 8 / 1024**2 / 1024):.1f} GB")
    print(f"  Estimated 10M neurons: ~{(10000000**2 * 8 / 1024**3):.1f} GB")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
