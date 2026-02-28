# Throng3.5: Regional Brain Architecture

## Overview

Throng3.5 implements a brain-inspired regional architecture where different learning systems operate independently with appropriate state representations and timing.

## Architecture

```
throng3.5/
├── core/           # Preserved from throng3
├── learning/       # Preserved from throng3
├── regions/        # NEW: Brain regions
├── coordination/   # NEW: Executive controller
├── environments/   # Preserved from throng3
└── benchmarks/     # Preserved from throng3
```

## Key Differences from Throng3

| Aspect | Throng3 | Throng3.5 |
|--------|---------|-----------|
| Architecture | Single pipeline | Regional brain |
| Learning | Mixed in one layer | Separated by region |
| Timing | Single step() call | Independent per region |
| State | Shared activations | Region-specific |
| Q-learning | 25% max (timing issue) | Target: 100% |

## Regions

### Striatum (Q-Learning)
- **Purpose:** Goal-directed action selection
- **Input:** Raw observations
- **Timing:** Action → Reward → Learn
- **Output:** Actions, Q-values, TD-error

### Cortex (Hebbian)
- **Purpose:** Pattern recognition, feature learning
- **Input:** Neuron activations
- **Timing:** Continuous pattern learning
- **Output:** Features, learned patterns

### Hippocampus (STDP)
- **Purpose:** Sequence learning, episodic memory
- **Input:** Spike sequences
- **Timing:** Temporal correlation learning
- **Output:** Sequence predictions

### Executive Controller (Meta^3)
- **Purpose:** Coordinate regions, route information
- **Mechanism:** Adaptive routing based on task complexity

## Development Status

- [ ] Core infrastructure copied
- [ ] Striatum region implemented
- [ ] Cortex region implemented
- [ ] Executive controller implemented
- [ ] GridWorld benchmark (target: 100%)
- [ ] Transfer learning validation

## Next Steps

1. Copy core infrastructure from throng3
2. Implement Striatum region
3. Test standalone (expect 100% on GridWorld)
4. Add Cortex region
5. Implement Executive controller
6. Full integration testing
