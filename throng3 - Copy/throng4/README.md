# Throng4 — Unified Compute Graph Architecture

**Status**: Alpha — Scaffolding complete, ready for testing

## Overview

Throng4 replaces Throng3's SNN substrate with a **dual-head ANN** (unified compute graph) while preserving the validated Meta^1-3 hierarchy. This enables real function approximation for complex tasks like Tetris and multi-game generalization.

## Key Innovation: Dual-Head Architecture

```
Input (state)
    ↓
Shared Backbone (128 hidden units, ReLU)
    ├→ Q-value Head (n_actions outputs) ← Primary task
    └→ Reward Prediction Head (1 output)  ← Auxiliary task
```

**Benefits:**
- ✅ Single forward pass per step (solves "double call" problem)
- ✅ Auxiliary learning signal improves feature quality
- ✅ Better transfer learning (reward features generalize)
- ✅ MAML can optimize strategies for both heads

## Architecture Comparison

| Component | Throng3 | Throng3.5 | Throng4 |
|-----------|---------|-----------|---------|
| Meta^0 | SNN (100 LIF neurons) | Tabular Q | **Dual-head ANN** |
| Q-learning | Tabular (bolted on) | Tabular | **DQN through network** |
| Reward prediction | ❌ None | ❌ None | ✅ **Auxiliary head** |
| Function approx | ❌ Linear only | ❌ Linear only | ✅ **Non-linear (ReLU)** |
| Calls per step | 2 (timing issue) | 1 (fixed) | 1 (unified graph) |

## Quick Start

```python
from throng4 import ANNLayer, DQNLearner, DQNConfig

# Create dual-head ANN
ann = ANNLayer(
    n_inputs=25,      # GridWorld: 5×5 grid
    n_hidden=128,     # Shared backbone size
    n_outputs=4       # 4 actions (up/down/left/right)
)

# Create DQN learner
config = DQNConfig(
    learning_rate=0.001,
    epsilon=0.1,
    aux_loss_weight=0.1  # Weight for reward prediction
)
learner = DQNLearner(ann, config)

# Training loop
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        # Select action
        action = learner.select_action(state)
        
        # Environment step
        next_state, reward, done = env.step(action)
        
        # Update (both Q-learning + reward prediction)
        errors = learner.update(state, action, reward, next_state, done)
        
        state = next_state
    
    print(f"Episode {episode}: ε={learner.epsilon:.3f}, "
          f"TD={errors['td_error']:.3f}, "
          f"Reward_err={errors['reward_error']:.3f}")
```

## Project Structure

```
throng4/
├── __init__.py
├── layers/
│   ├── __init__.py
│   └── meta0_ann.py          # Dual-head ANN layer
├── learning/
│   ├── __init__.py
│   └── dqn.py                # DQN with reward prediction
├── core/                     # (TODO: copy from throng3)
└── README.md                 # This file
```

## What's Implemented

- ✅ **ANNLayer**: Dual-head architecture with shared backbone
- ✅ **DQNLearner**: Q-learning + reward prediction with experience replay
- ✅ **Backward passes**: Separate gradients for Q-head and reward-head
- ✅ **Package structure**: Clean imports and initialization

## What's Next (TODO)

1. **Copy core utilities** from throng3 (signal protocol, holographic state, etc.)
2. **Create simple test** to verify dual-head forward/backward pass
3. **Test on GridWorld** to validate learning
4. **Compare to Throng3/3.5** baselines
5. **Implement Meta^1-3** adaptations for dual-head weights

## Differences from Throng3

### Meta^0 (Substrate)
- **Throng3**: SNN with LIF neurons, spike-based
- **Throng4**: Dual-head ANN with ReLU, gradient-based

### Meta^1 (Synapse Optimizer)
- **Throng3**: STDP, Hebbian, dopamine modulation
- **Throng4**: Gradient descent with dual loss (Q + reward prediction)

### Meta^2-3 (Higher Layers)
- **Same**: UCB rule selector, EWC, MAML
- **Change**: Operate on ANN weights instead of SNN weights

## Design Principles

1. **Unified compute graph**: Single forward pass, dual outputs
2. **Auxiliary learning**: Reward prediction improves features
3. **Drop-in compatibility**: Same MetaLayer interface as Throng3
4. **MAML-ready**: Dual heads enable richer meta-learning

## References

- **DoorDash approach** (Stanley Tang): Unified compute graph with shared backbone
- **DQN** (Mnih et al.): Q-learning through neural networks
- **Auxiliary tasks** (Jaderberg et al.): Improve representation learning

---

**Version**: 4.0.0-alpha  
**License**: Research code — Edgar Lab
