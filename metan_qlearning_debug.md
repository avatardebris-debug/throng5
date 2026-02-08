# Meta^N Q-Learning Integration Issue

## Problem

Standalone Q-learning with curriculum: **100% success**
Meta^N + Q-learning with curriculum: **0% success**

## Root Causes Identified

### 1. Epsilon Not Decaying
```
Standalone: 0.3 → 0.115 (decayed properly)
Meta^N: 0.3 → 0.3 (NO decay)
```

**Cause:** `qlearner.reset_episode()` not being called at episode end in Meta^N integration.

### 2. Wrong State Representation
```
Standalone: Uses raw observation [x, y]
Meta^N: Uses neuron activations (100-dim random noise)
```

**Cause:** Action selection uses `result['activations']` instead of raw observation.

Q-learner trained on [x,y] coordinates, but being queried with 100-dim activation vectors → complete mismatch!

### 3. No Episode Boundary Signal
Meta^N pipeline doesn't know when episodes end, so:
- Q-learner never resets
- Epsilon never decays
- Previous state/action never cleared

## Solution

Need to pass episode information through pipeline context:
1. Add `done` flag to context
2. Call `qlearner.reset_episode()` when done
3. Use raw observation for Q-learning, not activations

## Quick Fix Options

### Option A: Use Raw Obs for Q-Learning
```python
# In test, track raw obs separately
action = meta1.qlearner.select_action(raw_obs)  # Not activations!
```

### Option B: Train Q-Learner on Activations
```python
# Initialize Q-learner with n_states=100 (activation dim)
# Let it learn from activation space instead of raw obs
```

### Option C: Add Observation to Context
```python
# Pass raw obs through pipeline context
context['raw_observation'] = obs
# Q-learner uses this instead of activations
```

**Recommendation:** Option A (simplest, matches standalone test)
