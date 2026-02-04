# throng3 — Meta^N Recursive Self-Optimization Architecture

**Recursive self-optimization with holographic redundancy across fractal layers.**

## 🧠 What is Meta^N?

Every learning system has meta-levels. Traditional ML has one: gradient descent optimizes weights. But what if every level of the system could optimize itself, AND optimize the levels above and below it?

Meta^N is a fractal architecture where:

| Layer | Name | Role |
|-------|------|------|
| **Meta^0** | NeuronLayer | Raw neural substrate (weights, activations, spikes) |
| **Meta^1** | SynapseOptimizer | Self-tunes weights via STDP/Hebbian/pruning |
| **Meta^2** | LearningRuleSelector | Chooses which learning rule to use (bandit) |
| **Meta^3** | RepresentationOptimizer | Optimizes how information is encoded |
| **Meta^4** | GoalHierarchy | Manages short→medium→long term rewards |
| **Meta^5** | ArchitectureSearch | NAS-lite over the fractal stack |
| **Meta^N** | LLMInterface | LLM/Agent reasoning about the whole system |

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Meta^N: LLM Interface        │
                    │  Observes, reasons, suggests         │
                    └──────────┬──────────┬───────────────┘
                               │  ↑↓      │
                    ┌──────────┴──────────┴───────────────┐
                    │       Meta^5: Architecture Search     │
                    │  Evolves structural configurations    │
                    └──────────┬──────────┬───────────────┘
                               │  ↑↓      │
                    ┌──────────┴──────────┴───────────────┐
                    │       Meta^4: Goal Hierarchy          │
                    │  Short/medium/long rewards + curiosity │
                    └──────────┬──────────┬───────────────┘
                               │  ↑↓      │
                    ┌──────────┴──────────┴───────────────┐
                    │   Meta^3: Representation Optimizer    │
                    │  Sparsity, decorrelation, compression │
                    └──────────┬──────────┬───────────────┘
                               │  ↑↓      │
                    ┌──────────┴──────────┴───────────────┐
                    │    Meta^2: Learning Rule Selector     │
                    │  UCB bandit over {STDP, Hebbian, ...} │
                    └──────────┬──────────┬───────────────┘
                               │  ↑↓      │
                    ┌──────────┴──────────┴───────────────┐
                    │      Meta^1: Synapse Optimizer        │
                    │  STDP + Hebbian + Dopamine + Pruning  │
                    └──────────┬──────────┬───────────────┘
                               │  ↑↓      │
                    ┌──────────┴──────────┴───────────────┐
                    │       Meta^0: Neuron Layer            │
                    │  Spiking neurons, sparse weights      │
                    │  Spatial organization, Dale's law      │
                    └─────────────────────────────────────┘
```

### Signal Flow

```
    Every layer sends signals UP (data) and DOWN (guidance):

    UP signals:   State updates, performance metrics, gradients
    DOWN signals: Suggestions, commands, reward
    LATERAL:      Coordination between neighboring layers
    BROADCAST:    System-wide announcements

    ┌─────┐  SUGGEST  ┌─────┐  ACCEPT   ┌─────┐
    │  L5  │ ────────→ │  L1  │ ────────→ │  L5  │
    │      │           │      │  or       │      │
    │      │           │      │  REJECT   │      │
    └─────┘           └─────┘  ────────→ └─────┘

    The ACCEPT/REJECT protocol:
    1. Higher layer sends SUGGESTION with payload
    2. Target layer evaluates: score = _evaluate_suggestion()
    3. If score >= threshold → ACCEPT and apply
    4. If score >= 0.5*threshold → NEGOTIATE (counter-proposal)
    5. If score < 0.5*threshold → REJECT with reason
```

### Holographic Property

```
    ┌────────────────────────────────────────────┐
    │           HOLOGRAPHIC STATE                  │
    │                                              │
    │  Every layer's snapshot contains a           │
    │  compressed projection of ALL layers.         │
    │                                              │
    │  Layer 0 ──→ [████░░░░] ──→ Combined ←──    │
    │  Layer 1 ──→ [░░████░░] ──→   State   ──→   │
    │  Layer 2 ──→ [░░░░████] ──→           ──→   │
    │                                              │
    │  Any layer can approximately reconstruct     │
    │  any other layer's state from the combined   │
    │  holographic projection.                     │
    │                                              │
    │  Uses random projections (J-L lemma) for     │
    │  distance-preserving compression.            │
    └────────────────────────────────────────────┘
```

## 🚀 Quick Start

```python
from throng3.pipeline import MetaNPipeline
import numpy as np

# Create a full Meta^N pipeline
pipeline = MetaNPipeline.create_default(
    n_neurons=1000,
    n_inputs=64,
    n_outputs=32,
)

# Run optimization steps
for step in range(1000):
    x = np.random.randn(64)         # Input
    y = some_target_function(x)      # Target
    reward = compute_reward()        # External reward
    
    result = pipeline.step(x, target=y, reward=reward)
    print(f"Step {step}: loss={result['loss']:.4f}")

# Get system report
print(pipeline.get_report())
```

### Minimal Pipeline (Meta^0-2 only)

```python
pipeline = MetaNPipeline.create_minimal(
    n_neurons=100,
    n_inputs=16,
    n_outputs=8,
)
```

### With LLM-in-the-Loop

```python
def my_llm_callback(observation):
    # Your LLM analyzes the system state
    # Returns suggestions
    return {
        'suggestions': [{
            'target_level': 4,
            'payload': {'exploration_rate': 0.2},
        }],
    }

pipeline = MetaNPipeline.create_default(
    n_neurons=1000,
    include_llm=True,
    llm_callback=my_llm_callback,
)
```

## 📁 Project Structure

```
throng3/
├── throng3/
│   ├── core/
│   │   ├── signal.py          # Signal protocol (UP/DOWN/LATERAL)
│   │   ├── meta_layer.py      # Abstract base class
│   │   ├── fractal_stack.py   # Layer composition & routing
│   │   └── holographic.py     # Holographic state encoding
│   ├── layers/
│   │   ├── meta0_neuron.py    # Spiking neural substrate
│   │   ├── meta1_synapse.py   # STDP/Hebbian/pruning
│   │   ├── meta2_learning_rule.py  # Bandit rule selection
│   │   ├── meta3_representation.py # Encoding optimization
│   │   ├── meta4_goal.py      # Multi-timescale rewards
│   │   ├── meta5_architecture.py   # Evolutionary NAS
│   │   └── meta_n_llm.py      # LLM/Agent interface
│   ├── learning/
│   │   ├── stdp.py            # Spike-timing plasticity
│   │   ├── hebbian.py         # Hebbian + Oja's + BCM
│   │   ├── dopamine.py        # Three-factor learning
│   │   └── pruning.py         # Nash equilibrium pruning
│   ├── pipeline.py            # High-level API
│   └── utils.py               # Metrics, timers, utilities
├── tests/                     # Test harnesses (7 tests)
├── examples/                  # Usage examples
├── requirements.txt
└── setup.py
```

## 🔬 Accept/Reject Protocol

The core innovation of Meta^N is that every layer is autonomous.
Higher layers can SUGGEST changes, but lower layers can REJECT them.

```python
class MetaLayer(ABC):
    def accept_reject(self, signal) -> AcceptRejectDecision:
        score, reason = self._evaluate_suggestion(signal.payload)
        
        if score >= threshold:
            return AcceptRejectDecision(accepted=True, ...)
        elif score >= threshold * 0.5:
            counter = self._generate_counter_proposal(...)
            return AcceptRejectDecision(accepted=False, counter_proposal=counter)
        else:
            return AcceptRejectDecision(accepted=False, reason=reason)
```

This prevents catastrophic interference: a Meta^5 architecture
change can't destroy Meta^0's carefully learned weights if Meta^0
determines the change is too disruptive.

## 🧪 Test Harnesses

| Test | Description |
|------|-------------|
| `test_single_layer.py` | Meta^0 forward pass, optimization, snapshot |
| `test_synapse_optimization.py` | STDP, Hebbian, pruning, dopamine modulation |
| `test_cross_scale.py` | Signal routing, holographic reconstruction |
| `test_learning_rule_selection.py` | UCB bandit, rule switching |
| `test_superexponential.py` | Performance vs # meta-layers |
| `test_llm_loop.py` | LLM observation, alerts, callback |
| `test_transfer.py` | Task transfer, holographic state persistence |

## 📊 Key Concepts from throng2

Built on throng2's proven foundation:
- **Event-based processing** (293K× efficiency)
- **KDTree spatial indexing** (6862× speedup)  
- **STDP + Hebbian** learning rules
- **Dopamine reward signaling** with TD learning
- **Nash equilibrium pruning** for efficient networks
- **Actor-critic** action policy

throng3 elevates these from fixed mechanisms to self-optimizing layers
in a recursive meta-hierarchy.

## 🎯 Design Principles

1. **Every layer is autonomous**: Accept/reject protocol prevents unwanted changes
2. **Holographic redundancy**: Any slice contains the whole (for recovery)
3. **Recursive optimization**: Each layer optimizes itself AND negotiates with others
4. **Minimal coupling**: Layers communicate through signals, not direct access
5. **Fractal structure**: The same pattern repeats at every scale
6. **Graceful degradation**: System works with any subset of meta-layers

## License

Research code — Edgar Lab
