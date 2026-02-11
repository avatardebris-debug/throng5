# Throng3 — Project Status & Recovery Guide
**Last Updated:** 2026-02-10 (Phase 2 task conditioning complete)

---

## 🏗️ Architecture Overview

```
throng3/          ← Core Meta^N recursive self-optimization (COMPLETE, proof-of-concept)
throng35/         ← Regional brain architecture (Striatum/Cortex/Hippocampus/Executive)
```

### Key Modules (throng3/)
| Path | Purpose |
|------|---------|
| `throng3/core/meta_layer.py` | Abstract base class — all layers inherit from `MetaLayer` |
| `throng3/core/fractal_stack.py` | Layer composition & signal routing |
| `throng3/core/holographic.py` | Holographic state encoding (J-L projections) |
| `throng3/core/signal.py` | Signal protocol (UP/DOWN/LATERAL/BROADCAST) |
| `throng3/pipeline.py` | High-level API (`MetaNPipeline`) |
| `throng3/layers/meta0_neuron.py` | Meta^0: Spiking neural substrate |
| `throng3/layers/meta1_synapse.py` | Meta^1: STDP/Hebbian/pruning |
| `throng3/layers/meta2_learning_rule.py` | Meta^2: UCB bandit rule selection |
| `throng3/layers/meta3_consolidation.py` | Meta^3: EWC (Elastic Weight Consolidation) ✅ |
| `throng3/layers/meta3_maml.py` | Meta^3: Task-Conditioned MAML ✅ |
| `throng3/config/maml_config.py` | MAML configuration dataclass |

---

## ✅ What's Validated & Working

### GridWorld RL Training
- **ε-greedy Q-learning** achieves **0.861 mean reward** (standalone)
- Q-learning + curriculum: **100% success** standalone

### EWC Baseline (Compound Transfer)
- **+0.250 mean transfer**, **67% positive transfer** ← VALIDATED BASELINE
- Uses `WeightConsolidation` at Meta^3 (`meta3_consolidation.py`)
- Factory method: `MetaNPipeline.create_with_ewc()`
- Test: `test_compound_corrected.py`

### Fisher Boosting Experiments
- **Complete, informative failure** — proved supervised ≠ RL transfer
- Hand-coded heuristics (boost high-Fisher for supervised, invert for RL) don't generalize
- **Key insight:** Need *learned* strategies, not hand-coded ones → motivates MAML

### Core Components
- FractalStack, MetaLayer, HolographicState, Signal system — all functional
- Learning rules (QLearner, STDP, Hebbian) work individually
- Environments (GridWorld, FrozenLake) adapters work

---

## ✅ MAML Infrastructure (Phase 1 Complete)

### Files
| File | Status |
|------|--------|
| `throng3/layers/meta3_maml.py` | ✅ Fixed — follows consolidation init pattern |
| `throng3/config/maml_config.py` | ✅ Correct |
| `test_maml_basic.py` | ✅ Passing — inner + outer loop validated |

### What Was Fixed
- Init bug: dict config passed to `super().__init__()`, typed config stored as `self.maml_config`
- Added missing abstract methods (`forward`, `_compute_state_vector`, `_apply_suggestion`, etc.)
- Test: used correctly-shaped task weights `(1,4)` instead of NeuronLayer's `(1,50)`

### Phase 2: Task Conditioning (wired)
- Meta^2 signals `task_type` + `confidence` UP to Meta^3 via `SignalType.PERFORMANCE`
- MAML extracts task type in `_handle_performance_update()` override
- Fallback `TaskDetector` for first steps before any signals arrive
- `_resolve_task_type()` priority: explicit context → Meta^2 signal → fallback detector → default
- Test: `test_maml_task_conditioning.py` — 5 tests covering detection, switching, divergence, fallback, pipeline integration

### Factory Method
- `MetaNPipeline.create_with_maml()` in `pipeline.py` (lines 123-150)
- Constructs: Meta^0 (Neuron) → Meta^1 (Synapse) → Meta^2 (LearningRule) → Meta^3 (MAML)
- Called by `test_maml_basic.py`

---

## 📋 MAML Implementation Plan

### Phase 1: Basic MAML ✅
- [x] Create `MAMLConfig` dataclass
- [x] Create `TaskConditionedMAML` class with inner/outer loop
- [x] Create `create_with_maml()` factory method
- [x] Create `test_maml_basic.py`
- [x] **Fix init bug** (dict config to parent, `self.maml_config`)
- [x] Run `test_maml_basic.py` — inner loop adapts weights (L2=0.091)
- [x] Run `test_maml_basic.py` — outer loop meta-update executed

### Phase 2: Task Conditioning ✅
- [x] Wire Meta^2 task classification into MAML
- [x] Learn different strategies for supervised vs. RL tasks (L2 divergence = 0.035)
- [x] Test on mixed task curriculum

### Phase 3: GridWorld Validation ← **YOU ARE HERE**
- [ ] Run MAML on GridWorld RL transfer
- [ ] Compare against EWC baseline (+0.250 mean)
- [ ] Goal: MAML should outperform EWC on RL tasks

### Phase 4: Analysis
- [ ] Visualize learned lr_multipliers per task type
- [ ] Compare MAML vs Fisher boosting strategies
- [ ] Document what MAML learns that hand-coding couldn't

---

## 🔑 Key Insights (Don't Forget These)

1. **Supervised ≠ RL transfer** — Fisher boosting experiments proved that what works for supervised transfer hurts RL transfer and vice versa.
2. **Hand-coded heuristics fail** — We tried boosting shared features (supervised) and inverting for policy stability (RL). Neither generalizes.
3. **EWC-only is the baseline to beat** — +0.250 mean, 67% positive. Any new approach must exceed this.
4. **Pipeline timing mismatch** — throng3's single `step()` call is incompatible with RL's action→reward→learn loop (documented in `THRONG3_COMPLETE.md`). throng35 fixes this with regional architecture.

---

## 🧪 How to Test

```bash
# Activate venv
cd c:\Users\avata\aicompete\throng3
.\.venv\Scripts\activate

# Run MAML test (after fixing the bug)
python test_maml_basic.py

# Run EWC baseline (validated, should still pass)
python test_compound_corrected.py

# Run function family test
python test_function_family.py
```

---

## 📁 Important File Locations

| What | Where |
|------|-------|
| MAML layer (buggy) | `throng3/layers/meta3_maml.py` |
| MAML config | `throng3/config/maml_config.py` |
| MAML test | `test_maml_basic.py` |
| EWC consolidation (reference) | `throng3/layers/meta3_consolidation.py` |
| Pipeline factory | `throng3/pipeline.py` |
| MetaLayer base class | `throng3/core/meta_layer.py` |
| Architecture summary | `THRONG3_COMPLETE.md` |
| Phase 2 plan | `phase2_implementation_plan.md` |

---

## ❌ Known Limitations (Out of Scope for Now)

- **CartPole / complex RL:** Linear Q-learning can't approximate CartPole (would need neural net). Documented in conversation `c3159adb`.
- **Pipeline RL timing:** Reward passed BEFORE action in single `step()` call. Fixed in throng35 architecture.
- **STDP/Hebbian alone:** 0% success on RL (can't learn goals without reward signal).

---

**TL;DR:** Fix the MAML init bug (follow the `meta3_consolidation.py` pattern), run `test_maml_basic.py`, then move to Phase 2 task conditioning.
