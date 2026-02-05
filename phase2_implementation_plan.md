# Phase 2: Transfer Learning Benchmark - Implementation Plan

## Goal

Create a standardized benchmark framework to **statistically prove** that Meta^N transfer learning works, with p<0.05 significance and >1.5x speedup.

---

## Integration with Prior Work

### Phase 0 Dependencies (Core)
- `MetaNPipeline.create_adaptive()` — Main agent to benchmark
- `GlobalDynamicsOptimizer` — Provides complexity & gate metrics
- `FractalStack.get_snapshot()` — For state preservation across tasks

### Phase 1 Dependencies (Environments)
- [GridWorldAdapter](file:///c:/Users/avata/aicompete/throng3/throng3/environments/gym_envs.py#8-76) — Simple task (2D input, 4 actions)
- [CartPoleAdapter](file:///c:/Users/avata/aicompete/throng3/throng3/environments/gym_envs.py#78-123) — Medium task (4D input, 2 actions)
- [MountainCarAdapter](file:///c:/Users/avata/aicompete/throng3/throng3/environments/gym_envs.py#125-169) — Hard task (2D input, 3 actions)

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    TransferBenchmark                        │
│  • Orchestrates full experiment                             │
│  • Manages seeds, configs, and results                      │
└───────────────────────────┬────────────────────────────────┘
                            │
         ┌──────────────────┴──────────────────┐
         ▼                                     ▼
┌─────────────────────┐              ┌─────────────────────┐
│   BenchmarkRunner   │              │ StatisticalAnalyzer │
│  • train_on_task()  │              │  • compute_speedup()│
│  • measure_steps()  │───results───▶│  • t_test()         │
│  • save_checkpoint()│              │  • effect_size()    │
└─────────────────────┘              └─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  EnvironmentAdapter │ ← From Phase 1
│  (GridWorld, etc.)  │
└─────────────────────┘
```

---

## Proposed Changes (Ordered by Build Sequence)

### Stage 1: Data Structures (Low Risk)

#### [NEW] `throng3/benchmarks/__init__.py`
Package initialization.

#### [NEW] `throng3/benchmarks/config.py`
```python
@dataclass
class TaskConfig:
    name: str
    env_class: Type[EnvironmentAdapter]
    max_steps: int = 5000
    convergence_threshold: float = 0.1
    
@dataclass  
class ExperimentConfig:
    tasks: List[TaskConfig]
    n_seeds: int = 30
    pretrain_steps: int = 1000
```

**Test after Stage 1:** Import config, create instances, serialize/deserialize.

---

### Stage 2: Benchmark Runner (Medium Risk)

#### [NEW] `throng3/benchmarks/runner.py`

```python
class BenchmarkRunner:
    def __init__(self, pipeline: MetaNPipeline, env: EnvironmentAdapter)
    
    def train_until_convergence(self, max_steps: int) -> TrainResult:
        """Train until loss < threshold or max_steps reached."""
        
    def measure_steps_to_convergence(self) -> int:
        """Key metric: how many steps to reach threshold?"""
        
    def save_checkpoint(self, path: str) -> None:
        """Save pipeline state for transfer."""
        
    def load_checkpoint(self, path: str) -> None:
        """Restore pipeline state."""
```

**Bug Mitigation:**
- Use rolling average for convergence (not single-point)
- Timeout protection (max_steps hard limit)
- Checkpoint validation after save/load round-trip

**Test after Stage 2:** 
- Train on GridWorld for 100 steps → verify loss decreases
- Save/load checkpoint → verify weights match
- Convergence detection → verify correct step count

---

### Stage 3: Statistical Analyzer (Low Risk)

#### [NEW] `throng3/benchmarks/stats.py`

```python
class StatisticalAnalyzer:
    @staticmethod
    def compute_speedup(pretrained_steps: List[int], 
                        fresh_steps: List[int]) -> float:
        """Mean(fresh) / Mean(pretrained)."""
        
    @staticmethod
    def t_test(pretrained: List[int], fresh: List[int]) -> Tuple[float, float]:
        """Returns (t_statistic, p_value)."""
        
    @staticmethod
    def effect_size(pretrained: List[int], fresh: List[int]) -> float:
        """Cohen's d for effect magnitude."""
        
    @staticmethod
    def confidence_interval(samples: List[float], alpha: float = 0.05) -> Tuple[float, float]:
        """95% CI for mean."""
```

**Bug Mitigation:**
- Validate sample sizes before computation
- Handle edge cases (zero variance, empty lists)
- Use scipy.stats for statistical tests (battle-tested)

**Test after Stage 3:**
- Known inputs → verify correct p-values
- Edge cases → verify graceful handling

---

### Stage 4: Transfer Benchmark (Orchestrator)

#### [NEW] `throng3/benchmarks/transfer.py`

```python
class TransferBenchmark:
    def __init__(self, config: ExperimentConfig)
    
    def run_experiment(self) -> ExperimentResults:
        """Run full N-seed experiment."""
        for seed in range(config.n_seeds):
            # 1. Fresh agent on target task
            fresh_steps = self._train_fresh(seed)
            
            # 2. Pretrain on source tasks, then transfer
            pretrained_steps = self._train_with_transfer(seed)
            
        return self._analyze_results()
    
    def _train_fresh(self, seed: int) -> int:
        """Train from scratch."""
        
    def _train_with_transfer(self, seed: int) -> int:
        """Pretrain → transfer → measure."""
```

**Bug Mitigation:**
- Seed management: Use `np.random.seed(seed)` at each trial start
- Isolation: Create fresh pipeline for each trial
- Progress logging: Print every seed completion
- Intermediate saves: Write results after each seed (crash recovery)

---

### Stage 5: Test Suite

#### [NEW] `tests/test_benchmarks.py`

| Test | What It Verifies |
|------|------------------|
| `test_config_creation()` | Config dataclasses work |
| `test_runner_basic()` | Runner trains without crash |
| `test_checkpoint_roundtrip()` | Save/load preserves state |
| `test_convergence_detection()` | Correct step counting |
| `test_stats_known_values()` | Statistical functions correct |
| `test_stats_edge_cases()` | Handles empty/zero inputs |
| `test_mini_experiment()` | N=2 seeds, sanity check |

---

## Verification Plan

### Automated Testing (After Each Stage)

```powershell
# Stage 1
python -c "from throng3.benchmarks.config import TaskConfig, ExperimentConfig; print('✓ Config OK')"

# Stage 2
python -c "from throng3.benchmarks.runner import BenchmarkRunner; print('✓ Runner OK')"

# Stage 3
python -c "from throng3.benchmarks.stats import StatisticalAnalyzer; print('✓ Stats OK')"

# Stage 4
python tests/test_benchmarks.py

# Full regression
python tests/test_environments.py
python tests/test_global_dynamics.py
```

### Pilot Study (N=5)

Before running full N=30:
```powershell
python -c "
from throng3.benchmarks import TransferBenchmark, ExperimentConfig, TaskConfig
from throng3.environments import GridWorldAdapter, CartPoleAdapter

config = ExperimentConfig(
    tasks=[
        TaskConfig('gridworld', GridWorldAdapter, max_steps=500),
        TaskConfig('cartpole', CartPoleAdapter, max_steps=1000),
    ],
    n_seeds=5,
)
benchmark = TransferBenchmark(config)
results = benchmark.run_experiment()
print(results.summary())
"
```

**Expected:** Non-zero speedup (even if not significant with N=5)

---

## Bug Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| Random seed drift | Reset seed at each trial start |
| Memory leaks in long runs | Create fresh pipeline per trial |
| Checkpoint corruption | Validate round-trip immediately |
| Statistical errors | Use scipy.stats, not hand-rolled |
| Convergence never reached | Hard max_steps timeout |
| NaN/Inf in training | Check and log warnings |
| Results lost on crash | Save after each seed |

---

## Implementation Order

| Step | File | Tests | Est. Time |
|------|------|-------|-----------|
| 1 | `config.py` | Import test | 15 min |
| 2 | `runner.py` | `test_runner_*` | 1 hour |
| 3 | `stats.py` | `test_stats_*` | 30 min |
| 4 | [transfer.py](file:///c:/Users/avata/aicompete/throng3/tests/test_transfer.py) | `test_mini_experiment` | 1 hour |
| 5 | Full test suite | All | 30 min |
| 6 | Pilot study (N=5) | Manual | 1 hour |

**Total Estimate:** ~4-5 hours

---

## Review/Debug Plan (Post Phase 2)

### Code Review Checklist

- [ ] **Seed Isolation:** Each trial uses independent seed
- [ ] **Memory Safety:** No retained references across trials
- [ ] **Checkpoint Integrity:** Round-trip test passes
- [ ] **Statistical Validity:** P-values match scipy reference
- [ ] **Edge Cases:** Empty list/zero variance handled
- [ ] **Logging:** Each stage prints progress
- [ ] **Error Messages:** Clear, actionable on failure

### Debug Procedure

1. **Run Mini-Experiment (N=2)**
   ```powershell
   python -c "from tests.test_benchmarks import test_mini_experiment; test_mini_experiment()"
   ```

2. **Check for Warnings**
   ```powershell
   python -W all tests/test_benchmarks.py 2>&1 | findstr /i "warning"
   ```

3. **Memory Profiling (if issues)**
   ```powershell
   python -c "
   import tracemalloc
   tracemalloc.start()
   # ... run experiment ...
   snapshot = tracemalloc.take_snapshot()
   for stat in snapshot.statistics('lineno')[:10]:
       print(stat)
   "
   ```

4. **Statistical Sanity Check**
   - Compare results against hand-calculated values
   - Verify p-values are in expected range (0-1)
   - Check effect sizes are reasonable (0.2 small, 0.5 medium, 0.8 large)

5. **Regression Verification**
   ```powershell
   python tests/test_environments.py
   python tests/test_global_dynamics.py
   python tests/test_single_layer.py
   ```

### Success Criteria

| Metric | Target | Acceptable |
|--------|--------|------------|
| All tests pass | 100% | 100% |
| Pilot speedup | >1.0x | Any positive |
| P-value (N=5 pilot) | <0.20 | <0.50 |
| Memory growth | 0 | <10 MB/trial |
| Runtime (N=5) | <10 min | <30 min |

---

## Decision Points

After Phase 2 implementation:

1. **If pilot shows positive transfer:** Proceed to Phase 3 (full N=30 run)
2. **If pilot shows no transfer:** Debug, check pipeline state preservation
3. **If tests fail:** Fix before proceeding, document issues
4. **If memory issues:** Refactor to use generators/streaming
