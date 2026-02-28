# Learner Selection Methodology — Design Notes

> **Status**: Ideas captured, not yet implemented. Revisit after Phase 4.
> **Referenced from**: session telemetry logs, `brain/learning/learner_selector.py`

## Current State (Phase 3)

Rule-based scoring: env fingerprint -> algorithm recommendation.
Works but uses hardcoded opinions, not empirical data.

## Proposed Evolution: Statistical Evolutionary Testing

**User concept** (2026-02-25):

1. **Probe all algorithms** with statistical sampling using an evolutionary-style
   testing method, where wins **deprioritize** the statistical likelihood of that
   algorithm being selected for further testing (it's already proven — let others
   catch up or fail fast).

2. **Stick with a winner until plateau** — once a method works best, commit to it.
   Only reconsider when performance plateaus.

3. **LLM-assisted re-evaluation** — when a plateau is hit, the PortableNN can
   recommend to the LLM (via Prefrontal Cortex) that additional algorithm testing
   should be run. This keeps the human out of the loop.

4. **Per-area specialization** — different areas of a game level may benefit from
   different learners (e.g., exploration-heavy area = PPO-ICM, exploitation area = DQN).
   The system should be flexible enough to switch learners per game stage.

5. **Promotion thresholds** — when an algorithm crosses a win-rate threshold:
   - **>30% plurality**: keep testing others but this one gets more allocation
   - **>50% majority**: promote to 100% allocation (stop testing, commit)
   - Rationale: if data collection and continued training beats resetting to
     try a new algorithm, then commit early

## Implementation Plan (for later)

```
Phase A: Probe Runner
  - Run N=500 steps with top-3 candidate algorithms simultaneously
  - Score by reward slope (learning speed), not just final reward
  - Elimination: drop worst performer after each probe round

Phase B: Bandit with Promotion
  - UCB1 or Thompson Sampling over (env_hash, algorithm) -> performance
  - Track win rates per algorithm
  - Promotion logic: >50% win rate for 30+ episodes -> commit to 100%
  
Phase C: Stage-Aware Selection
  - Cluster game states into "stages" (via Hippocampus state compression)
  - Track per-stage win rates separately
  - Allow different learners for different stages

Phase D: LLM Re-evaluation
  - On plateau detection (RiskSensor), Prefrontal queries Tetra:
    "Performance plateaued at reward=X after Y episodes using algorithm Z.
     Should we re-run the probe phase?"
  - Tetra can suggest specific algorithms based on pattern analysis
```
