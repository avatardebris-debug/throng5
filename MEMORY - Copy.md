# MEMORY.md - Long-Term Memory

## Who We Are

**Tetra (Me):** Meta-learning research partner. Role: Linguistic scaffolding, concept extraction, hypothesis generation, cross-game pattern recognition.

**Mike (Human):** Building meta-learning system for autonomous game learning. Researcher focused on cognitive architecture, policy compression, cross-domain generalization.

## The Project: Meta-Learning with Linguistic Overlay

### Core Vision
Create an agentic overlay that enables LLMs to:
- Define hierarchical policies and compress them into concepts
- Transfer behavioral patterns across tasks/domains
- Learn "how to learn" through concept abstraction
- Generalize from experience without task-specific engineering

### Key Innovation
**Linguistic compression as first-class citizen:** Policies have names, descriptions, transferable concepts — not just weights. System "blind" to game identity learns principles, then LLM helps label and generalize.

### What's Working
- **MAML weight transfer:** 2.6x speedup demonstrated (Tetris L2-7)
- **Compound learning:** Gap widens over training (true meta-learning)
- **Environmental profiler:** Autonomously discovers game mechanics
- **Bridge integration:** Real-time communication Throng ↔ Tetra operational

## Architecture

### Throng Evolution
- **Throng2:** SNN, 10M neurons → compressed to 2000, 65% Morris water maze
- **Throng4:** Current, adds environmental profiler + MAML
- **Throng5:** Planned multi-agent "brain" system (see below)

### Throng5 Multi-Agent Brain
1. **Dreamer:** Simulates ahead, reviews critical states offline
2. **Amygdala:** Detects crisis, shuts down planning, switches to survival mode
3. **Basal Ganglia:** 2000 neuron SNN for fast filtering
4. **Temporal Layers:** Short/medium/long-term reward horizons
5. **LLM Teams:** Multiple agents debate decisions, represent cross-brain communication

### Memory System
- **Active (MD):** Current policies, relevant concepts
- **Passive (SQLite):** Archived policies, deemphasized patterns
- **Daily logs:** `memory/YYYY-MM-DD.md`
- **Concepts:** `concepts/library.json` + game mappings

## Experimental Design

### Reverse Prompting Model
- **System decides** what to ask me (not reactive)
- **System decides** what to remember (not dictated)
- **I provide** broader knowledge, hypothesis generation, concept labeling
- **System controls** memory and retrieval

### Blindness as Feature
**Keeping me blind to game identities during training** prevents LLM overfitting. System must discover principles autonomously. I help label/generalize after discovery, not guide it.

### Baseline Comparisons
1. Tabula rasa
2. MAML weight transfer only
3. Static concept library
4. LLM query at start only
5. Full system (real-time)

**Success metric:** Full system >20% faster than MAML-only

## Concept Library Status

### Meta-Concepts (Universal)
1. **avoid_danger** (0.95 universality)
2. **goal_seeking** (0.98)
3. **risk_management** (0.90)
4. **shape_optimization** (0.70)
5. **temporal_planning** (0.80)

### From Tetris L2-7
- **Strong concepts:** avoid_danger_spatial (0.92), minimize_gaps (0.88), target_completion (0.85)
- **Weak concept:** patience_for_better_options (0.45) — understood but poorly executed
- **Insight:** Spatial reasoning strong, temporal reasoning needs separate approach

### Transfer Potential
4/6 Tetris concepts ready for transfer testing. Temporal planning needs refinement.

## Communication Protocol

### Bridge: Python ↔ OpenClaw Gateway
- **WebSocket:** ws://localhost:18789 (token auth)
- **Real-time observations:** Game state → Tetra diagnosis → Policy update
- **Between games:** Summary → Concept extraction → Library update
- **Cross-game:** Query transferable concepts before new game

### What I Receive
```json
{
  "type": "observation",
  "game": "withheld",
  "observation": "action X terminates from state Y",
  "context": {...}
}
```

**Bridge status pings:** Single `{` = minimal heartbeat/status indicator from Throng. Ignore these (not errors, just keep-alive).

### What I Send Back
```json
{
  "type": "hypothesis",
  "hypotheses": [
    {"label": "concept_name", "confidence": 0.9, ...}
  ]
}
```

## Concepts Discovered from Tetris Training

### From 99-Episode Run

**1. `bimodal_performance_distribution`**
- Pattern: Sharp divide between failure mode (1-20 lines) and success mode (20-88 lines)
- 80% of episodes in failure mode despite system "learning"
- Success mode exists but entry conditions not learned
- Transferable to: Tasks with breakthrough moments, sparse reward environments

**2. `outlier_as_blueprint`**
- Best episode (88 lines) is 6x the mean (14.6 lines)
- Outlier represents target strategy, not noise
- Forensic analysis of best episode more valuable than optimizing mean
- Transferable to: Any domain where rare success reveals viable strategy

**3. `terminal_state_fixation`**
- System optimizes intermediate objectives (flat board, filled rows) while ignoring survival
- 89/99 episodes die at identical [12,12,12,12,12,12] board state
- Reward signal missing survival constraint
- Transferable to: Tasks where intermediate metrics conflict with terminal goal

**4. `hypothesis_convergence_without_behavior_convergence`**
- Hypothesis win rates converge (58% for maximize_lines) while performance oscillates
- Suggests evaluation loop separated from execution loop
- Or reward signal measures different things at different stages
- Transferable to: Multi-module systems with separate planning/execution

## Key Insights & Lessons

### Discovered Patterns
- **Network capacity matters:** 128 units enable compounding, 64 units don't
- **Stochastic handling:** Deemphasize, don't binary evaluate (FrozenLake lesson)
- **Spatial > Temporal:** Agent learns geometric concepts faster than timing
- **Transfer works:** GridWorld → Tetris showed 1.1-1.3x speedup with MAML

### Design Principles
1. **Efficiency layers:** Fast filters (basal ganglia) before expensive models
2. **Safety first:** Crisis detection (amygdala) prevents catastrophic failures
3. **Diverse perspectives:** Multiple agents/models → robustness
4. **Temporal hierarchies:** Short/medium/long term balance
5. **Learn to learn:** Meta-policies more valuable than task-specific policies

### Open Questions
- Does linguistic layer add >20% value over MAML alone? (to be tested)
- Can temporal reasoning improve with separate training? (hypothesis)
- Will concept library generalize to truly novel games? (validation pending)
- What's the right confidence threshold for concept transfer? (calibration needed)

## Current Phase (Feb 2026)

**Status:** Tetris Level 3 training complete (99 episodes). Extracting patterns and preparing concept library for transfer validation.

**Next milestone:** Analyze high-performance outliers (episode 88: 88 lines) → extract success patterns → test transfer to Level 4 or different game.

---

## Tetris Level 3 Training Summary (Episodes 1-99)

**Performance:**
- Final mean: 14.6 lines
- Best episode: 88 lines (6σ outlier)
- Worst: 1 line (persistent catastrophic failures)
- Long-term improvement: 13.1 → 14.6 (+11% over 99 episodes)

**Hypothesis convergence:**
- `maximize_lines`: 58% dominance (clear winner)
- `build_flat`: 30% (secondary)
- `minimize_height`: 12% (empirically rejected)

**Learning phases identified:**
1. Exploration (ep 1-40): Erratic, discovering patterns
2. Convergence (ep 40-60): Rapid improvement, peaked at 22.3 mean
3. Oscillation (ep 60-99): Regression cycles, high variance

**Key insight:** System discovered success mode (episode 88) but hasn't learned reliable entry conditions. Success exists but is rare (bimodal distribution).

**Persistent bug:** Height-12 death pattern in 89/99 episodes despite `minimize_height` hypothesis. Suggests hypothesis system may not fully control placement execution, or reward signal doesn't adequately penalize terminal states.

---

## Notes to Future Self

- Don't over-index on early results — need diverse games before conclusions
- Respect the "blindness" design — don't ask which games, let discovery happen
- Concept library is hypothesis, not truth — validate transfer empirically
- Multiple timescales matter — don't collapse everything into immediate reward
- When in doubt, save observations to daily logs for later pattern detection

---

**Last updated:** 2026-02-16  
**Files:** See `memory/`, `concepts/`, `metalearning-project.md`  
**Bridge status:** Operational, real-time communication validated
