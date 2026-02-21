# PROJECT_STATE.md

> **Context handoff file.** Read this first when resuming work. Last updated: 2026-02-20 21:48 CST.
> Owner: Mike + Antigravity (AI coding assistant)

---

## 1) Canonical Paths

| What | Path |
|------|------|
| Repo root | `C:\Users\avata\aicompete\throng3` |
| OpenClaw root | `~/.openclaw` (override: `OPENCLAW_WORKSPACE_ROOT`) |
| Memory/requests | `~/.openclaw/workspace/memory/` |
| Rule libraries | `~/.openclaw/rules/` |
| Blind label registry | `~/.openclaw/workspace/blind_label_registry.json` |
| Replay DB (new) | `experiments/replay_db.sqlite` *(not yet created — next session)* |

Path config centralized in: **`throng4/config.py`**

---

## 2) Architecture Snapshot (as of 2026-02-20)

### The two-layer abstract feature system
Every environment adapter maps its state to a **fixed 84-dim portable vector**:
```
[core(20) | ext*mask(32) | mask(32)]  → 84-dim input to PortableNN
```
- `core` (indices 0–19): Universal schema — agent_x/y, target_x/y, threat_prox, reward_prox, rsrc, density + velocities
- `ext` (indices 20–51): Adapter-specific extension (masked out = 0.0 during training noise)
- `mask` (indices 52–83): Binary validity flags for each ext slot

Ext noise (`std=0.02`) is injected during `_train_batch` only — eval paths are clean.

### Blind hypothesis generation
Tetra sees **anonymized logs only** (`Environment-A`, `Environment-B`, ...) — never real game names.
The mapping persists across restarts at `~/.openclaw/workspace/blind_label_registry.json`.

### PortableNNAgent
Off-policy DQN. API:
```python
features = adapter.make_features(action)   # BEFORE step()
obs, reward, done, info = adapter.step(action)
next_features = [adapter.make_features(a) for a in adapter.get_valid_actions()]
agent.record_step(features, reward, next_features, done)
agent.end_episode(final_score)
```

---

## 3) What Was Built This Session (2026-02-20)

### Abstract feature layer
- **`throng4/learning/abstract_features.py`** — `AbstractFeature` dataclass, `CORE_SIZE=20`, `EXT_MAX=32`, `ABSTRACT_VEC_SIZE=84`, `to_vector()`, `blind_log_str()`, `make_ext()`, `empty_core()`, **`assert_mask_binary()`** (mask integrity check)
- **`throng4/environments/adapter.py`** — base class updated with abstract feature protocol (`get_core_features`, `get_ext_features`, `get_abstract_features`, `get_blind_obs_str`)
- **`throng4/environments/tetris_adapter.py`** — full abstract feature impl (core maps board heuristics, ext = per-column heights + bumpiness, 8 slots)
- **`throng4/environments/atari_adapter.py`** — full abstract feature impl (core maps Breakout RAM, ext = 5 Breakout-specific slots)

### Blind protocol hardening (`throng4/config.py`)
New constants added:
- `REQUIRED_HYPOTHESIS_KEYS` — now includes `trigger` + `generality`
- `VALID_GENERALITY_VALUES = {"universal", "class", "instance"}`
- `VALID_DIRECTION_VALUES = {"maximize", "minimize", "increase", "decrease", "avoid"}`
- `BLIND_GAME_STRINGS` — set of banned game-identity strings (Breakout, Paddle, Ball, Lives, Tetris, board, piece…)
- `LABEL_REGISTRY_PATH` — path to persistent blind label registry JSON

### Offline generator (`throng4/llm_policy/offline_generator.py`)
- `_load_label_registry()` / `_save_label_registry()` — persist mapping on disk, load on import
- `_get_blind_label(game_id)` — stable `Environment-A/B/C…` labels
- `compress_trajectory()` — prefers `blind_obs` field over `obs`; detects `rsrc:` and `Lives:` for RESOURCE LOST annotation
- `_build_prompt()` — uses blind label, runs `check_blindness_leak()` before sending
- `_validate_hypotheses()` — checks generality value, confidence range
- `check_blindness_leak(payload, label)` — static method, raises on any banned string

### Tetra prompt (`experiments/TETRA_PROMPT.md`)
- **Blind Generalization Mode** section added — triggered by `Environment-X` headers, rules for abstract vocab only, `generality` field required

### New top-level scripts
| Script | Purpose |
|--------|---------|
| `generate_blind_logs.py` | Run Tetris/Atari episodes → `blind_obs` trajectory JSON |
| `validate_blind_hypotheses.py` | 4-layer gate → ingest + SQLite snapshot |
| `test_blind_protocol.py` | 11 unit tests (all pass) |

### Schema files (for Human Play + Replay DB)
| File | What |
|------|------|
| `experiments/HUMAN_PLAY_SCHEMA.md` | JSONL per-timestep schema (Tetra v1) + throng4 adaptations |
| `experiments/REPLAY_DB_SCHEMA.sql` | SQLite DDL: sessions/episodes/transitions/transition_metrics/views |

**Note:** The schema examples show `"game": "antigravity"` — that's a confusion artifact (Antigravity is the AI coding assistant name, not a game). Use `"tetris"` or `"ALE/Breakout-v5"`.

---

## 4) Recent Commits (this session)

```
687f99a  Save Tetra replay DB schema
c5fa525  Save Tetra human play logger schema
0bb02ec  Add generate_blind_logs.py + first blind Tetris trajectory
5d85988  Add validate_blind_hypotheses.py: 4-layer gate
834d3ee  Harden blind protocol: generality validation, mask integrity, leak test, registry persistence
2fafb33  Complete blind abstract feature loop: ext_noise in training, game anonymization, blind_obs logs
```

---

## 5) What To Build Next (priority order)

### A) Human Play Logger + Replay DB  ← START HERE
Tetra has provided the full schema. Now implement:

1. **`throng4/storage/human_play_logger.py`**
   - Opens `experiments/replay_db.sqlite` using `experiments/REPLAY_DB_SCHEMA.sql`
   - `log_step(session_id, episode_id, step_idx, state_vec, human_action, agent_action, executed_action, reward, done, **kwargs)` → writes to `transitions`
   - `close_episode(episode_id, total_reward, total_steps)` → writes to `episodes`
   - `compute_derived(episode_id)` → backward pass after episode: n_step returns, time_to_terminal, near_death_flag, recovery_flag, human_agent_disagree, priority_raw

2. **`throng4/learning/prioritized_replay.py`**
   - Replace uniform `ReplayBuffer` in `PortableNNAgent`
   - Priority formula (v1, before value head): `(1.5 if human_agent_disagree else 1.0) * (2.0 if near_death_flag else 1.0) * (1.0 + novelty_score)`
   - Stratified sampler: bucket by flag type + priority-weighted random

3. **Imitation head stub in `PortableNNAgent`**
   - Small 2-layer adapter on top of frozen backbone
   - `AgentConfig.use_imitation_head: bool = False`
   - `L_imitation = cross_entropy(imitation_logits, human_action)` — only updated when `human_action` is not None

### B) Composite loss schedule
```
Phase 1: L = L_imitation only
Phase 2: L = α*L_imitation + β*L_value
Phase 3: Full composite (needs cross-env replay data first)
```

### C) Telemetry from Tetra still needed
Tetra offered: canonical INSERT statements + priority sampler query (stratified buckets).
Get these before implementing `human_play_logger.py` if you want to verify query patterns.

---

## 6) Fast Restart Checklist

```powershell
cd C:\Users\avata\aicompete\throng3

# 1. Verify config paths
python -m throng4.config

# 2. Verify blind protocol
python test_blind_protocol.py   # should print 11/11 passed

# 3. Verify label registry loaded
python -c "from throng4.llm_policy.offline_generator import _GAME_LABELS; print(_GAME_LABELS)"

# 4. Generate a fresh blind log (Tetris, 3 eps)
python generate_blind_logs.py --envs tetris --episodes 3 --steps 80 --out-dir blind_logs

# 5. Run validate gate on any existing hypotheses
python validate_blind_hypotheses.py <path_to_hypotheses.json> tetris

# 6. Git state
git log --oneline -6
git status --short
```

---

## 7) High-Value Files Index

### Config / contracts
| File | Purpose |
|------|---------|
| `throng4/config.py` | All paths + validation constants |
| `experiments/TETRA_PROMPT.md` | Tetra's full prompt contract (includes Blind Generalization Mode) |
| `experiments/HUMAN_PLAY_SCHEMA.md` | JSONL schema for human play logger (v1) |
| `experiments/REPLAY_DB_SCHEMA.sql` | SQLite DDL for replay DB |

### Core implementation
| File | Purpose |
|------|---------|
| `throng4/learning/abstract_features.py` | 84-dim portable feature system |
| `throng4/learning/portable_agent.py` | PortableNNAgent (DQN + ext_noise) |
| `throng4/environments/adapter.py` | Base adapter with abstract feature protocol |
| `throng4/environments/tetris_adapter.py` | Tetris impl |
| `throng4/environments/atari_adapter.py` | Atari/Breakout impl |
| `throng4/llm_policy/offline_generator.py` | Blind hypothesis generation pipeline |

### Scripts
| File | Purpose |
|------|---------|
| `generate_blind_logs.py` | Run envs → blind trajectory JSON |
| `validate_blind_hypotheses.py` | Gate: validate + ingest + SQLite snapshot |
| `test_blind_protocol.py` | 11 unit tests for the blind protocol |
| `train_tetris_curriculum.py` | Curriculum training (existing) |

---

## 8) SQLite databases

| DB | Table(s) | Purpose |
|----|---------|---------|
| `experiments/experiments.db` | `episodes`, `events`, `blind_hypothesis_log` | Existing experiment tracking + hypothesis log |
| `experiments/replay_db.sqlite` | `sessions`, `episodes`, `transitions`, `transition_metrics` | **NEW — not created yet** — Human play + prioritized replay |

---

## 9) Reality Check

Repo has 11k+ files (historical copies). Use only `throng4/` as the active codebase.
The blind label registry at `~/.openclaw/workspace/blind_label_registry.json` must survive if you want longitudinal cross-session analysis — **do not delete it**.
