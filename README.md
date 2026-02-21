# Throng3 / Throng4 — Resume Guide

> **Start here after any context reset.** Full state is in `PROJECT_STATE.md`.

**Last updated**: 2026-02-20 | **Repo**: `C:\Users\avata\aicompete\throng3`

---

## What This Project Is

AI agent that learns across multiple games using:
- **PortableNNAgent** — value-based DQN with abstract portable features
- **Abstract feature layer** — fixed 84-dim vector any game maps into (`[core | ext*mask | mask]`)
- **Tetra** — LLM collaborator generating hypotheses from blind gameplay logs
- **Blind hypothesis protocol** — Tetra never sees game names; reasons from abstract features only

---

## Right Now: Build Human Play Logger

The next thing to implement is the **human play + prioritized replay system**.
All schemas are already designed and committed:

| Schema file | What it defines |
|-------------|----------------|
| [`experiments/HUMAN_PLAY_SCHEMA.md`](experiments/HUMAN_PLAY_SCHEMA.md) | JSONL per-timestep format |
| [`experiments/REPLAY_DB_SCHEMA.sql`](experiments/REPLAY_DB_SCHEMA.sql) | SQLite tables + indexes + views |

**What to build (in order):**
1. `throng4/storage/human_play_logger.py` — JSONL writer + SQLite indexer
2. `throng4/learning/prioritized_replay.py` — replaces uniform ReplayBuffer
3. Imitation head stub in `PortableNNAgent` (`use_imitation_head` flag)

See **Section 5** of `PROJECT_STATE.md` for full spec.

---

## Quick Restart Commands

```powershell
cd C:\Users\avata\aicompete\throng3

python -m throng4.config                    # verify paths
python test_blind_protocol.py               # 11/11 should pass
git log --oneline -6                        # recent commits
```

---

## Key Components

### Abstract Features (NEW — 2026-02-20)
| File | Purpose |
|------|---------|
| `throng4/learning/abstract_features.py` | 84-dim portable vector, `assert_mask_binary()` |
| `throng4/environments/adapter.py` | Base adapter + abstract feature protocol |
| `throng4/environments/tetris_adapter.py` | Tetris → abstract features |
| `throng4/environments/atari_adapter.py` | Atari/Breakout → abstract features |

### Blind Hypothesis Pipeline (NEW — 2026-02-20)
| File | Purpose |
|------|---------|
| `throng4/config.py` | All constants: `REQUIRED_HYPOTHESIS_KEYS`, `VALID_GENERALITY_VALUES`, `VALID_DIRECTION_VALUES`, `BLIND_GAME_STRINGS`, `LABEL_REGISTRY_PATH` |
| `throng4/llm_policy/offline_generator.py` | Blind prompt builder, registry persistence, leak check |
| `experiments/TETRA_PROMPT.md` | Tetra's full prompt contract (Blind Generalization Mode section) |
| `generate_blind_logs.py` | Run envs → blind trajectory JSON |
| `validate_blind_hypotheses.py` | **One-command gate**: validate + ingest + SQLite snapshot |
| `test_blind_protocol.py` | 11 unit tests — run after any changes to config or generator |

### Core Learning
| File | Purpose |
|------|---------|
| `throng4/learning/portable_agent.py` | PortableNNAgent: off-policy DQN + ext_noise during training |
| `throng4/runners/fast_loop.py` | High-speed episode runner (no per-step logging) |
| `train_tetris_curriculum.py` | Curriculum L1-7 training script |

### OpenClaw / Tetra Integration
| File | Purpose |
|------|---------|
| `throng4/llm_policy/openclaw_bridge.py` | Real-time Tetra communication |
| `throng4/llm_policy/eval_auditor.py` | Independent reward hacking detection |

---

## Blind Label Registry

Tetra-facing anonymization: `tetris` → `Environment-B`, `ALE/Breakout-v5` → `Environment-A`

Persisted at: `~/.openclaw/workspace/blind_label_registry.json`

**Do not delete** — needed for longitudinal cross-session analysis.

---

## SQLite Databases

| DB path | Contains |
|---------|----------|
| `experiments/experiments.db` | Episode logs + `blind_hypothesis_log` table |
| `experiments/replay_db.sqlite` | **NOT YET CREATED** — human play replay DB (next session) |

Query blind hypothesis log:
```sql
SELECT ts, blind_label, total, valid_count, gen_universal, gen_class, gen_instance
FROM blind_hypothesis_log ORDER BY ts;
```

---

## Directory Structure

```
throng3/
├── throng4/                          ← active codebase
│   ├── config.py                     ← ALL path + validation constants
│   ├── environments/
│   │   ├── adapter.py                ← base adapter + abstract feature protocol
│   │   ├── tetris_adapter.py
│   │   └── atari_adapter.py
│   ├── learning/
│   │   ├── abstract_features.py      ← 84-dim portable vector
│   │   └── portable_agent.py         ← DQN agent
│   ├── llm_policy/
│   │   ├── offline_generator.py      ← blind hypothesis pipeline
│   │   ├── openclaw_bridge.py
│   │   └── eval_auditor.py
│   ├── runners/
│   │   └── fast_loop.py
│   └── storage/
│       └── [human_play_logger.py]    ← TO BUILD
├── experiments/
│   ├── TETRA_PROMPT.md               ← Tetra prompt contract
│   ├── HUMAN_PLAY_SCHEMA.md          ← JSONL schema (Tetra v1)
│   └── REPLAY_DB_SCHEMA.sql          ← SQLite DDL
├── generate_blind_logs.py            ← run envs → blind trajectory JSON
├── validate_blind_hypotheses.py      ← 4-layer gate + ingest + snapshot
├── test_blind_protocol.py            ← 11 unit tests
├── PROJECT_STATE.md                  ← full state (read this for details)
└── blind_logs/                       ← generated blind trajectory files
    └── blind_traj_tetris_4ep.json
```

---

**Full detail in**: [`PROJECT_STATE.md`](PROJECT_STATE.md)
