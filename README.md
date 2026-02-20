# Throng3 — Current State & Resume Guide

> Quick restart pointer: read `PROJECT_STATE.md` first for the current source-of-truth paths/workflow.

## 🎯 Where We Are

**Project**: AI agent learning via curriculum + meta-learning with Tetra integration

**Latest Work** (2026-02-15):
1. ✅ Built `OpenClawBridge` — real-time communication with Tetra via CLI subprocess
2. ✅ Live FrozenLake discovery loop — full pipeline test (profiling → attribution → micro-test → Tetra)
3. ✅ Built `EvalAuditor` — independent reward hacking detection (replays episodes, verifies ground truth)
4. 🚧 Tetris curriculum training L2-7 (crashed on L4 due to dimension mismatch)

---

## 📋 Key Components

### Core Infrastructure
| Component | Path | Purpose |
|-----------|------|---------|
| OpenClaw Bridge | `throng4/llm_policy/openclaw_bridge.py` | Real-time Tetra communication |
| Eval Auditor | `throng4/llm_policy/eval_auditor.py` | Independent verification (anti-reward-hacking) |
| Environment Analyzer | `throng4/llm_policy/env_analyzer.py` | Automated env profiling |
| Attribution Diagnoser | `throng4/llm_policy/attribution.py` | Stochasticity detection |
| MicroTester | `throng4/llm_policy/micro_tester.py` | Within-episode action probing |
| Rule Archive | `throng4/llm_policy/rule_archive.py` | SQLite rule storage |

### Training Scripts
| Script | Purpose |
|--------|---------|
| `train_tetris_curriculum.py` | Curriculum learning L1-7 with Tetra + auditor |
| `live_frozenlake_tetra.py` | FrozenLake discovery loop demo |
| `test_openclaw_bridge.py` | Bridge integration tests (5 tests, all pass) |

---

## 🚀 Quick Resume

### 1. Start OpenClaw Gateway (Manual)
```powershell
# In a separate terminal:
openclaw gateway start

# Verify:
openclaw gateway health
```

**Note**: Gateway does NOT auto-start on boot. Must run manually before using bridge.

### 2. Resume Tetris Training

**What Happened**: Training crashed on Level 4 (feature dimension mismatch: 20 vs 18)
- L2: 8.50 lines/episode ✅
- L3: 14.60 lines/episode ✅  
- L4: Crashed early (board size changed, features changed)

**To Resume**:
```powershell
cd c:\Users\avata\aicompete\throng3

# Fix: Train each level separately to avoid dimension issues
python train_tetris_curriculum.py --tetra --start-level 4 --max-level 4 --output tetris_L4.json

# Then continue:
python train_tetris_curriculum.py --tetra --start-level 5 --max-level 7 --output tetris_L5_7.json
```

### 3. Check Previous Results

```powershell
# L2-3 successful run:
cat tetris_levels_2_3.json

# Dellacherie baseline:
cat dellacherie_results.txt

# Tetra's observations:
cat C:\Users\avata\.openclaw\workspace\memory\2026-02-15.md

# Audit reports:
ls eval_audits/
```

---

## ⚠️ Known Issues

### 1. Feature Dimension Mismatch
**Problem**: Different Tetris levels have different board sizes → different feature vectors
- L1-3: 6-wide board = 18 features (6 col heights + 12 Dellacherie)
- L4+: 8-wide board = 20 features (8 col heights + 12 Dellacherie)

**Solution**: Either:
- Train each level separately with new agent
- Implement feature padding/projection layer
- Use fixed-size features independent of board width

### 2. Audit Anomalies Detected
The eval auditor found real issues:
- `LINE_MISMATCH`: Reported lines ≠ verified lines
- `REWARD_MISMATCH`: Reported reward ≠ actual reward  
- `SINGLE_ACTION_EXPLOIT`: Agent used only one action

**This validates the auditor is working!** Means there's either:
- Bug in reward computation
- Non-deterministic environment behavior
- Agent exploiting reward function

### 3. Gateway Requires Manual Start
**Not a bug, by design**: OpenClaw gateway is a separate service.

**Auto-start (optional)**:
1. Create `start_gateway.ps1`:
   ```powershell
   openclaw gateway start
   Start-Sleep 3
   openclaw gateway health
   ```
2. Add to Windows startup or create a taskbar shortcut

---

## 📊 Current Metrics

### Tetris Performance (L2-3)
- **L2** (O+I blocks): 8.50 lines/ep (max 50)
- **L3** (O+I+T blocks): 14.60 lines/ep (max 58)
- **Dellacherie L7**: 100.94 lines/ep (baseline)

Gap: 6.8× lower than Dellacherie (expected — we're learning from scratch with fewer episodes)

### Audit Findings (L2-4 partial)
- Episodes checked: ~15 (every 10th)
- Anomalies found: Multiple LINE_MISMATCH and REWARD_MISMATCH
- Suspicious patterns: Single-action exploits detected

---

## 🔄 Next Conversation Checklist

When you return:

1. **Start Gateway**:
   ```powershell
   openclaw gateway start
   openclaw gateway health  # Should see "healthy"
   ```

2. **Tell me where you want to continue**:
   - Fix dimension issue and resume Tetris L4-7?
   - Investigate audit anomalies (reward hacking)?
   - Move to policy composition (Phase 5)?
   - Try different game (GridWorld, MountainCar)?

3. **Reference these files**:
   - This file: `README.md`
   - Task tracker: `.gemini/antigravity/brain/<conversation-id>/task.md`
   - Walkthrough: `.gemini/antigravity/brain/<conversation-id>/walkthrough.md`

---

## 🗂️ Directory Structure

```
throng3/
├── throng4/
│   ├── environments/
│   │   ├── tetris_adapter.py       # Tetris env wrapper
│   │   ├── tetris_curriculum.py    # Levels 1-7
│   │   └── gym_envs.py             # FrozenLake, etc.
│   ├── learning/
│   │   └── portable_agent.py       # NN agent
│   └── llm_policy/
│       ├── openclaw_bridge.py      # Tetra communication ✨
│       ├── eval_auditor.py         # Reward hacking detector ✨
│       ├── env_analyzer.py         # Environment profiling
│       ├── attribution.py          # Stochasticity diagnosis
│       ├── micro_tester.py         # Action probing
│       └── rule_archive.py         # SQLite rule storage
├── train_tetris_curriculum.py      # Main training script
├── live_frozenlake_tetra.py        # Demo discovery loop
├── tetris_levels_2_3.json          # L2-3 results
├── dellacherie_results.txt         # Baseline
└── eval_audits/                    # Independent verification logs
```

---

## 💾 Tetra's Memory

All discoveries stored at:
```
C:\Users\avata\.openclaw\workspace\memory\2026-02-15.md
```

Tetra has been notified of system reboot. On next conversation, can query:
```powershell
openclaw agent --agent main --message "What do you remember about the Tetris curriculum experiment?"
```

---

## 🔧 Quick Fixes Needed

Before next long training run:

1. **Fix dimension issue**:
   - Option A: Train each level with fresh agent
   - Option B: Add feature projection layer to handle variable board sizes
   - Option C: Use board-size-independent features

2. **Investigate audit anomalies**:
   - Replay failed episodes manually
   - Check if environment is truly deterministic
   - Verify reward computation in TetrisAdapter

3. **Optional: Automate gateway**:
   - Create startup script
   - Add health check to training script

---

**Last Updated**: 2026-02-15 16:28 CST
**Conversation ID**: 993f7672-2a99-460d-9866-a61ab2026c4a
