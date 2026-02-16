---
description: Resume Tetris curriculum training after reboot
---

# Resume Tetris Training Workflow

## Prerequisites

// turbo
1. Start OpenClaw gateway:
```powershell
openclaw gateway start
```

// turbo
2. Verify gateway health:
```powershell
openclaw gateway health
```

Expected output: Status "healthy"

---

## Resume Training

### Option A: Continue from Level 4 (Recommended)

Previous run completed L2-3 successfully but crashed on L4 due to feature dimension mismatch.

// turbo
Train Level 4 separately (fresh agent for new board size):
```powershell
cd c:\Users\avata\aicompete\throng3
python train_tetris_curriculum.py --tetra --start-level 4 --max-level 4 --output tetris_L4.json
```

// turbo
Then train L5-7:
```powershell
python train_tetris_curriculum.py --tetra --start-level 5 --max-level 7 --output tetris_L5_7.json
```

### Option B: Full Re-run with Fixed Agent

If we fix the dimension issue first, can run full curriculum:

```powershell
# TODO: Implement feature padding/projection in PortableNNAgent first
python train_tetris_curriculum.py --tetra --start-level 2 --max-level 7 --output tetris_full_fixed.json
```

---

## Review Previous Results

// turbo-all
```powershell
# L2-3 results:
cat tetris_levels_2_3.json

# Dellacherie baseline:
cat dellacherie_results.txt

# Audit reports (check for reward hacking):
ls eval_audits/
cat eval_audits\audit_*.json | ConvertFrom-Json | Format-List

# Tetra's memory:
cat C:\Users\avata\.openclaw\workspace\memory\2026-02-15.md
```

---

## Expected Metrics

Based on previous run (L2-3):
- **L2**: ~8-10 lines/episode
- **L3**: ~14-16 lines/episode
- **L4+**: Unknown (board size increases, more pieces)

Dellacherie baseline (L7): **100.94 lines/episode**

---

## Troubleshooting

### Gateway Not Starting
```powershell
# Check if already running:
openclaw gateway status

# Force restart:
openclaw gateway stop
openclaw gateway start
```

### Dimension Mismatch Error
If you see: `ValueError: size 20 is different from 18`

**Cause**: Board width changes between levels → feature size changes

**Fix**: Train each level group separately:
- L1-3: 6-wide board
- L4: 8-wide board  
- L5-7: 10-wide board

### Audit Anomalies
If auditor reports `LINE_MISMATCH` or `REWARD_MISMATCH`:

1. Check audit JSON for details: `eval_audits/audit_*.json`
2. Replay suspicious episodes manually
3. Verify environment is deterministic
4. May indicate reward hacking or environment bugs

---

**Last Updated**: 2026-02-15
**Dependencies**: OpenClaw gateway, Python 3.x, throng4 modules
