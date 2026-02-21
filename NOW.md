# NOW.md

## Current Phase
Blind generalization pipeline hardening + first production blind batches (Tetra offline mode).

## Last Known Good Commits
- `0bb02ec` — first blind batch live
- `5d85988` — `validate_blind_hypotheses.py` gate added and verified

## System State (Quick)
- Blind labels active (e.g., `Environment-D`)
- Identity leakage checks passing
- Mask integrity checks passing
- Validator gate active (schema + enums + confidence + blind-vocab lint)
- Rules ingest path: `~/.openclaw/rules/tetris_rules.json`
- Snapshot table: `experiments/experiments.db` → `blind_hypothesis_log`

## Next Exact Command
```bash
python validate_blind_hypotheses.py ~/.openclaw/workspace/memory/hypotheses_<ts>.json tetris
```

## If Validation Passes
```bash
python tools/restart_archive.py snapshot --summary "Blind batch <ts> validated+ingested"
```

## Quick Health Checks
```bash
python -m throng4.config
git status --short
```

## One-Command Helpers
```powershell
powershell -ExecutionPolicy Bypass -File .\tools\status_now.ps1
powershell -ExecutionPolicy Bypass -File .\tools\checkpoint_now.ps1 -Summary "<what changed>"
```

## Last Artifact Locations
- Request: `~/.openclaw/workspace/memory/hyp_request_<ts>.md`
- Response: `~/.openclaw/workspace/memory/hypotheses_<ts>.json`
- Rules: `~/.openclaw/rules/tetris_rules.json`

## Known Blockers
- None currently. (If blocked, write one bullet here before ending session.)

## Session Handoff Rule
Before ending a session, update ONLY these 4 lines:
1. Current Phase
2. Last Known Good Commits
3. Next Exact Command
4. Known Blockers
