# PROJECT_STATE.md

Purpose: single restart/debug handoff for Throng3/Throng4 work (including Google Antigravity context resets).

Last updated: 2026-02-20
Owner: Mike + Tetra

---

## 1) Canonical Paths (source of truth)

- Repo root: `C:\Users\avata\aicompete\throng3`
- OpenClaw root (default): `~/.openclaw` (override with `OPENCLAW_WORKSPACE_ROOT`)
- Memory requests/responses: `~/.openclaw/workspace/memory`
- Rule library output: `~/.openclaw/rules`

Path config is centralized in:
- `throng4/config.py`

Environment overrides:
- `OPENCLAW_WORKSPACE_ROOT`
- `THRONG_ROOT`

---

## 2) Current Active Architecture

### Active runtime code
- `throng4/` = current main implementation (env adapters, learning loop, LLM policy integration)
- `throng4/llm_policy/offline_generator.py` = hardened offline hypothesis pipeline
- `throng4/llm_policy/openclaw_bridge.py` = OpenClaw bridge
- `throng4/llm_policy/eval_auditor.py` = audit/verification path

### Legacy/reference code
- `throng3/` = original Meta^N core prototype and experiments
- `throng35/` = regional architecture experiments
- `throng2-master - Copy/` and `openclaw-main - Copy/` = reference/vendor snapshots (do not treat as active runtime)

---

## 3) What is implemented and verified (recent)

### Offline batch hypothesis flow (Breakout/Atari)
Implemented and verified end-to-end:
1. Python writes request file: `hyp_request_<ts>.md`
2. Tetra writes JSON atomically to: `hypotheses_<ts>.json.tmp` then renames to `.json`
3. Tetra replies with machine-parseable ack: `ACK: WRITTEN <absolute_path>`
4. Python parses ACK, then validates JSON and required keys
5. If schema invalid, Python sends automatic repair prompt and retries
6. `--inject <file>` available as fallback for chat-only replies

Required hypothesis fields:
- `id, description, object, feature, direction, trigger, confidence`

Prompt contract location:
- `experiments/TETRA_PROMPT.md` (Atari Offline Batch Mode v2 section)

### DQN Experience Replay — PortableNNAgent (2026-02-20)

`PortableNNAgent` is now fully off-policy DQN. Key API change:

```python
# OLD (n-step on-policy):
agent.record_step(features, reward)
agent.end_episode(final_score)           # training happened here

# NEW (off-policy DQN):
features = adapter.make_features(action) # MUST be called BEFORE adapter.step()
obs, reward, done, info = adapter.step(action)
next_features = [adapter.make_features(a) for a in adapter.get_valid_actions()]
agent.record_step(features, reward, next_features, done)  # training every N steps
agent.end_episode(final_score)           # ε-decay + housekeeping only
```

Critical bug fixed: `make_features` in `TetrisAdapter` simulates placement on the *current* board. Calling it *after* `step()` gave it the post-move board — the features were always one step stale. Now called pre-step everywhere.



## 4) Directory Organization (practical working model)

Use this mental model when debugging:

- `throng4/` → production-ish code paths
- `experiments/` → prompts, DB, one-off experimental assets
- `tests/` + top-level `test_*.py` → validation and diagnostics
- `atari_logs/` → trajectory logs
- `eval_audits/` → audit outputs
- `results/` → curated outputs
- top-level `*.txt/json` scatter → historical artifacts; keep but avoid for active logic decisions

### Suggested cleanup policy (non-breaking)
- Keep active files where they are for now (stability first)
- Move stale root artifacts into `archive/YYYY-MM-DD/` in batches
- Keep only these status files at repo root:
  - `README.md`
  - `PROJECT_STATE.md` (this file)
  - `STATUS.md` (optional high-level)

---

## 5) Fast Restart Checklist (after context loss)

1. Read this file (`PROJECT_STATE.md`)
2. Verify paths:
   - `python -m throng4.config`
3. Verify bridge/gateway availability
4. Check git working state:
   - `git status --short`
5. If debugging offline hypotheses:
   - confirm `experiments/TETRA_PROMPT.md` has Offline Batch v2 section
   - run generator on a known log
6. Inspect latest outputs:
   - `~/.openclaw/workspace/memory/hyp_request_*.md`
   - `~/.openclaw/workspace/memory/hypotheses_*.json`
   - `~/.openclaw/rules/*_rules.json`

---

## 6) Known high-value files

- Prompt contract: `experiments/TETRA_PROMPT.md`
- Path config: `throng4/config.py`
- Offline pipeline: `throng4/llm_policy/offline_generator.py`
- OpenClaw bridge: `throng4/llm_policy/openclaw_bridge.py`
- Eval auditor: `throng4/llm_policy/eval_auditor.py`
- Legacy context docs: `README.md`, `STATUS.md`, `THRONG3_COMPLETE.md`

---

## 7) Update Protocol (keep this file useful)

When major changes land, update ONLY these sections:
- Section 1 (paths) if roots/overrides changed
- Section 3 (implemented+verified) with date + bullet summary
- Section 5 (restart checklist) if workflow changed
- Section 6 (high-value files) if entry points moved

Template entry:
- Date:
- Change:
- Why:
- Verification:
- Impact on restart/debug:

---

## 8) SQLite restart archive (optional but useful)

Created:
- `state/restart_archive.sqlite`
- builder: `tools/restart_archive.py`

Use:
- `python tools/restart_archive.py init`
- `python tools/restart_archive.py snapshot --summary "<what changed>"`

What it stores:
- snapshots (`git commit`, branch, timestamp, summary)
- high-value context files (restart reading order)
- key decisions/checkpoints
- quick commands for recovery/debug

## 9) Current reality check

Repo is powerful but noisy (11k+ files + historical copies). The right strategy is:
- stabilize runtime contracts first (done)
- centralize path truth (done)
- use this file as the single restart index
- archive clutter incrementally (later, safe)
