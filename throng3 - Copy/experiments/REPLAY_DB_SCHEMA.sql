-- ============================================================
-- Replay DB Schema — throng4 Human Play + Prioritized Replay
-- Schema by Tetra (2026-02-20). Verbatim + throng4 context.
--
-- Adaptations for throng4:
--   obs_ref → use "abstract://..." prefix for 84-dim abstract
--             feature vectors stored inline in transition_metrics
--   logits_ref → unused until imitation head is added
--   rom_id → optional; not used for Tetris
-- ============================================================

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=MEMORY;
PRAGMA foreign_keys=ON;

-- =========================
-- 1) Sessions / Episodes
-- =========================

CREATE TABLE IF NOT EXISTS sessions (
  session_id         TEXT PRIMARY KEY,
  created_at_ms      INTEGER NOT NULL,
  source             TEXT NOT NULL,              -- 'human_play' | 'agent_selfplay' | 'mixed'
  env_name           TEXT NOT NULL,              -- 'tetris' | 'ALE/Breakout-v5' etc.
  env_version        TEXT,
  rom_id             TEXT,
  config_json        TEXT                        -- serialized run config
);

CREATE TABLE IF NOT EXISTS episodes (
  episode_id         TEXT PRIMARY KEY,
  session_id         TEXT NOT NULL,
  episode_index      INTEGER NOT NULL,
  started_at_ms      INTEGER NOT NULL,
  ended_at_ms        INTEGER,
  seed               INTEGER,
  total_reward       REAL,
  total_steps        INTEGER,
  final_score        REAL,
  final_lives        INTEGER,
  terminated         INTEGER NOT NULL DEFAULT 0, -- bool 0/1
  truncated          INTEGER NOT NULL DEFAULT 0, -- bool 0/1
  FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_episodes_session
  ON episodes(session_id, episode_index);

-- =========================
-- 2) Step-level transitions
-- =========================

CREATE TABLE IF NOT EXISTS transitions (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id           TEXT NOT NULL,
  episode_id           TEXT NOT NULL,
  step_idx             INTEGER NOT NULL,
  timestamp_ms         INTEGER NOT NULL,

  -- state refs (store heavy arrays on disk/object store)
  -- throng4: use "abstract://<session>/<ep>/<step>" for abstract feature vecs
  obs_ref              TEXT NOT NULL,
  next_obs_ref         TEXT,
  obs_shape_json       TEXT,                     -- e.g. "[84]" for abstract, "[128]" for RAM
  latent_ref           TEXT,

  -- actions
  human_action         INTEGER,                  -- nullable if no human input
  agent_action         INTEGER,                  -- nullable during pure human mode
  executed_action      INTEGER NOT NULL,
  action_source        TEXT NOT NULL,            -- 'human' | 'agent' | 'blended'
  action_space_n       INTEGER NOT NULL,

  -- reward / termination
  reward               REAL NOT NULL,
  done                 INTEGER NOT NULL,          -- bool 0/1
  truncated            INTEGER NOT NULL,          -- bool 0/1

  -- optional env signals
  score                REAL,
  lives                INTEGER,

  -- lightweight model outputs
  -- throng4: value_pred = PortableNN Q-value of executed_action
  value_pred           REAL,
  policy_entropy       REAL,
  policy_logits_ref    TEXT,                     -- blob ref/path
  imitation_logits_ref TEXT,                     -- blob ref/path

  -- generic tags/notes
  tags_json            TEXT,
  notes                TEXT,

  FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  FOREIGN KEY(episode_id) REFERENCES episodes(episode_id) ON DELETE CASCADE,
  UNIQUE(episode_id, step_idx)                   -- prevents duplicate on crash/restart
);

CREATE INDEX IF NOT EXISTS idx_transitions_episode_step
  ON transitions(episode_id, step_idx);

CREATE INDEX IF NOT EXISTS idx_transitions_session
  ON transitions(session_id);

CREATE INDEX IF NOT EXISTS idx_transitions_action_source
  ON transitions(action_source);

CREATE INDEX IF NOT EXISTS idx_transitions_done
  ON transitions(done);

-- =========================
-- 3) Derived metrics (for replay / prioritization)
-- =========================

CREATE TABLE IF NOT EXISTS transition_metrics (
  transition_id        INTEGER PRIMARY KEY,      -- 1:1 with transitions.id

  n_step_return_5      REAL,
  n_step_return_20     REAL,
  time_to_terminal     INTEGER,
  near_death_flag      INTEGER,                  -- bool
  recovery_flag        INTEGER,                  -- bool
  breakthrough_flag    INTEGER,                  -- bool
  human_agent_disagree INTEGER,                  -- bool: human_action != agent_action
  td_error             REAL,
  novelty_score        REAL,

  -- abstract feature snapshot (throng4 addition — inline, no disk ref needed)
  abstract_vec_json    TEXT,                     -- JSON array of 84 floats
  ext_slots_active     INTEGER,                  -- number of active ext mask slots

  -- prioritization
  priority_raw         REAL,
  priority_clipped     REAL,
  sampling_weight      REAL,                     -- importance-sampling weight
  last_updated_ms      INTEGER,

  FOREIGN KEY(transition_id) REFERENCES transitions(id) ON DELETE CASCADE
);

-- KEY: replay sampler uses this index
CREATE INDEX IF NOT EXISTS idx_metrics_priority
  ON transition_metrics(priority_clipped DESC);

CREATE INDEX IF NOT EXISTS idx_metrics_flags
  ON transition_metrics(near_death_flag, recovery_flag, breakthrough_flag);

CREATE INDEX IF NOT EXISTS idx_metrics_disagree
  ON transition_metrics(human_agent_disagree);

-- =========================
-- 4) Replay views
-- =========================

-- Human-labeled transitions (imitation training)
CREATE VIEW IF NOT EXISTS v_human_transitions AS
SELECT
  t.id, t.session_id, t.episode_id, t.step_idx, t.timestamp_ms,
  t.obs_ref, t.next_obs_ref, t.human_action, t.agent_action, t.executed_action,
  t.reward, t.done, t.score, t.lives,
  m.priority_clipped, m.sampling_weight, m.td_error, m.novelty_score,
  m.abstract_vec_json, m.ext_slots_active
FROM transitions t
LEFT JOIN transition_metrics m ON m.transition_id = t.id
WHERE t.human_action IS NOT NULL;

-- Top-priority replay candidates (prioritized experience replay)
CREATE VIEW IF NOT EXISTS v_priority_replay AS
SELECT
  t.id, t.episode_id, t.step_idx, t.obs_ref, t.next_obs_ref,
  t.executed_action, t.reward, t.done,
  m.priority_clipped, m.sampling_weight,
  m.near_death_flag, m.recovery_flag, m.human_agent_disagree,
  m.abstract_vec_json
FROM transitions t
JOIN transition_metrics m ON m.transition_id = t.id
ORDER BY m.priority_clipped DESC, t.timestamp_ms DESC;
