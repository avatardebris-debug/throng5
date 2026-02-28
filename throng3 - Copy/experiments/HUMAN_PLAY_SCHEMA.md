# Human Play Logger Schema — v1

> Schema from Tetra (2026-02-20 chat). Saved verbatim + adaptations for throng4.
> **Note:** The example uses `"game": "antigravity"` — that's a confusion artifact.
> "Antigravity" is the AI coding assistant present in the session, not a game.
> Use `"tetris"` or `"ALE/Breakout-v5"` for actual game fields.

---

## Primary Record (JSONL — one line per timestep)

```json
{
  "schema_version": "1.0",
  "session_id":     "hp_2026-02-20_tetris_01",
  "episode_id":     "ep_000123",
  "step_idx":        417,
  "timestamp_ms":    1771653065123,

  "env": {
    "game":  "tetris",
    "mode":  "human_play",
    "seed":  938421
  },

  "state": {
    "obs_ref":   "replay://obs/hp_2026-02-20_tetris_01/ep_000123/0417.npy",
    "obs_shape": [84, 84, 4]
  },

  "action": {
    "human_action":    3,
    "agent_action":    2,
    "executed_action": 3,
    "action_source":   "human",
    "action_space_n":  18
  },

  "signals": {
    "reward":    0.0,
    "done":      false,
    "truncated": false,
    "score":     240,
    "lives":     2
  },

  "model": {
    "policy_logits":    [0.1, -0.8, 0.4, 1.3],
    "imitation_logits": [0.2, -0.5, 0.3, 1.0],
    "value_pred":       0.42,
    "policy_entropy":   1.18
  },

  "meta": {
    "input_latency_ms": 37,
    "notes": "",
    "tags":  ["neutral"]
  }
}
```

---

## Derived / Replay Fields (computed after episode closes)

```json
{
  "session_id":  "hp_2026-02-20_tetris_01",
  "episode_id":  "ep_000123",
  "step_idx":     417,

  "derived": {
    "n_step_return_5":    0.75,
    "n_step_return_20":   1.9,
    "time_to_terminal":   83,
    "near_death_flag":    false,
    "recovery_flag":      true,
    "breakthrough_flag":  false,
    "human_agent_disagree": true,
    "td_error":           0.63,
    "novelty_score":      0.21
  },

  "priority": {
    "priority_raw":     1.47,
    "priority_clipped": 1.47,
    "sampling_weight":  0.81
  }
}
```

---

## Required vs Optional (v1)

**Required**
- `schema_version`, `session_id`, `episode_id`, `step_idx`, `timestamp_ms`
- `env.game`, `state.obs_ref`
- `action.human_action`, `action.executed_action`, `action.action_source`
- `signals.reward`, `signals.done`

**Optional (strongly recommended)**
- `action.agent_action`
- `signals.score`, `signals.lives`
- `model.policy_logits`, `model.value_pred`
- All `derived.*` and `priority.*`

---

## Throng4 Adaptations

### 1. `state` — use abstract feature vector instead of pixel frames

We use 84-dim `to_vector()` output, not pixel frames. Replace `obs_ref` with inline vector:

```json
"state": {
  "abstract_vec":     [0.54, 0.95, 0.76, "..."],
  "obs_source":       "abstract_features",
  "ext_slots_active": 8
}
```

### 2. `model` — PortableNN outputs Q-values, not policy logits

```json
"model": {
  "q_values": {"(0,2)": 0.42, "(1,3)": 0.38},
  "chosen_q": 0.42,
  "value_pred": 0.42,
  "epsilon": 0.15
}
```

### 3. `env.game` — use real game identifiers

- Tetris: `"tetris"`
- Atari: `"ALE/Breakout-v5"`, `"ALE/Pong-v5"`, etc.

---

## Priority Formula (v1, before value head exists)

```python
priority_raw = (
    (1.5 if human_agent_disagree else 1.0) *
    (2.0 if near_death_flag else 1.0) *
    (1.0 + novelty_score)
)
```

## Replay Priority Ordering

1. `human_agent_disagree=True` — richest imitation signal
2. `near_death_flag=True` — survival signal
3. `recovery_flag=True` — rare, high-value
4. `td_error` — standard PER (after value head exists)

---

## Next Steps

- [ ] Ask Tetra for matching SQLite schema (tables + indexes)
- [ ] Implement `HumanPlayLogger` (JSONL writer + SQLite indexer)
- [ ] Implement `PrioritizedReplayBuffer` (replaces uniform `ReplayBuffer`)
- [ ] Add imitation head to `PortableNNAgent` (soft-update, frozen backbone)
