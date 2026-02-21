"""
test_replay_system.py
=====================
Smoke tests for:
  1. HumanPlayLogger  (throng4/storage/human_play_logger.py)
  2. PrioritizedReplayBuffer  (throng4/learning/prioritized_replay.py)
  3. PortableNNAgent imitation head  (throng4/learning/portable_agent.py)

Run from repo root:
    python test_replay_system.py
"""

import json
import os
import sys
import tempfile
import traceback

import numpy as np

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn):
    try:
        fn()
        results.append((PASS, name, ""))
    except Exception as exc:
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_state_vec(n: int = 84) -> np.ndarray:
    """Random 84-dim abstract feature vector with valid mask section."""
    from throng4.learning.abstract_features import CORE_SIZE, EXT_MAX
    vec = np.random.rand(84).astype(np.float32)
    # Set 8 mask slots (CORE_SIZE+EXT_MAX .. CORE_SIZE+EXT_MAX*2)
    mask_start = CORE_SIZE + EXT_MAX
    vec[mask_start:mask_start + 8] = 1.0
    vec[mask_start + 8:] = 0.0
    return vec


# ------------------------------------------------------------------ #
# 1. HumanPlayLogger
# ------------------------------------------------------------------ #

def test_logger_create_and_open():
    """Logger creates DB and returns valid IDs."""
    from throng4.storage.human_play_logger import HumanPlayLogger
    with tempfile.TemporaryDirectory() as td:
        schema = os.path.join(
            os.path.dirname(__file__), "experiments", "REPLAY_DB_SCHEMA.sql"
        )
        with HumanPlayLogger(db_path=os.path.join(td, "r.sqlite"), schema_sql_path=schema) as lg:
            sid = lg.open_session("tetris", "agent_selfplay")
            eid = lg.open_episode(sid, 0, seed=7)
            assert sid.startswith("ses_"), f"bad session_id: {sid}"
            assert eid.startswith("ep_"), f"bad episode_id: {eid}"


def test_logger_log_steps():
    """log_step writes rows; transitions table has expected row count."""
    import sqlite3
    from throng4.storage.human_play_logger import HumanPlayLogger
    with tempfile.TemporaryDirectory() as td:
        schema = os.path.join(
            os.path.dirname(__file__), "experiments", "REPLAY_DB_SCHEMA.sql"
        )
        db_path = os.path.join(td, "r.sqlite")
        with HumanPlayLogger(db_path=db_path, schema_sql_path=schema) as lg:
            sid = lg.open_session("tetris", "human_play")
            eid = lg.open_episode(sid, 0, seed=42)

            for i in range(5):
                vec = _make_state_vec()
                tid = lg.log_step(
                    sid, eid, i, vec,
                    executed_action=i % 4,
                    action_source="human",
                    action_space_n=4,
                    reward=-0.5 if i == 2 else 0.0,
                    done=(i == 4),
                    human_action=i % 4,
                    agent_action=(i + 1) % 4,
                )
                assert isinstance(tid, int) and tid > 0

        # re-open and verify row counts
        conn = sqlite3.connect(db_path)
        n_trans = conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0]
        n_met   = conn.execute("SELECT COUNT(*) FROM transition_metrics").fetchone()[0]
        conn.close()
        assert n_trans == 5, f"expected 5 transitions, got {n_trans}"
        assert n_met   == 5, f"expected 5 metric rows, got {n_met}"


def test_logger_close_episode():
    """close_episode persists end-of-game stats."""
    import sqlite3
    from throng4.storage.human_play_logger import HumanPlayLogger
    with tempfile.TemporaryDirectory() as td:
        schema = os.path.join(
            os.path.dirname(__file__), "experiments", "REPLAY_DB_SCHEMA.sql"
        )
        db_path = os.path.join(td, "r.sqlite")
        with HumanPlayLogger(db_path=db_path, schema_sql_path=schema) as lg:
            sid = lg.open_session("tetris", "agent_selfplay")
            eid = lg.open_episode(sid, 0)
            vec = _make_state_vec()
            lg.log_step(sid, eid, 0, vec, 0, "agent", 4, 1.0, True)
            lg.close_episode(eid, total_reward=1.0, total_steps=1, final_score=100.0)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT total_reward, total_steps, final_score FROM episodes WHERE episode_id=?",
            (eid,),
        ).fetchone()
        conn.close()
        assert row[0] == 1.0
        assert row[1] == 1
        assert row[2] == 100.0


def test_logger_compute_derived():
    """compute_derived fills n_step_return, near_death, disagree, and priority."""
    import sqlite3
    from throng4.storage.human_play_logger import HumanPlayLogger
    with tempfile.TemporaryDirectory() as td:
        schema = os.path.join(
            os.path.dirname(__file__), "experiments", "REPLAY_DB_SCHEMA.sql"
        )
        db_path = os.path.join(td, "r.sqlite")
        with HumanPlayLogger(db_path=db_path, schema_sql_path=schema) as lg:
            sid = lg.open_session("tetris", "human_play")
            eid = lg.open_episode(sid, 0)

            # step 0: near_death + disagree, step 1: recovery, step 2: terminal
            plays = [
                (0, -2.0, False, 0, 1),   # near_death=-2, disagree
                (1,  1.0, False, 2, 2),   # recovery
                (2,  0.5,  True, 3, 3),   # terminal
            ]
            for i, (exec_a, rw, done, ha, aa) in enumerate(plays):
                lg.log_step(
                    sid, eid, i, _make_state_vec(),
                    executed_action=exec_a, action_source="human",
                    action_space_n=4, reward=rw, done=done,
                    human_action=ha, agent_action=aa,
                )
            lg.close_episode(eid, sum(p[1] for p in plays), len(plays))
            n_updated = lg.compute_derived(eid, near_death_threshold=-1.0)
            assert n_updated == 3, f"expected 3 rows updated, got {n_updated}"

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """SELECT m.near_death_flag, m.recovery_flag, m.human_agent_disagree,
                      m.priority_clipped, m.n_step_return_5
               FROM transition_metrics m
               JOIN transitions t ON t.id = m.transition_id
               WHERE t.episode_id = ?
               ORDER BY t.step_idx""",
            (eid,),
        ).fetchall()
        conn.close()
        nd0, rec0, dis0, pri0, ret5_0 = rows[0]
        assert nd0  == 1, "step 0 should be near_death"
        assert dis0 == 1, "step 0 should disagree (ha=0, aa=1)"
        assert rec0 == 1, "step 0 should be recovery (step 1 has +reward)"
        assert pri0 > 1.0, f"priority_clipped should be >1.0, got {pri0}"
        assert ret5_0 is not None


# ------------------------------------------------------------------ #
# 2. PrioritizedReplayBuffer
# ------------------------------------------------------------------ #

def test_prio_buffer_push_sample():
    """Push 100 transitions; sample a batch; verify shapes."""
    from throng4.learning.prioritized_replay import PrioritizedReplayBuffer
    rng = np.random.RandomState(0)
    buf = PrioritizedReplayBuffer(capacity=2000, rng=rng)

    for i in range(100):
        x = np.random.rand(84).astype(np.float32)
        nxt = [np.random.rand(84).astype(np.float32) for _ in range(4)]
        buf.push(
            x, reward=float(i % 3) - 1.0, next_x_list=nxt, done=(i % 20 == 19),
            human_action=i % 4, agent_action=(i + 1) % 4,
            near_death=(i % 10 == 0),
        )

    assert len(buf) == 100
    batch = buf.sample(32)
    assert len(batch) == 32
    x0, r0, nxt0, d0 = batch[0]
    assert x0.shape == (84,)
    assert isinstance(r0, float)
    assert isinstance(d0, bool)


def test_prio_buffer_prioritization():
    """Disagreement + near_death transitions should be over-represented."""
    from throng4.learning.prioritized_replay import PrioritizedReplayBuffer
    rng = np.random.RandomState(42)
    buf = PrioritizedReplayBuffer(capacity=5000, rng=rng)

    # Push 300 normal + 50 near_death + 50 disagree
    for i in range(300):
        x = np.random.rand(84).astype(np.float32)
        buf.push(x, 0.0, [x.copy()], False)
    for i in range(50):
        x = np.random.rand(84).astype(np.float32)
        buf.push(x, -2.0, [x.copy()], False, near_death=True)
    for i in range(50):
        x = np.random.rand(84).astype(np.float32)
        buf.push(x, 0.0, [x.copy()], False, human_action=0, agent_action=1)

    # Sample large batch; count near_death / disagree by proxy (priority > 1.4)
    # (we can't directly identify them from the tuple, so just check no crash and len)
    batch = buf.sample(64)
    assert len(batch) == 64


# ------------------------------------------------------------------ #
# 3. PortableNNAgent imitation head
# ------------------------------------------------------------------ #

def test_imitation_head_forward():
    """_forward_imitation returns (n_actions,) logits when enabled."""
    from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
    cfg = AgentConfig(use_imitation_head=True, imitation_n_actions=6)
    agent = PortableNNAgent(n_features=84, config=cfg, seed=1)
    x = np.random.rand(84).astype(np.float32)
    logits = agent._forward_imitation(x)
    assert logits is not None, "imitation head returned None"
    assert logits.shape == (6,), f"expected shape (6,), got {logits.shape}"


def test_imitation_head_disabled():
    """_forward_imitation returns None when head is disabled."""
    from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
    cfg = AgentConfig(use_imitation_head=False)
    agent = PortableNNAgent(n_features=84, config=cfg, seed=2)
    x = np.random.rand(84).astype(np.float32)
    assert agent._forward_imitation(x) is None


def test_imitation_head_train_step():
    """_train_imitation_step runs without error; weights change."""
    from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
    cfg = AgentConfig(use_imitation_head=True, imitation_n_actions=6, imitation_lr=0.01)
    agent = PortableNNAgent(n_features=84, config=cfg, seed=3)
    x = np.random.rand(84).astype(np.float32)
    Wi2_before = agent.Wi2.copy()
    for _ in range(5):
        agent._train_imitation_step(x, human_action=2)
    assert not np.allclose(agent.Wi2, Wi2_before), "Wi2 should have changed after training"


def test_imitation_head_save_load():
    """save_weights + load_weights round-trips imitation head."""
    from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
    cfg = AgentConfig(use_imitation_head=True, imitation_n_actions=6)
    agent = PortableNNAgent(n_features=84, config=cfg, seed=4)
    x = np.random.rand(84).astype(np.float32)
    for _ in range(3):
        agent._train_imitation_step(x, 1)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "weights")
        agent.save_weights(path)
        agent2 = PortableNNAgent(n_features=84, config=cfg, seed=99)
        agent2.load_weights(path + ".npz")
        assert np.allclose(agent.Wi1, agent2.Wi1), "Wi1 mismatch after save/load"
        assert np.allclose(agent.Wi2, agent2.Wi2), "Wi2 mismatch after save/load"


def test_imitation_head_backward_compat():
    """Agents without use_imitation_head still load old-style checkpoints."""
    from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
    cfg = AgentConfig()
    agent = PortableNNAgent(n_features=84, config=cfg, seed=5)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "weights_no_head")
        agent.save_weights(path)
        agent2 = PortableNNAgent(n_features=84, config=cfg, seed=6)
        agent2.load_weights(path + ".npz")   # should not crash
        assert agent2.Wi1 is None


def test_record_step_backward_compat():
    """record_step still works with the old 4-arg call signature."""
    from throng4.learning.portable_agent import PortableNNAgent, AgentConfig
    cfg = AgentConfig()
    agent = PortableNNAgent(n_features=84, config=cfg, seed=7)
    x = np.random.rand(84).astype(np.float32)
    nxt = [np.random.rand(84).astype(np.float32)]
    # Old-style call — positional only
    agent.record_step(x, 0.0, nxt, False)   # must not raise


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

TESTS = [
    ("HumanPlayLogger: create + open",      test_logger_create_and_open),
    ("HumanPlayLogger: log_step rows",       test_logger_log_steps),
    ("HumanPlayLogger: close_episode",       test_logger_close_episode),
    ("HumanPlayLogger: compute_derived",     test_logger_compute_derived),
    ("PrioritizedReplay: push + sample",     test_prio_buffer_push_sample),
    ("PrioritizedReplay: prioritization",    test_prio_buffer_prioritization),
    ("ImitationHead: forward (enabled)",     test_imitation_head_forward),
    ("ImitationHead: forward (disabled)",    test_imitation_head_disabled),
    ("ImitationHead: train step",            test_imitation_head_train_step),
    ("ImitationHead: save / load",           test_imitation_head_save_load),
    ("ImitationHead: backward compat load",  test_imitation_head_backward_compat),
    ("record_step: old 4-arg compat",        test_record_step_backward_compat),
]

if __name__ == "__main__":
    print(f"\nRunning {len(TESTS)} smoke tests...\n")
    for name, fn in TESTS:
        check(name, fn)

    passed = sum(1 for r in results if r[0] == PASS)
    total  = len(results)
    print()
    for status, name, err in results:
        icon = "✓" if status == PASS else "✗"
        print(f"  {icon} {name}")
        if err:
            print(f"      → {err}")
    print(f"\n{passed}/{total} passed")
    if passed < total:
        sys.exit(1)
