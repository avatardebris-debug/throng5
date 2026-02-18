"""
ExperimentDB — SQLite-backed structured persistence for the dual-loop architecture.

Stores hypotheses, episodes, events, promotions, and policy packs. Nothing is ever
deleted. Follows the same pattern as RuleArchive (SQLite + row_factory + context manager).

Tables:
- hypotheses: Hypothesis lifecycle (candidate → testing → active → retired)
- microhypotheses: Concrete testable micro-actions derived from hypotheses
- episodes: Per-episode training results
- events: Novelty/failure/edge-case flags
- promotions: Hypothesis status change audit trail
- policy_packs: Frozen versioned sets of active hypotheses
- concepts: Discovered meta-learning concepts with transferability
"""

import sqlite3
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path


class ExperimentDB:
    """
    SQLite-backed experiment database for the dual-loop architecture.

    Usage:
        with ExperimentDB("experiments/experiments.db") as db:
            db.log_episode(game="tetris", level=2, score=88, ...)
            db.upsert_hypothesis(name="maximize_lines", ...)
    """

    def __init__(self, db_path: str = "experiments/experiments.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database schema if it doesn't exist."""
        c = self.conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                scope TEXT DEFAULT 'game',
                game TEXT,
                level INTEGER,
                description TEXT,
                confidence REAL DEFAULT 0.5,
                win_rate REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'candidate',
                parent_id TEXT,
                specialization_score REAL DEFAULT 0.0,
                metadata TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS microhypotheses (
                id TEXT PRIMARY KEY,
                parent_hypothesis_id TEXT NOT NULL,
                test_protocol TEXT,
                expected_signal TEXT,
                actual_signal TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL,
                tested_at REAL,
                FOREIGN KEY (parent_hypothesis_id) REFERENCES hypotheses(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                game TEXT NOT NULL,
                level INTEGER,
                episode_num INTEGER,
                score REAL,
                lines_cleared INTEGER,
                pieces_placed INTEGER,
                max_height INTEGER,
                holes INTEGER,
                bumpiness REAL,
                outcome_tags TEXT,
                hypothesis_performance TEXT,
                policy_pack_version INTEGER,
                session_id TEXT,
                timestamp REAL NOT NULL
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                episode_id TEXT,
                event_type TEXT NOT NULL,
                data TEXT,
                timestamp REAL NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS promotions (
                id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                from_status TEXT NOT NULL,
                to_status TEXT NOT NULL,
                evidence_summary TEXT,
                policy_pack_version INTEGER,
                timestamp REAL NOT NULL,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS policy_packs (
                version INTEGER PRIMARY KEY,
                active_hypothesis_ids TEXT NOT NULL,
                created_at REAL NOT NULL,
                notes TEXT
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                discovered_at REAL NOT NULL,
                discovered_in TEXT,
                transferability REAL DEFAULT 0.0,
                validated_on TEXT,
                evidence TEXT,
                status TEXT DEFAULT 'active',
                metadata TEXT
            )
        """)

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_game ON episodes(game)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_level ON episodes(game, level)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_hypotheses_game ON hypotheses(game)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_concepts_name ON concepts(name)")

        self.conn.commit()

    # ────────────────────── Episode logging ──────────────────────

    def log_episode(self, game: str, level: int = 0, episode_num: int = 0,
                    score: float = 0, lines_cleared: int = 0, pieces_placed: int = 0,
                    max_height: int = 0, holes: int = 0, bumpiness: float = 0,
                    outcome_tags: Optional[Dict] = None,
                    hypothesis_performance: Optional[Dict] = None,
                    policy_pack_version: int = 0,
                    session_id: Optional[str] = None,
                    timestamp: Optional[float] = None) -> str:
        """Log a single episode result. Returns the episode ID."""
        eid = str(uuid.uuid4())[:8]
        ts = timestamp or time.time()
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO episodes (id, game, level, episode_num, score, lines_cleared,
                pieces_placed, max_height, holes, bumpiness, outcome_tags,
                hypothesis_performance, policy_pack_version, session_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (eid, game, level, episode_num, score, lines_cleared, pieces_placed,
              max_height, holes, bumpiness,
              json.dumps(outcome_tags) if outcome_tags else None,
              json.dumps(hypothesis_performance) if hypothesis_performance else None,
              policy_pack_version, session_id, ts))
        self.conn.commit()
        return eid

    # ────────────────────── Hypothesis management ──────────────────────

    def upsert_hypothesis(self, name: str, game: str = "", level: int = 0,
                          description: str = "", confidence: float = 0.5,
                          win_rate: float = 0.0, evidence_count: int = 0,
                          status: str = "candidate", parent_id: Optional[str] = None,
                          specialization_score: float = 0.0,
                          metadata: Optional[Dict] = None,
                          hypothesis_id: Optional[str] = None) -> str:
        """Insert or update a hypothesis. Returns the hypothesis ID."""
        hid = hypothesis_id or str(uuid.uuid4())[:8]
        now = time.time()
        c = self.conn.cursor()

        # Check if exists by name+game+level
        c.execute("SELECT id, created_at FROM hypotheses WHERE name = ? AND game = ? AND level = ?",
                  (name, game, level))
        existing = c.fetchone()

        if existing:
            hid = existing['id']
            c.execute("""
                UPDATE hypotheses SET confidence = ?, win_rate = ?, evidence_count = ?,
                    status = ?, specialization_score = ?, metadata = ?, updated_at = ?,
                    description = ?
                WHERE id = ?
            """, (confidence, win_rate, evidence_count, status, specialization_score,
                  json.dumps(metadata) if metadata else None, now, description, hid))
        else:
            c.execute("""
                INSERT INTO hypotheses (id, name, game, level, description, confidence,
                    win_rate, evidence_count, status, parent_id, specialization_score,
                    metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hid, name, game, level, description, confidence, win_rate,
                  evidence_count, status, parent_id, specialization_score,
                  json.dumps(metadata) if metadata else None, now, now))

        self.conn.commit()
        return hid

    # ────────────────────── Events ──────────────────────

    def log_event(self, event_type: str, data: Optional[Dict] = None,
                  episode_id: Optional[str] = None,
                  timestamp: Optional[float] = None) -> str:
        """Log an event (novelty, failure, edge_case, concept_discovered, transfer_result)."""
        eid = str(uuid.uuid4())[:8]
        ts = timestamp or time.time()
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO events (id, episode_id, event_type, data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (eid, episode_id, event_type, json.dumps(data) if data else None, ts))
        self.conn.commit()
        return eid

    # ────────────────────── Concepts ──────────────────────

    def upsert_concept(self, name: str, description: str = "",
                       discovered_in: str = "", transferability: float = 0.0,
                       validated_on: Optional[List[str]] = None,
                       evidence: str = "", status: str = "active",
                       metadata: Optional[Dict] = None,
                       timestamp: Optional[float] = None) -> str:
        """Insert or update a discovered concept."""
        ts = timestamp or time.time()
        c = self.conn.cursor()

        c.execute("SELECT id FROM concepts WHERE name = ?", (name,))
        existing = c.fetchone()

        if existing:
            cid = existing['id']
            c.execute("""
                UPDATE concepts SET description = ?, transferability = ?,
                    validated_on = ?, evidence = ?, status = ?, metadata = ?
                WHERE id = ?
            """, (description, transferability,
                  json.dumps(validated_on) if validated_on else None,
                  evidence, status,
                  json.dumps(metadata) if metadata else None, cid))
        else:
            cid = str(uuid.uuid4())[:8]
            c.execute("""
                INSERT INTO concepts (id, name, description, discovered_at, discovered_in,
                    transferability, validated_on, evidence, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (cid, name, description, ts, discovered_in, transferability,
                  json.dumps(validated_on) if validated_on else None,
                  evidence, status,
                  json.dumps(metadata) if metadata else None))

        self.conn.commit()
        return cid

    # ────────────────────── Promotions ──────────────────────

    def promote(self, hypothesis_id: str, from_status: str, to_status: str,
                evidence_summary: Optional[Dict] = None,
                policy_pack_version: int = 0) -> str:
        """Record a hypothesis promotion/demotion."""
        pid = str(uuid.uuid4())[:8]
        now = time.time()
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO promotions (id, hypothesis_id, from_status, to_status,
                evidence_summary, policy_pack_version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (pid, hypothesis_id, from_status, to_status,
              json.dumps(evidence_summary) if evidence_summary else None,
              policy_pack_version, now))

        # Also update the hypothesis status
        c.execute("UPDATE hypotheses SET status = ?, updated_at = ? WHERE id = ?",
                  (to_status, now, hypothesis_id))
        self.conn.commit()
        return pid

    # ────────────────────── Policy Packs ──────────────────────

    def create_policy_pack(self, active_hypothesis_ids: List[str],
                           notes: str = "") -> int:
        """Create a new policy pack. Returns the version number."""
        now = time.time()
        c = self.conn.cursor()
        c.execute("SELECT MAX(version) as max_v FROM policy_packs")
        row = c.fetchone()
        version = (row['max_v'] or 0) + 1

        c.execute("""
            INSERT INTO policy_packs (version, active_hypothesis_ids, created_at, notes)
            VALUES (?, ?, ?, ?)
        """, (version, json.dumps(active_hypothesis_ids), now, notes))
        self.conn.commit()
        return version

    # ────────────────────── Queries ──────────────────────

    def get_episodes(self, game: Optional[str] = None, level: Optional[int] = None,
                     session_id: Optional[str] = None,
                     since: Optional[float] = None,
                     limit: int = 1000) -> List[Dict]:
        """Query episodes with optional filters."""
        c = self.conn.cursor()
        query = "SELECT * FROM episodes WHERE 1=1"
        params = []

        if game:
            query += " AND game = ?"
            params.append(game)
        if level is not None:
            query += " AND level = ?"
            params.append(level)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        c.execute(query, params)
        return [dict(row) for row in c.fetchall()]

    def get_hypotheses(self, game: Optional[str] = None,
                       status: Optional[str] = None) -> List[Dict]:
        """Query hypotheses with optional filters."""
        c = self.conn.cursor()
        query = "SELECT * FROM hypotheses WHERE 1=1"
        params = []

        if game:
            query += " AND game = ?"
            params.append(game)
        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY updated_at DESC"
        c.execute(query, params)
        return [dict(row) for row in c.fetchall()]

    def get_top_hypotheses(self, limit: int = 5,
                           min_win_rate: float = 0.2,
                           min_evidence: int = 10,
                           exclude_game: Optional[str] = None) -> List[Dict]:
        """
        Fetch the best-performing hypotheses across all games.

        Used by the bootstrap system to seed a new game's hypothesis pool
        with proven strategies from prior experience.

        Args:
            limit:          Max hypotheses to return.
            min_win_rate:   Only include hypotheses above this win rate.
            min_evidence:   Only include hypotheses with enough evaluations.
            exclude_game:   Exclude hypotheses from this game (avoids
                            re-importing the same game's own hypotheses).

        Returns:
            List of hypothesis dicts ordered by win_rate DESC.
        """
        c = self.conn.cursor()
        query = """
            SELECT * FROM hypotheses
            WHERE win_rate >= ?
              AND evidence_count >= ?
              AND status != 'retired'
        """
        params: List[Any] = [min_win_rate, min_evidence]

        if exclude_game:
            query += " AND (game != ? OR game IS NULL OR game = '')"
            params.append(exclude_game)

        query += " ORDER BY win_rate DESC, confidence DESC LIMIT ?"
        params.append(limit)

        c.execute(query, params)
        return [dict(row) for row in c.fetchall()]

    def get_concepts(self, status: Optional[str] = None) -> List[Dict]:
        """Query all concepts."""
        c = self.conn.cursor()
        if status:
            c.execute("SELECT * FROM concepts WHERE status = ?", (status,))
        else:
            c.execute("SELECT * FROM concepts")
        return [dict(row) for row in c.fetchall()]

    def get_events(self, event_type: Optional[str] = None,
                   limit: int = 100) -> List[Dict]:
        """Query events."""
        c = self.conn.cursor()
        if event_type:
            c.execute("SELECT * FROM events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?",
                      (event_type, limit))
        else:
            c.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        return [dict(row) for row in c.fetchall()]

    def summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across the entire DB."""
        c = self.conn.cursor()
        stats = {}

        c.execute("SELECT COUNT(*) as n FROM episodes")
        stats['total_episodes'] = c.fetchone()['n']

        c.execute("SELECT COUNT(*) as n FROM hypotheses")
        stats['total_hypotheses'] = c.fetchone()['n']

        c.execute("SELECT COUNT(*) as n FROM concepts")
        stats['total_concepts'] = c.fetchone()['n']

        c.execute("SELECT COUNT(*) as n FROM events")
        stats['total_events'] = c.fetchone()['n']

        c.execute("SELECT COUNT(*) as n FROM policy_packs")
        stats['total_policy_packs'] = c.fetchone()['n']

        c.execute("SELECT game, level, COUNT(*) as n, AVG(lines_cleared) as avg_lines, "
                  "MAX(lines_cleared) as max_lines FROM episodes GROUP BY game, level "
                  "ORDER BY game, level")
        stats['episodes_by_level'] = [dict(row) for row in c.fetchall()]

        c.execute("SELECT name, status, win_rate, confidence, evidence_count "
                  "FROM hypotheses ORDER BY game, level, name")
        stats['hypotheses'] = [dict(row) for row in c.fetchall()]

        return stats

    # ────────────────────── Lifecycle ──────────────────────

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
