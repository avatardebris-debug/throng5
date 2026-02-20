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
                fail_mode TEXT,
                active_hypothesis TEXT,
                outcome_tags TEXT,
                hypothesis_performance TEXT,
                policy_pack_version INTEGER,
                session_id TEXT,
                timestamp REAL NOT NULL
            )
        """)

        # Migrate existing DBs: episodes columns
        for col, col_type in [('fail_mode', 'TEXT'), ('active_hypothesis', 'TEXT')]:
            try:
                c.execute(f"ALTER TABLE episodes ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists

        # Migrate existing DBs: hypothesis LLM columns
        for col, col_type in [
            ('llm_score',    'REAL'),
            ('llm_priority', 'TEXT'),   # 'explore' | 'test' | 'retire' | None
            ('generation',   'INTEGER DEFAULT 0'),
            ('llm_notes',    'TEXT'),
        ]:
            try:
                c.execute(f"ALTER TABLE hypotheses ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # Column already exists

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
                   score: float = 0.0, lines_cleared: int = 0,
                   pieces_placed: int = 0, max_height: int = 0,
                   holes: int = 0, bumpiness: float = 0.0,
                   fail_mode: Optional[str] = None,
                   active_hypothesis: Optional[str] = None,
                   outcome_tags: Optional[Dict] = None,
                   hypothesis_performance: Optional[Dict] = None,
                   policy_pack_version: Optional[int] = None,
                   session_id: Optional[str] = None,
                   timestamp: Optional[float] = None) -> str:
        """Log a single episode result. Returns the episode ID."""
        eid = str(uuid.uuid4())[:8]
        ts = timestamp or time.time()
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO episodes (id, game, level, episode_num, score, lines_cleared,
                pieces_placed, max_height, holes, bumpiness, fail_mode, active_hypothesis,
                outcome_tags, hypothesis_performance, policy_pack_version, session_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (eid, game, level, episode_num, score, lines_cleared, pieces_placed,
              max_height, holes, bumpiness, fail_mode, active_hypothesis,
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

    def generate_tetra_brief(self, game: str = 'tetris',
                              last_n_episodes: int = 500) -> Dict[str, Any]:
        """
        Generate a rich, structured brief for Tetra to analyze.

        This is the primary interface for LLM-based hypothesis generation.
        Returns a dict that can be JSON-serialized and sent to Tetra as context.

        Structure:
            - game_context:       What game, what levels, what actions are possible
            - learning_summary:   Level-by-level performance over time
            - hypothesis_ledger:  Full lifecycle of every hypothesis tried
            - failure_patterns:   What the failing episodes have in common
            - success_patterns:   What the best episodes have in common
            - novelty_events:     Surprising episodes that broke expectations
            - open_questions:     Gaps the current hypotheses don't explain
            - recommended_focus:  Where Tetra's attention would be most useful

        Usage:
            brief = db.generate_tetra_brief(game='tetris')
            tetra_response = call_tetra(json.dumps(brief))
        """
        c = self.conn.cursor()

        # ── 1. Game context ───────────────────────────────────────────────
        game_context = {
            'game': game,
            'levels_trained': [],
            'total_episodes': 0,
            'training_span_days': 0,
        }
        c.execute("SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM episodes WHERE game = ?",
                  (game,))
        row = c.fetchone()
        if row[0]:
            game_context['total_episodes'] = row[2]
            span = (row[1] - row[0]) / 86400
            game_context['training_span_days'] = round(span, 1)

        c.execute("""SELECT level, COUNT(*) as n, AVG(lines_cleared) as avg,
                            MAX(lines_cleared) as best, MIN(lines_cleared) as worst,
                            AVG(bumpiness) as avg_bump, AVG(holes) as avg_holes,
                            AVG(max_height) as avg_height
                     FROM episodes WHERE game = ?
                     GROUP BY level ORDER BY level""", (game,))
        game_context['levels_trained'] = [dict(r) for r in c.fetchall()]

        # ── 2. Hypothesis ledger ──────────────────────────────────────────
        c.execute("""SELECT name, status, win_rate, evidence_count, confidence,
                            description, metadata, created_at, updated_at
                     FROM hypotheses WHERE game = ?
                     ORDER BY win_rate DESC""", (game,))
        hyp_rows = [dict(r) for r in c.fetchall()]

        # Parse metadata JSON
        for h in hyp_rows:
            if h.get('metadata'):
                try:
                    h['metadata'] = json.loads(h['metadata'])
                except Exception:
                    pass

        # Categorize
        active   = [h for h in hyp_rows if h['status'] in ('active', 'testing') and h['evidence_count'] >= 10]
        retired  = [h for h in hyp_rows if h['status'] == 'retired']
        untested = [h for h in hyp_rows if h['evidence_count'] < 10]

        hypothesis_ledger = {
            'active': active,
            'retired': retired,
            'untested_candidates': untested,
            'total': len(hyp_rows),
            'key_insight': (
                f"Top hypothesis '{active[0]['name']}' wins {active[0]['win_rate']:.0%} "
                f"of the time after {active[0]['evidence_count']} evaluations."
                if active else "No active hypotheses yet."
            ),
        }

        # ── 3. Failure patterns (recent N episodes, bottom 20%) ───────────
        c.execute("""SELECT lines_cleared, pieces_placed, max_height, holes, bumpiness,
                            fail_mode, active_hypothesis
                     FROM episodes WHERE game = ?
                     ORDER BY timestamp DESC LIMIT ?""", (game, last_n_episodes))
        recent = [dict(r) for r in c.fetchall()]

        if recent:
            sorted_by_lines = sorted(recent, key=lambda x: x['lines_cleared'] or 0)
            n = len(sorted_by_lines)
            failures = sorted_by_lines[:max(1, n // 5)]   # bottom 20%
            successes = sorted_by_lines[-(n // 5):]       # top 20%

            def _avg(lst, key):
                vals = [r.get(key) or 0 for r in lst]
                return round(sum(vals) / len(vals), 2) if vals else 0

            failure_patterns = {
                'sample_size': len(failures),
                'avg_lines': _avg(failures, 'lines_cleared'),
                'avg_pieces': _avg(failures, 'pieces_placed'),
                'avg_max_height': _avg(failures, 'max_height'),
                'avg_holes': _avg(failures, 'holes'),
                'avg_bumpiness': _avg(failures, 'bumpiness'),
                'common_fail_modes': self._count_top(failures, 'fail_mode', 3),
                'active_hypotheses_at_failure': self._count_top(failures, 'active_hypothesis', 3),
                'interpretation': (
                    f"Failures have avg height={_avg(failures,'max_height'):.1f}, "
                    f"holes={_avg(failures,'holes'):.1f}, "
                    f"bumpiness={_avg(failures,'bumpiness'):.1f}. "
                    f"Games end after only {_avg(failures,'pieces_placed'):.0f} pieces."
                ),
            }

            success_patterns = {
                'sample_size': len(successes),
                'avg_lines': _avg(successes, 'lines_cleared'),
                'avg_pieces': _avg(successes, 'pieces_placed'),
                'avg_max_height': _avg(successes, 'max_height'),
                'avg_holes': _avg(successes, 'holes'),
                'avg_bumpiness': _avg(successes, 'bumpiness'),
                'common_hypotheses': self._count_top(successes, 'active_hypothesis', 3),
                'interpretation': (
                    f"Successes have avg height={_avg(successes,'max_height'):.1f}, "
                    f"holes={_avg(successes,'holes'):.1f}, "
                    f"{_avg(successes,'lines_cleared'):.0f} lines avg."
                ),
                'delta_vs_failure': {
                    'height_delta': round(_avg(failures,'max_height') - _avg(successes,'max_height'), 2),
                    'holes_delta': round(_avg(failures,'holes') - _avg(successes,'holes'), 2),
                    'bumpiness_delta': round(_avg(failures,'bumpiness') - _avg(successes,'bumpiness'), 2),
                },
            }
        else:
            failure_patterns = success_patterns = {'sample_size': 0}

        # ── 4. Novelty events ─────────────────────────────────────────────
        c.execute("""SELECT data, timestamp FROM events
                     WHERE event_type = 'novelty'
                     ORDER BY timestamp DESC LIMIT 20""")
        novelty_events = []
        for row in c.fetchall():
            try:
                d = json.loads(row['data']) if row['data'] else {}
                d['timestamp_str'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(row['timestamp']))
                novelty_events.append(d)
            except Exception:
                pass

        # ── 5. Open questions & recommended focus ─────────────────────────
        open_questions = []

        if failure_patterns.get('sample_size', 0) > 10:
            fd = failure_patterns.get('delta_vs_failure', {}) if 'delta_vs_failure' in success_patterns else {}
            h_delta = success_patterns.get('delta_vs_failure', {}).get('height_delta', 0)
            if h_delta > 2:
                open_questions.append(
                    f"Height delta between failures and successes is {h_delta:.1f} — "
                    "is there a height threshold beyond which recovery is impossible?"
                )
            ho_delta = success_patterns.get('delta_vs_failure', {}).get('holes_delta', 0)
            if ho_delta > 1:
                open_questions.append(
                    f"Failures have {ho_delta:.1f} more holes on average — "
                    "at what hole count does performance collapse?"
                )

        if len(retired) > len(active):
            open_questions.append(
                f"{len(retired)} hypotheses have been retired vs {len(active)} active — "
                "what do the retired ones have in common? Is there a pattern to what fails?"
            )

        if untested:
            open_questions.append(
                f"{len(untested)} candidate hypotheses have fewer than 10 evaluations — "
                f"they need more evidence: {[h['name'] for h in untested[:5]]}"
            )

        # Recommended focus based on data gaps
        recommended_focus = []
        if not any(h.get('description') for h in active):
            recommended_focus.append(
                "DESCRIPTIONS MISSING: Active hypotheses have no natural language descriptions. "
                "Please provide a 1-2 sentence description of what each hypothesis means strategically."
            )
        if failure_patterns.get('avg_holes', 0) == 0 and failure_patterns.get('sample_size', 0) > 10:
            recommended_focus.append(
                "HOLES NOT TRACKED: All episodes show 0 holes — the hole detection may not be working. "
                "Height-based failure patterns may be more reliable."
            )
        recommended_focus.append(
            "LEVEL-SPECIFIC HYPOTHESES: Performance varies significantly by level. "
            "Consider hypotheses that specify WHEN they apply (e.g., 'at L7, prioritize flat boards')."
        )

        # ── 6. Inbox schema doc (instructions for LLM to write back) ─────
        inbox_schema = {
            'description': (
                'To update the hypothesis list, write a JSON array to '
                'experiments/tetra_inbox.json. The SlowLoop will ingest it '
                'at next nightly consolidation and archive the file.'
            ),
            'supported_ops': [
                {
                    'op': 'ADD',
                    'required': ['name', 'description'],
                    'optional': ['llm_score', 'llm_priority', 'llm_notes', 'game'],
                    'example': {
                        'op': 'ADD', 'name': 'reduce_holes_early',
                        'description': 'Prioritise avoiding holes in first 10 pieces.',
                        'llm_score': 0.75, 'llm_priority': 'explore',
                        'llm_notes': 'Failure data shows holes at piece 10 predict outcome.',
                        'game': 'tetris'
                    }
                },
                {
                    'op': 'UPDATE',
                    'required': ['name'],
                    'optional': ['llm_score', 'llm_priority', 'llm_notes'],
                    'example': {
                        'op': 'UPDATE', 'name': 'reduce_holes_v58',
                        'llm_score': 0.4, 'llm_priority': 'retire',
                        'llm_notes': 'Win rate plateaued; superseded by reduce_holes_early.'
                    }
                },
                {
                    'op': 'RETIRE',
                    'required': ['name'],
                    'optional': ['llm_notes'],
                    'example': {
                        'op': 'RETIRE', 'name': 'auto_top_fill',
                        'llm_notes': 'AutoMetric shows no correlation with outcome.'
                    }
                },
                {
                    'op': 'MUTATE',
                    'required': ['parent', 'name', 'description'],
                    'optional': ['llm_score', 'llm_priority', 'llm_notes', 'game'],
                    'note': 'Creates new hypothesis derived from parent; increments generation.',
                    'example': {
                        'op': 'MUTATE', 'parent': 'reduce_holes_v58',
                        'name': 'reduce_holes_v59_height_gated',
                        'description': 'Reduce holes only when height > 60% of board.',
                        'llm_score': 0.7, 'llm_priority': 'test',
                        'llm_notes': 'Adds height gate based on failure cluster analysis.'
                    }
                },
            ],
            'llm_priority_values': {
                'explore': 'New idea, low confidence, worth a quick test.',
                'test':    'Promising — allocate evidence budget to this.',
                'retire':  'Mark for deprecation after current cycle.',
            },
            'file_path': 'experiments/tetra_inbox.json',
            'format':    'JSON array (even for a single op)',
        }

        return {
            'brief_type': 'tetra_hypothesis_brief',
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'game_context': game_context,
            'hypothesis_ledger': hypothesis_ledger,
            'failure_patterns': failure_patterns,
            'success_patterns': success_patterns,
            'novelty_events': novelty_events[:10],
            'open_questions': open_questions,
            'recommended_focus': recommended_focus,
            'inbox_schema': inbox_schema,
            'prompt_for_tetra': (
                f"You are analyzing training data for a {game} AI agent. "
                f"The agent has run {game_context['total_episodes']} episodes across "
                f"{len(game_context['levels_trained'])} levels. "
                f"There are currently {len(active)} active hypotheses guiding its play. "
                f"Based on the failure patterns, success patterns, and open questions above, "
                "please: (1) identify why the agent fails, (2) suggest 2-3 new hypotheses "
                "to test using ADD ops, (3) recommend which candidates should be RETIRE'd, "
                "(4) optionally MUTATE a weak hypothesis into a stronger variant. "
                "Write your response as a valid JSON array of ops to experiments/tetra_inbox.json. "
                "See inbox_schema for the exact format."
            ),
        }

    def ingest_tetra_inbox(self,
                            inbox_path: str = 'experiments/tetra_inbox.json',
                            archive_dir: str = 'experiments/tetra_archive') -> Dict[str, Any]:
        """
        Ingest a Tetra-written inbox file and apply hypothesis updates.

        The inbox file is a JSON array of operation objects. Supported ops:

            {"op": "ADD",    "name": "...", "description": "...",
             "llm_score": 0.8, "llm_priority": "explore",
             "llm_notes": "why this is worth trying", "game": "tetris"}

            {"op": "UPDATE", "name": "...",
             "llm_score": 0.6, "llm_priority": "test", "llm_notes": "..."}

            {"op": "RETIRE", "name": "...", "llm_notes": "reason"}

            {"op": "MUTATE", "parent": "...", "name": "...",
             "description": "...", "llm_notes": "what changed"}

        After ingestion, the inbox file is moved to the archive directory
        with a timestamp suffix so nothing is ever lost.

        Returns a summary dict: {added, updated, retired, mutated, errors}
        """
        import shutil
        inbox = Path(inbox_path)
        if not inbox.exists():
            return {'skipped': True, 'reason': 'no inbox file'}

        try:
            ops = json.loads(inbox.read_text(encoding='utf-8'))
        except Exception as e:
            return {'error': f'Failed to parse inbox JSON: {e}'}

        if not isinstance(ops, list):
            ops = [ops]  # tolerate single-op dict

        summary = {'added': 0, 'updated': 0, 'retired': 0, 'mutated': 0, 'errors': []}
        c = self.conn.cursor()
        now = time.time()

        for op_dict in ops:
            try:
                op = op_dict.get('op', '').upper()

                # Skip template/comment markers (used in tetra_inbox_dummy.json)
                if op.startswith('_'):
                    continue

                if op == 'ADD':
                    name = op_dict['name']
                    existing = c.execute(
                        'SELECT id FROM hypotheses WHERE name=? AND game=?',
                        (name, op_dict.get('game', 'tetris'))
                    ).fetchone()
                    if existing:
                        summary['errors'].append(f'ADD skipped: {name} already exists')
                        continue
                    self.upsert_hypothesis(
                        name=name,
                        game=op_dict.get('game', 'tetris'),
                        description=op_dict.get('description', ''),
                        confidence=op_dict.get('llm_score', 0.5),
                        win_rate=0.0, evidence_count=0, status='candidate',
                        metadata={
                            'llm_score':    op_dict.get('llm_score'),
                            'llm_priority': op_dict.get('llm_priority', 'explore'),
                            'llm_notes':    op_dict.get('llm_notes', ''),
                            'source':       'tetra_inbox',
                        },
                    )
                    c.execute(
                        '''UPDATE hypotheses SET llm_score=?, llm_priority=?,
                           llm_notes=?, generation=0, updated_at=?
                           WHERE name=? AND game=?''',
                        (op_dict.get('llm_score'), op_dict.get('llm_priority', 'explore'),
                         op_dict.get('llm_notes', ''), now,
                         name, op_dict.get('game', 'tetris'))
                    )
                    summary['added'] += 1

                elif op == 'UPDATE':
                    name = op_dict['name']
                    c.execute(
                        '''UPDATE hypotheses SET llm_score=?, llm_priority=?,
                           llm_notes=?, updated_at=?
                           WHERE name=?''',
                        (op_dict.get('llm_score'), op_dict.get('llm_priority'),
                         op_dict.get('llm_notes'), now, name)
                    )
                    summary['updated'] += 1

                elif op == 'RETIRE':
                    name = op_dict['name']
                    c.execute(
                        '''UPDATE hypotheses SET status='retired', llm_priority='retire',
                           llm_notes=?, updated_at=? WHERE name=?''',
                        (op_dict.get('llm_notes', 'Retired by Tetra'), now, name)
                    )
                    summary['retired'] += 1

                elif op == 'MUTATE':
                    parent_name = op_dict['parent']
                    parent = c.execute(
                        'SELECT id, generation FROM hypotheses WHERE name=?',
                        (parent_name,)
                    ).fetchone()
                    parent_id  = parent['id']   if parent else None
                    parent_gen = parent['generation'] if parent else 0
                    self.upsert_hypothesis(
                        name=op_dict['name'],
                        game=op_dict.get('game', 'tetris'),
                        description=op_dict.get('description', ''),
                        confidence=op_dict.get('llm_score', 0.5),
                        win_rate=0.0, evidence_count=0, status='candidate',
                        parent_id=parent_id,
                        metadata={
                            'llm_score':    op_dict.get('llm_score'),
                            'llm_priority': op_dict.get('llm_priority', 'explore'),
                            'llm_notes':    op_dict.get('llm_notes', ''),
                            'source':       'tetra_mutation',
                            'parent_name':  parent_name,
                        },
                    )
                    c.execute(
                        '''UPDATE hypotheses SET llm_score=?, llm_priority=?,
                           llm_notes=?, generation=?, updated_at=?
                           WHERE name=?''',
                        (op_dict.get('llm_score'), op_dict.get('llm_priority', 'explore'),
                         op_dict.get('llm_notes', ''), (parent_gen or 0) + 1, now,
                         op_dict['name'])
                    )
                    summary['mutated'] += 1

                else:
                    summary['errors'].append(f'Unknown op: {op}')

            except Exception as e:
                summary['errors'].append(f"{op_dict.get('op','?')} failed: {e}")

        self.conn.commit()

        # Archive the inbox so it isn't re-processed
        Path(archive_dir).mkdir(parents=True, exist_ok=True)
        ts = time.strftime('%Y-%m-%d_%H-%M-%S')
        shutil.move(str(inbox), f'{archive_dir}/tetra_inbox_{ts}.json')

        return summary

    @staticmethod
    def _count_top(rows: list, key: str, n: int) -> Dict[str, int]:
        """Count top-N values for a key across a list of row dicts."""
        counts: Dict[str, int] = {}
        for row in rows:
            val = row.get(key) or 'unknown'
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1])[:n])

    # ────────────────────── Lifecycle ──────────────────────

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
