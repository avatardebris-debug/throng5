"""
Rule Archive — SQLite-backed persistent storage for all discovered rules.

Nothing is ever deleted. Rules from all environments are stored with full context,
searchable, and revisitable. Supports:
- Confidence decay across sessions
- Cross-game pattern matching
- Re-evaluation triggers (staleness, curriculum change)
- Environment context tracking
"""

import sqlite3
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from .hypothesis import DiscoveredRule, RuleStatus, RuleLibrary


class RuleArchive:
    """
    SQLite-backed archive of ALL discovered rules from ALL environments.
    
    Tables:
    - rules: All discovered rules with full serialization
    - environments: Environment metadata (profile summaries)
    - sessions: Training session metadata
    """
    
    def __init__(self, db_path: str = "policy_archive.db"):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self._create_tables()
    
    def _create_tables(self):
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()
        
        # Rules table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                feature TEXT NOT NULL,
                direction TEXT,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                stochasticity REAL DEFAULT 0.0,
                n_tests INTEGER DEFAULT 0,
                n_successes INTEGER DEFAULT 0,
                n_failures INTEGER DEFAULT 0,
                source TEXT DEFAULT 'micro_test',
                discovered_at REAL NOT NULL,
                last_tested REAL NOT NULL,
                decay_rate REAL DEFAULT 0.01,
                environment_context TEXT DEFAULT '',
                transferable INTEGER DEFAULT 0,
                anti_policy_id TEXT,
                parent_rule_id TEXT,
                conditions TEXT,  -- JSON
                outcome_distribution TEXT,  -- JSON
                full_data TEXT NOT NULL  -- Full serialized DiscoveredRule
            )
        """)
        
        # Environments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS environments (
                env_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                obs_shape TEXT,  -- JSON
                n_actions INTEGER,
                reward_sparsity REAL,
                profile_data TEXT,  -- JSON of full EnvProfile
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL
            )
        """)
        
        # Sessions table (for tracking when rules were discovered)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                env_id TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL,
                n_rules_discovered INTEGER DEFAULT 0,
                notes TEXT,
                FOREIGN KEY (env_id) REFERENCES environments(env_id)
            )
        """)
        
        # Indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_env ON rules(environment_context)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_confidence ON rules(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_transferable ON rules(transferable)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rules_last_tested ON rules(last_tested)")
        
        self.conn.commit()
    
    def store_rule(self, rule: DiscoveredRule, session_id: Optional[str] = None):
        """
        Store or update a rule in the archive.
        
        Args:
            rule: DiscoveredRule to store
            session_id: Optional session ID for tracking
        """
        cursor = self.conn.cursor()
        
        rule_dict = rule.to_dict()
        
        cursor.execute("""
            INSERT OR REPLACE INTO rules (
                id, description, feature, direction, status, confidence,
                stochasticity, n_tests, n_successes, n_failures,
                source, discovered_at, last_tested, decay_rate,
                environment_context, transferable, anti_policy_id, parent_rule_id,
                conditions, outcome_distribution, full_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.id,
            rule.description,
            rule.feature,
            rule.direction,
            rule.status.value,
            rule.confidence,
            rule.stochasticity,
            rule.n_tests,
            rule.n_successes,
            rule.n_failures,
            rule.source,
            rule.discovered_at,
            rule.last_tested,
            rule.decay_rate,
            rule.environment_context,
            1 if rule.transferable else 0,
            rule.anti_policy_id,
            rule.parent_rule_id,
            json.dumps(rule.conditions),
            json.dumps(rule.outcome_distribution.to_dict()),
            json.dumps(rule_dict)
        ))
        
        self.conn.commit()
    
    def store_library(self, library: RuleLibrary, session_id: Optional[str] = None):
        """Store all rules from a RuleLibrary."""
        for rule in library.rules.values():
            self.store_rule(rule, session_id)
    
    def get_rule(self, rule_id: str) -> Optional[DiscoveredRule]:
        """Retrieve a specific rule by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT full_data FROM rules WHERE id = ?", (rule_id,))
        row = cursor.fetchone()
        
        if row:
            data = json.loads(row['full_data'])
            return DiscoveredRule.from_dict(data)
        return None
    
    def get_rules_for_env(self, env_context: str) -> List[DiscoveredRule]:
        """Get all rules from a specific environment."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT full_data FROM rules WHERE environment_context = ?",
            (env_context,)
        )
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def get_active_rules(self, env_context: Optional[str] = None) -> List[DiscoveredRule]:
        """Get all active (high-confidence) rules, optionally filtered by environment."""
        cursor = self.conn.cursor()
        
        if env_context:
            cursor.execute(
                "SELECT full_data FROM rules WHERE status = ? AND environment_context = ?",
                (RuleStatus.ACTIVE.value, env_context)
            )
        else:
            cursor.execute(
                "SELECT full_data FROM rules WHERE status = ?",
                (RuleStatus.ACTIVE.value,)
            )
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def get_dormant_rules(self, min_age_hours: float = 24.0) -> List[DiscoveredRule]:
        """Get dormant rules that haven't been tested recently."""
        cursor = self.conn.cursor()
        cutoff = time.time() - (min_age_hours * 3600)
        
        cursor.execute(
            "SELECT full_data FROM rules WHERE status = ? AND last_tested < ?",
            (RuleStatus.DORMANT.value, cutoff)
        )
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def get_anti_policies(self, env_context: Optional[str] = None) -> List[DiscoveredRule]:
        """Get all anti-policies."""
        cursor = self.conn.cursor()
        
        if env_context:
            cursor.execute(
                "SELECT full_data FROM rules WHERE status = ? AND environment_context = ?",
                (RuleStatus.ANTI_POLICY.value, env_context)
            )
        else:
            cursor.execute(
                "SELECT full_data FROM rules WHERE status = ?",
                (RuleStatus.ANTI_POLICY.value,)
            )
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def get_transferable_rules(self) -> List[DiscoveredRule]:
        """Get rules marked as potentially transferable to other games."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT full_data FROM rules WHERE transferable = 1")
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def get_stale_rules(self, max_age_hours: float = 48.0) -> List[DiscoveredRule]:
        """Get rules that haven't been tested recently (candidates for re-evaluation)."""
        cursor = self.conn.cursor()
        cutoff = time.time() - (max_age_hours * 3600)
        
        cursor.execute(
            "SELECT full_data FROM rules WHERE last_tested < ? AND status != ?",
            (cutoff, RuleStatus.DORMANT.value)
        )
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def search_by_description(self, query: str, limit: int = 20) -> List[DiscoveredRule]:
        """Search rules by description (case-insensitive substring match)."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT full_data FROM rules WHERE description LIKE ? LIMIT ?",
            (f"%{query}%", limit)
        )
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def search_by_feature(self, feature: str) -> List[DiscoveredRule]:
        """Get all rules affecting a specific feature."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT full_data FROM rules WHERE feature = ?", (feature,))
        
        rules = []
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rules.append(DiscoveredRule.from_dict(data))
        return rules
    
    def apply_decay_all(self, decay_hours: float = 1.0):
        """
        Apply confidence decay to all rules in the archive.
        
        Args:
            decay_hours: How many hours have passed since last decay
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT full_data FROM rules")
        
        current_time = time.time()
        updated_rules = []
        
        for row in cursor.fetchall():
            data = json.loads(row['full_data'])
            rule = DiscoveredRule.from_dict(data)
            rule.apply_confidence_decay(current_time)
            rule.update_status()
            updated_rules.append(rule)
        
        # Batch update
        for rule in updated_rules:
            self.store_rule(rule)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get archive statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        # Total rules
        cursor.execute("SELECT COUNT(*) as count FROM rules")
        stats['total_rules'] = cursor.fetchone()['count']
        
        # By status
        cursor.execute("SELECT status, COUNT(*) as count FROM rules GROUP BY status")
        stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # By environment
        cursor.execute("SELECT environment_context, COUNT(*) as count FROM rules GROUP BY environment_context")
        stats['by_environment'] = {row['environment_context']: row['count'] for row in cursor.fetchall()}
        
        # Transferable
        cursor.execute("SELECT COUNT(*) as count FROM rules WHERE transferable = 1")
        stats['transferable'] = cursor.fetchone()['count']
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) as avg_conf FROM rules")
        stats['avg_confidence'] = cursor.fetchone()['avg_conf']
        
        return stats
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
