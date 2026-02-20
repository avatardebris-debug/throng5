import argparse
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DEFAULT = ROOT / "state" / "restart_archive.sqlite"
PROJECT_STATE = ROOT / "PROJECT_STATE.md"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  git_commit TEXT,
  git_branch TEXT,
  summary TEXT
);

CREATE TABLE IF NOT EXISTS context_files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  path TEXT NOT NULL,
  role TEXT NOT NULL,
  priority INTEGER DEFAULT 50,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  title TEXT NOT NULL,
  detail TEXT NOT NULL,
  source_path TEXT,
  status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS quick_commands (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  command TEXT NOT NULL,
  purpose TEXT
);
"""

DEFAULT_FILES = [
    ("PROJECT_STATE.md", "primary_restart_index", 1, "Read first after context reset"),
    ("README.md", "overview", 5, "High-level status and resume notes"),
    ("STATUS.md", "status_legacy", 10, "Older status summary"),
    ("THRONG3_COMPLETE.md", "architecture_legacy", 15, "Throng3 diagnostic conclusions"),
    ("throng4/config.py", "paths", 2, "Source of truth for runtime paths"),
    ("throng4/llm_policy/offline_generator.py", "offline_pipeline", 3, "Offline JSON file-handshake flow"),
    ("experiments/TETRA_PROMPT.md", "prompt_contract", 4, "Tetra prompt + Atari offline protocol"),
]

DEFAULT_COMMANDS = [
    ("path_summary", "python -m throng4.config", "Verify active path roots and derived folders"),
    ("git_status", "git status --short", "See what changed since last known state"),
    ("latest_hyp_requests", "dir ~/.openclaw/workspace/memory/hyp_request_*.md", "Find latest offline requests"),
    ("latest_hyp_outputs", "dir ~/.openclaw/workspace/memory/hypotheses_*.json", "Find latest offline hypothesis outputs"),
]


def git(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def init_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO context_files(path, role, priority, notes) VALUES(?,?,?,?)",
            DEFAULT_FILES,
        )
        con.executemany(
            "INSERT INTO quick_commands(name, command, purpose) VALUES(?,?,?)",
            DEFAULT_COMMANDS,
        )
        con.commit()
    finally:
        con.close()


def snapshot(path: Path, summary: str):
    con = sqlite3.connect(path)
    try:
        commit = git(["git", "rev-parse", "HEAD"])
        branch = git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        con.execute(
            "INSERT INTO snapshots(created_at, git_commit, git_branch, summary) VALUES(?,?,?,?)",
            (now_iso(), commit, branch, summary),
        )
        if PROJECT_STATE.exists():
            con.execute(
                "INSERT INTO decisions(created_at, title, detail, source_path) VALUES(?,?,?,?)",
                (
                    now_iso(),
                    "Project state checkpoint",
                    "Captured latest restart context from PROJECT_STATE.md",
                    str(PROJECT_STATE.relative_to(ROOT)),
                ),
            )
        con.commit()
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser(description="Create and maintain restart context SQLite archive")
    parser.add_argument("action", choices=["init", "snapshot"]) 
    parser.add_argument("--db", default=str(DB_DEFAULT))
    parser.add_argument("--summary", default="Manual checkpoint")
    args = parser.parse_args()

    db = Path(args.db)
    if args.action == "init":
        init_db(db)
        print(f"Initialized: {db}")
    elif args.action == "snapshot":
        if not db.exists():
            init_db(db)
        snapshot(db, args.summary)
        print(f"Snapshot saved: {db}")


if __name__ == "__main__":
    main()
