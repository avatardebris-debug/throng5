"""
ingest_tetra_inbox.py — Standalone Tetra inbox ingestor.

Called by the cron job immediately after Tetra writes tetra_inbox.json.
Ingests the ops, archives the file, and exits with code 0 (success) or 1 (error).

Usage:
    python ingest_tetra_inbox.py
    python ingest_tetra_inbox.py --db experiments/experiments.db
    python ingest_tetra_inbox.py --inbox experiments/tetra_inbox.json
"""

import sys
import io
import argparse
from throng4.storage.experiment_db import ExperimentDB

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError on ✓ ↺ etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')



def main():
    parser = argparse.ArgumentParser(description='Ingest Tetra inbox file')
    parser.add_argument('--db',    default='experiments/experiments.db')
    parser.add_argument('--inbox', default='experiments/tetra_inbox.json')
    args = parser.parse_args()

    db = ExperimentDB(args.db)
    result = db.ingest_tetra_inbox(inbox_path=args.inbox)
    db.close()

    if result.get('skipped'):
        print(f'[tetra] No inbox file found at {args.inbox} — nothing to do.')
        sys.exit(0)

    if 'error' in result:
        print(f'[tetra] ✗ Ingest failed: {result["error"]}')
        sys.exit(1)

    added   = result.get('added',   0)
    updated = result.get('updated', 0)
    retired = result.get('retired', 0)
    mutated = result.get('mutated', 0)
    errors  = result.get('errors',  [])

    print(f'[tetra] ✓ Inbox ingested: '
          f'+{added} added  ~{updated} updated  '
          f'-{retired} retired  ↺{mutated} mutated')
    if errors:
        print(f'[tetra] ⚠ {len(errors)} op error(s):')
        for e in errors:
            print(f'  ↳ {e}')
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()
