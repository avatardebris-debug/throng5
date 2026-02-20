import sys, json, shutil
sys.path.insert(0, '.')
from pathlib import Path

# Validate JSON
data = json.loads(Path('experiments/tetra_inbox_dummy.json').read_text(encoding='utf-8'))
print(f'Valid JSON: {len(data)} entries in array')
print(f'Ops: {[d.get("op") for d in data]}')

# Copy to inbox (dummy stays intact)
shutil.copy('experiments/tetra_inbox_dummy.json', 'experiments/tetra_inbox.json')

from throng4.storage.experiment_db import ExperimentDB
db = ExperimentDB('experiments/experiments.db')
result = db.ingest_tetra_inbox()
db.close()
print(f'Ingestion result: {result}')
print('Dummy still exists:', Path('experiments/tetra_inbox_dummy.json').exists())
print('Inbox consumed:    ', not Path('experiments/tetra_inbox.json').exists())
