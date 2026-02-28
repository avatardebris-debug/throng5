# state/

SQLite-backed restart context archive.

## Create DB

```bash
python tools/restart_archive.py init
```

## Save checkpoint snapshot

```bash
python tools/restart_archive.py snapshot --summary "After offline pipeline hardening"
```

Default DB path:
- `state/restart_archive.sqlite`

This is a quick-recovery index for context resets.
