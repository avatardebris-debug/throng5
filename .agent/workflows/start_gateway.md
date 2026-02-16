---
description: Start OpenClaw gateway and verify health
---

# Start OpenClaw Gateway

Quick workflow to start the OpenClaw gateway service.

## Steps

// turbo-all

1. Start the gateway:
```powershell
openclaw gateway start
```

2. Wait for startup (3 seconds):
```powershell
Start-Sleep 3
```

3. Verify health:
```powershell
openclaw gateway health
```

Expected output: Status shows "healthy"

4. Check if Tetra is available:
```powershell
openclaw agent --agent main --message "ping" 2>&1 | Select-Object -First 5
```

---

## Troubleshooting

If gateway fails to start:

```powershell
# Check status:
openclaw gateway status

# Stop if running:
openclaw gateway stop

# Restart:
openclaw gateway start
```

If health check fails, wait a bit longer:
```powershell
Start-Sleep 5
openclaw gateway health
```

---

**Usage**: Run this workflow before any training that uses OpenClawBridge
**Auto-run**: All steps marked with `// turbo-all` will auto-execute
