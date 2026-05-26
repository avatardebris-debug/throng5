# Testing Notes

Use marker filtering for a fast local loop (throng5 root + `tests/` only):

```powershell
python -m pytest -m "not slow" -q
```

Fast loop for the core wiring/integration suite:

```powershell
python -m pytest -m "not slow" -q test_option_critic_wiring.py test_oc_ctx_parity.py test_brain_integration.py test_blind_protocol.py test_puffer_bridge.py test_phase3_boundary.py test_intentional_planning_stack.py test_integration_smoke.py test_openclaw_bridge.py
```

Run full coverage (including slow integration/pretrain checks):

```powershell
python -m pytest -q
```

`slow` tests are intentionally longer-running checks (for example, serial pretrain
or extended Option-Critic wiring validation) and are still part of the full suite.
