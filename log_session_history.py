"""
log_session_history.py — Record all Throng 5 work done so far.

Run this once to create a comprehensive session log of Phases 1-3.
This is the "console log" that tracks everything we've done so future
context windows know exactly what happened and where we branched.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from brain.telemetry.session_logger import SessionLogger

log = SessionLogger("throng5_build_session")

# ── Phase 1: Foundation ──────────────────────────────────────────────

log.milestone("phase1", "Phase 1: Foundation started")

log.event("phase1", "audit", "Audited throng3/throng4/ — 83 Python files, 11 packages")
log.event("phase1", "audit", "Audited deep_rl_zoo — 21 RL algorithms available")
log.event("phase1", "audit", "Identified redundancies: dual learning paths, dual threat systems, game-specific hardcoding")

log.decision("system", "module_name", {"name": "brain/"}, reason="Clean break from throng4 naming")
log.decision("system", "canonical_source", {"source": "throng3 - Copy/throng4/"}, reason="Most evolved version (90 files vs 25 in throng4_new)")

log.event("phase1", "restructure", "Created brain/ module with 8 packages")
log.event("phase1", "restructure", "Migrated 18 core files: learning/ (4), networks/ (4), environments/ (5), games/montezuma/ (5)")

log.event("phase1", "build", "Built brain/telemetry/session_logger.py — JSONL with branch tracking")
log.event("phase1", "build", "Built brain/telemetry/context_restorer.py — log reader + divergence detection")
log.event("phase1", "build", "Built brain/message_bus.py — BrainMessage routing with priority/broadcast/halt")
log.event("phase1", "build", "Built brain/regions/base_region.py — abstract BrainRegion interface")
log.event("phase1", "build", "Built brain/config.py — centralized paths, constants, timing")

log.milestone("phase1", "Phase 1: Foundation complete — smoke test 10/10 pass")

# ── Phase 2: Brain Regions ───────────────────────────────────────────

log.milestone("phase2", "Phase 2: Brain Regions started")

log.decision("system", "threat_unification",
    {"merged": ["Amygdala", "ThreatEstimator", "ModeController"], "into": "AmygdalaThalamus"},
    reason="Three competing 'should I panic?' systems merged into one pipeline")

log.decision("system", "learner_unification",
    {"merged": ["PortableNNAgent", "MetaStackPipeline"], "into": "Striatum"},
    reason="Two competing DQN systems merged into single configurable learner")

log.event("phase2", "build", "Built brain/regions/sensory_cortex.py — perception + feature extraction")
log.event("phase2", "build", "Built brain/regions/basal_ganglia.py — SNN context + dream simulation")
log.event("phase2", "build", "Built brain/regions/amygdala_thalamus.py — UNIFIED threat + mode gating")
log.event("phase2", "build", "Built brain/regions/hippocampus.py — prioritized replay + dream storage")
log.event("phase2", "build", "Built brain/regions/striatum.py — UNIFIED DQN action-value learner")
log.event("phase2", "build", "Built brain/regions/prefrontal_cortex.py — LLM strategy + synthesis")
log.event("phase2", "build", "Built brain/regions/motor_cortex.py — fast execution + heuristic fallback")
log.event("phase2", "build", "Built brain/orchestrator.py — WholeBrain step() API")

log.milestone("phase2", "Phase 2: Brain Regions complete — integration test 100 steps, ALL PASS")

# ── Phase 3: RLZoo Integration ───────────────────────────────────────

log.milestone("phase3", "Phase 3: RLZoo Integration started")

log.event("phase3", "build", "Built brain/learning/rl_registry.py — 18 algorithms (1 builtin + 17 deep_rl_zoo)")
log.event("phase3", "build", "Built brain/learning/learner_selector.py — env fingerprint recommendations")
log.event("phase3", "build", "Built Learner abstract interface + BuiltinDQN implementation")

log.decision("system", "selector_methodology",
    {"current": "rule-based scoring", "future": "evolutionary statistical testing with promotion thresholds"},
    reason="User feedback: probe all algos, winners deprioritized from testing, >50% -> promote to 100%")

log.event("phase3", "design_note",
    "Learner selection methodology ideas saved to brain/learning/LEARNER_SELECTION_DESIGN.md",
    data={"reference": "brain/learning/LEARNER_SELECTION_DESIGN.md"})

log.milestone("phase3", "Phase 3: RLZoo Integration complete — all selector tests pass")

# ── Reference Locations ──────────────────────────────────────────────

log.event("reference", "locations", "Key reference code locations", data={
    "throng2_archive": "throng3 - Copy/archive/2026-02-20/throng2-master-copy/",
    "throng35_regions": "throng4_new/throng35/regions/ (region_base, cortex, hippocampus, striatum prototypes)",
    "deep_rl_zoo": "throng3 - Copy/deep_rl_zoo-main/ (21 algorithms)",
    "muzero": "throng3 - Copy/muzero-main/",
    "blind_registry": "~/.openclaw/workspace/blind_label_registry.json (DO NOT DELETE)",
    "learner_design": "brain/learning/LEARNER_SELECTION_DESIGN.md",
})

log.close()

print(f"Session log written: {log.log_file}")
print(f"Total events: {log._event_count}")
