"""
OpenClaw Bridge — Real-time link between discovery loop and Tetra.

Communication modes:
1. CLI subprocess: `openclaw agent --agent main --message "..."` for real-time messaging
2. Memory files: Write to ~/.openclaw/workspace/memory/ for persistent context
3. Offline queue: Buffer observations when gateway is down, replay when back

Message protocol (from Tetra's spec):
- Python sends: observation, summary, query
- Tetra replies: hypothesis, concept, diagnosis
"""

import subprocess
import sys
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field

# On Windows, openclaw is a .ps1 script — subprocess needs shell=True to find it
_SHELL = sys.platform == 'win32'

from .hypothesis import DiscoveredRule, RuleStatus, RuleLibrary


# ─── Configuration ───────────────────────────────────────────────────────────

def _load_openclaw_config() -> Dict[str, Any]:
    """Load OpenClaw config from ~/.openclaw/openclaw.json"""
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def _workspace_path() -> Path:
    """Get OpenClaw workspace path."""
    config = _load_openclaw_config()
    ws = config.get("agents", {}).get("defaults", {}).get("workspace", "")
    if ws:
        return Path(ws)
    return Path.home() / ".openclaw" / "workspace"


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """Observation to send to Tetra during gameplay."""
    game: str
    episode: int
    observation: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def to_message(self) -> str:
        """Format as JSON message for Tetra."""
        return json.dumps({
            "type": "observation",
            "id": self.obs_id,
            "timestamp": self.timestamp,
            "game": self.game,
            "episode": self.episode,
            "observation": self.observation,
            "context": self.context
        }, indent=2)


@dataclass
class GameSummary:
    """Between-game summary to send to Tetra."""
    game: str
    episodes: int
    discovered_rules: List[Dict[str, Any]]
    failed_hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_message(self) -> str:
        return json.dumps({
            "type": "summary",
            "game": self.game,
            "episodes": self.episodes,
            "discovered_rules": self.discovered_rules,
            "failed_hypotheses": self.failed_hypotheses
        }, indent=2)


@dataclass
class TetraResponse:
    """Parsed response from Tetra."""
    raw: str
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


# ─── Bridge ──────────────────────────────────────────────────────────────────

class OpenClawBridge:
    """
    Real-time bridge between the discovery loop and Tetra via OpenClaw.
    
    Usage:
        bridge = OpenClawBridge(game="GridWorld_5x5")
        
        # During gameplay
        response = bridge.send_observation(
            episode=42,
            observation="Action 2 terminates from states with dim[0]>3",
            context={"action": 2, "reward": -100, "terminated": True}
        )
        
        # Between games
        bridge.send_summary(episodes=100, rules=[...])
        
        # Query for cross-game transfer
        response = bridge.query("Which GridWorld rules might apply to Tetris?")
    """
    
    def __init__(
        self,
        game: str = "unknown",
        agent_id: str = "main",
        timeout: int = 120,
        offline_queue_dir: Optional[str] = None
    ):
        """
        Args:
            game: Current game/environment name
            agent_id: OpenClaw agent ID (default: "main")
            timeout: Max seconds to wait for Tetra's response
            offline_queue_dir: Where to buffer observations if gateway is down
        """
        self.game = game
        self.agent_id = agent_id
        self.timeout = timeout
        self.workspace = _workspace_path()
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Offline queue
        if offline_queue_dir:
            self.queue_dir = Path(offline_queue_dir)
        else:
            self.queue_dir = Path.home() / ".openclaw" / "bridge_queue"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_observations = []
        self.session_responses = []
        self.total_sent = 0
        self.total_errors = 0
        self._gateway_available = None  # lazy check
    
    # ─── Core messaging ─────────────────────────────────────────────────
    
    def _send_to_agent(self, message: str) -> TetraResponse:
        """
        Send a message to Tetra via `openclaw agent` CLI.
        
        Returns:
            TetraResponse with Tetra's reply
        """
        try:
            result = subprocess.run(
                [
                    "openclaw", "agent",
                    "--agent", self.agent_id,
                    "--message", message
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='replace',
                shell=_SHELL
            )
            
            if result.returncode == 0:
                self.total_sent += 1
                self._gateway_available = True
                
                # Parse response
                raw = result.stdout.strip()
                return self._parse_response(raw)
            else:
                self.total_errors += 1
                error_msg = result.stderr.strip() or result.stdout.strip()
                return TetraResponse(
                    raw=error_msg,
                    success=False,
                    error=f"CLI exit code {result.returncode}: {error_msg[:200]}"
                )
                
        except subprocess.TimeoutExpired:
            self.total_errors += 1
            return TetraResponse(
                raw="",
                success=False,
                error=f"Timeout after {self.timeout}s"
            )
        except FileNotFoundError:
            self.total_errors += 1
            self._gateway_available = False
            return TetraResponse(
                raw="",
                success=False,
                error="openclaw CLI not found in PATH"
            )
    
    def _parse_response(self, raw: str) -> TetraResponse:
        """Parse Tetra's raw text response into structured data."""
        response = TetraResponse(raw=raw)
        
        # Try to extract JSON blocks from the response
        try:
            # Look for JSON in the response
            if "{" in raw:
                start = raw.index("{")
                # Find matching closing brace
                depth = 0
                for i, c in enumerate(raw[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            json_str = raw[start:i+1]
                            data = json.loads(json_str)
                            if "hypotheses" in data:
                                response.hypotheses = data["hypotheses"]
                            if "concepts" in data:
                                response.concepts = data["concepts"]
                            break
        except (json.JSONDecodeError, ValueError):
            pass  # Response is plain text, that's fine
        
        return response
    
    # ─── Real-time observation ───────────────────────────────────────────
    
    def send_observation(
        self,
        episode: int,
        observation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TetraResponse:
        """
        Send a real-time observation during gameplay.
        
        Args:
            episode: Current episode number
            observation: Human-readable observation string
            context: Optional structured context (state, action, reward, etc.)
            
        Returns:
            TetraResponse with Tetra's diagnosis/hypothesis
        """
        obs = Observation(
            game=self.game,
            episode=episode,
            observation=observation,
            context=context or {}
        )
        
        self.session_observations.append(obs)
        
        message = obs.to_message()
        
        # Try live send
        response = self._send_to_agent(message)
        
        if response.success:
            self.session_responses.append(response)
        else:
            # Queue for later
            self._queue_observation(obs)
        
        return response
    
    def send_observation_batch(
        self,
        observations: List[Dict[str, Any]]
    ) -> TetraResponse:
        """
        Send multiple observations as a batch (more efficient).
        
        Args:
            observations: List of {"episode": int, "observation": str, "context": dict}
        """
        batch = {
            "type": "observation_batch",
            "game": self.game,
            "count": len(observations),
            "observations": observations
        }
        
        return self._send_to_agent(json.dumps(batch, indent=2))
    
    # ─── Between-game summary ────────────────────────────────────────────
    
    def send_summary(
        self,
        episodes: int,
        rules: Optional[List[DiscoveredRule]] = None,
        failed_hypotheses: Optional[List[Dict[str, Any]]] = None
    ) -> TetraResponse:
        """
        Send between-game summary with discovered rules.
        
        Args:
            episodes: Total episodes played
            rules: Discovered rules from the game
            failed_hypotheses: Hypotheses that didn't pan out
        """
        discovered = []
        if rules:
            for r in rules:
                discovered.append({
                    "id": r.id,
                    "label": r.description,
                    "status": r.status.value,
                    "confidence": r.confidence,
                    "n_tests": r.n_tests,
                    "success_rate": r.n_successes / max(1, r.n_tests)
                })
        
        summary = GameSummary(
            game=self.game,
            episodes=episodes,
            discovered_rules=discovered,
            failed_hypotheses=failed_hypotheses or []
        )
        
        message = summary.to_message()
        response = self._send_to_agent(message)
        
        # Also write to memory
        self._write_daily_memory(
            f"## Game Summary: {self.game}\n"
            f"- Episodes: {episodes}\n"
            f"- Active rules: {len([r for r in (rules or []) if r.status == RuleStatus.ACTIVE])}\n"
            f"- Anti-policies: {len([r for r in (rules or []) if r.status == RuleStatus.ANTI_POLICY])}\n"
            f"- Dormant: {len([r for r in (rules or []) if r.status == RuleStatus.DORMANT])}\n"
        )
        
        return response
    
    # ─── Cross-game query ────────────────────────────────────────────────
    
    def query(self, question: str) -> TetraResponse:
        """
        Ask Tetra a free-form question (e.g., cross-game transfer).
        
        Args:
            question: Natural language question
        """
        msg = json.dumps({
            "type": "query",
            "game": self.game,
            "question": question
        }, indent=2)
        
        return self._send_to_agent(msg)
    
    def query_transfer(
        self,
        source_game: str,
        target_game: str,
        transferable_rules: List[DiscoveredRule]
    ) -> TetraResponse:
        """
        Ask Tetra which rules from source_game might apply to target_game.
        """
        rules_summary = []
        for r in transferable_rules:
            rules_summary.append({
                "id": r.id,
                "description": r.description,
                "feature": r.feature,
                "confidence": r.confidence
            })
        
        msg = json.dumps({
            "type": "transfer_query",
            "source_game": source_game,
            "target_game": target_game,
            "rules": rules_summary,
            "question": f"Which of these {len(rules_summary)} rules from {source_game} "
                       f"might apply to {target_game}? Rate each 0-1 for transferability."
        }, indent=2)
        
        return self._send_to_agent(msg)
    
    # ─── Memory management ───────────────────────────────────────────────
    
    def _write_daily_memory(self, content: str):
        """Append to today's memory file."""
        today = datetime.now().strftime("%Y-%m-%d")
        memory_file = self.memory_dir / f"{today}.md"
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(f"\n### [{timestamp}] {self.game}\n")
            f.write(content)
            f.write("\n")
    
    def write_rule_to_memory(self, rule: DiscoveredRule):
        """Write a discovered rule to today's memory."""
        self._write_daily_memory(
            f"**Rule Discovered:** {rule.description}\n"
            f"- Status: {rule.status.value}\n"
            f"- Confidence: {rule.confidence:.2f}\n"
            f"- Tests: {rule.n_tests} ({rule.n_successes} successes)\n"
            f"- Source: {rule.source}\n"
            f"- Stochasticity: {rule.stochasticity:.2f}\n"
        )
    
    # ─── Offline queue ───────────────────────────────────────────────────
    
    def _queue_observation(self, obs: Observation):
        """Buffer observation for later replay."""
        queue_file = self.queue_dir / f"obs_{obs.obs_id}.json"
        with open(queue_file, "w") as f:
            json.dump({
                "observation": obs.to_message(),
                "queued_at": time.time()
            }, f)
    
    def replay_queue(self) -> int:
        """Replay queued observations. Returns count replayed."""
        replayed = 0
        
        for queue_file in sorted(self.queue_dir.glob("obs_*.json")):
            with open(queue_file) as f:
                data = json.load(f)
            
            response = self._send_to_agent(data["observation"])
            
            if response.success:
                queue_file.unlink()  # Delete from queue
                replayed += 1
            else:
                break  # Gateway still down, stop
        
        return replayed
    
    def queue_size(self) -> int:
        """Number of queued observations."""
        return len(list(self.queue_dir.glob("obs_*.json")))
    
    # ─── Gateway health ──────────────────────────────────────────────────
    
    def check_gateway(self) -> bool:
        """Quick check: is the gateway alive?"""
        try:
            result = subprocess.run(
                ["openclaw", "gateway", "health"],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace',
                shell=_SHELL
            )
            self._gateway_available = result.returncode == 0
            return self._gateway_available
        except Exception:
            self._gateway_available = False
            return False
    
    @property
    def is_available(self) -> bool:
        """Is the gateway available? Uses cached value if recent."""
        if self._gateway_available is None:
            return self.check_gateway()
        return self._gateway_available
    
    # ─── Session info ────────────────────────────────────────────────────
    
    def get_summary(self) -> str:
        """Get bridge session summary."""
        return (
            f"OpenClaw Bridge ({self.game}):\n"
            f"  Agent: {self.agent_id}\n"
            f"  Gateway: {'UP' if self.is_available else 'DOWN'}\n"
            f"  Messages sent: {self.total_sent}\n"
            f"  Errors: {self.total_errors}\n"
            f"  Observations this session: {len(self.session_observations)}\n"
            f"  Queue size: {self.queue_size()}\n"
            f"  Workspace: {self.workspace}"
        )
