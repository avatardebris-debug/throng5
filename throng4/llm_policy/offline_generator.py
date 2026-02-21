"""
Offline LLM Hypothesis Generator (v2)
=====================================

Reads semantic game logs, compresses to key events, and requests hypothesis
discovery from the Tetra LLM via a file-based handshake.

Protocol:
  1. Python writes full prompt to MEMORY_DIR/hyp_request_<ts>.md
  2. Python sends Tetra a short CLI message: read <request_file>, write JSON to <output_file>
  3. Tetra responds with ACK: WRITTEN <absolute_output_path> in chat (not the JSON)
  4. Python detects ACK in stdout OR polls <output_file> for valid JSON
  5. Python validates schema, retries once with a repair prompt if invalid
  6. Hypotheses ingested into RuleLibrary
"""

import json
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from throng4.config import (
    MEMORY_DIR, RULES_DIR, REQUIRED_HYPOTHESIS_KEYS,
    rules_path, ensure_dirs
)
from throng4.llm_policy.openclaw_bridge import OpenClawBridge
from throng4.llm_policy.hypothesis import DiscoveredRule, RuleLibrary


# Stable anonymized labels for blind hypothesis generation
# Maps real game_id -> Environment-X label so Tetra never sees game names
_GAME_LABELS: Dict[str, str] = {}
_LABEL_COUNTER = [0]  # mutable int via list

def _get_blind_label(game_id: str) -> str:
    """Return a stable anonymous label (Environment-A, -B, ...) for a game_id."""
    if game_id not in _GAME_LABELS:
        idx = _LABEL_COUNTER[0]
        letter = chr(ord('A') + (idx % 26))
        suffix = '' if idx < 26 else str(idx // 26)
        _GAME_LABELS[game_id] = f"Environment-{letter}{suffix}"
        _LABEL_COUNTER[0] += 1
    return _GAME_LABELS[game_id]


class OfflineGenerator:
    """Processes game logs and runs offline hypothesis generation via file handshake."""

    def __init__(self, game_id: str, agent_id: str = "main"):
        self.game_id = game_id
        self.blind_label = _get_blind_label(game_id)  # e.g. "Environment-A"
        self.bridge = OpenClawBridge(game=game_id, agent_id=agent_id)
        self.library = self._load_library()
        ensure_dirs()

    # ------------------------------------------------------------------
    # Library persistence
    # ------------------------------------------------------------------

    def _load_library(self) -> RuleLibrary:
        path = rules_path(self.game_id)
        lib = RuleLibrary()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for r_dict in data.get("rules", {}).values():
                    lib.add_rule(DiscoveredRule.from_dict(r_dict))
            except Exception as e:
                print(f"Warning: could not load RuleLibrary: {e}")
        return lib

    def save_library(self):
        path = rules_path(self.game_id)
        path.write_text(json.dumps(self.library.to_dict(), indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Trajectory compression
    # ------------------------------------------------------------------

    def compress_trajectory(self, trajectory: List[Dict[str, Any]]) -> str:
        """
        Compress full trajectory to key events:
          - First / last step
          - Steps with reward != 0
          - Steps where resource count decreases  (<-- RESOURCE LOST annotated)
          - Every 50th step for context

        Prefers `blind_obs` field (abstract format) over `obs` (game-specific)
        so the compressed log is safe to send to Tetra in blind mode.
        """
        if not trajectory:
            return "Empty trajectory"

        key_events = []
        last_resource: Optional[float] = None

        for i, step_data in enumerate(trajectory):
            step_idx = step_data.get("step", i)
            # Prefer blind_obs; fall back to obs for backwards compatibility
            obs_str = step_data.get("blind_obs") or step_data.get("obs", "")

            reward_val = 0.0
            for marker in ("reward:", "Reward: "):
                if marker in obs_str:
                    try:
                        reward_val = float(obs_str.split(marker)[1].split("|")[0].strip())
                        break
                    except ValueError:
                        pass

            # Resource detection: works for both blind_obs (rsrc:) and obs (Lives:)
            resource_val = last_resource
            for marker, scale in (("rsrc:", 1.0), ("Lives: ", 0.2)):
                if marker in obs_str:
                    try:
                        resource_val = float(obs_str.split(marker)[1].split()[0].strip())
                        resource_val *= scale
                        break
                    except (ValueError, IndexError):
                        pass

            is_key = (
                i == 0 or i == len(trajectory) - 1
                or reward_val != 0.0
                or step_idx % 50 == 0
            )
            if last_resource is not None and resource_val is not None and resource_val < last_resource:
                is_key = True
                obs_str += "  <-- RESOURCE LOST"

            if is_key:
                key_events.append(obs_str)

            last_resource = resource_val

        return "\n".join(key_events)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, log: str, out_path: Path) -> str:
        """
        Build the full prompt Tetra reads from the request file.
        Uses blind_label (not game_id) so Tetra never sees the real game name.
        Includes generality field so hypotheses are tagged for rule-tree placement.
        """
        label = self.blind_label
        return (
            f"# Hypothesis Request — {label}\n\n"
            f"You are in OFFLINE BATCH MODE. This is a BLIND generalization task.\n"
            f"You do not know the game name. Reason only from the log data.\n\n"
            f"## Step 1 — Write your response to this EXACT file path:\n"
            f"  {out_path}\n\n"
            f"## Step 2 — Write atomically:\n"
            f"  First write to: {out_path}.tmp\n"
            f"  Then rename: {out_path}.tmp → {out_path}\n\n"
            f"## Step 3 — Confirm in chat with ONLY this line (no other text):\n"
            f"  ACK: WRITTEN {out_path}\n\n"
            f"## Step 4 — JSON schema (top-level key 'hypotheses', list of objects):\n"
            f"  id          (string)  snake_case slug\n"
            f"  description (string)  one English sentence using ABSTRACT terms only\n"
            f"                        (agent, target, resource, threat — not game names)\n"
            f"  object      (string)  abstract object: agent, target, threat, resource\n"
            f"  feature     (string)  abstract state var: agent_x, target_y, rsrc, threat_prox\n"
            f"  direction   (string)  maximize | minimize | increase | decrease | avoid\n"
            f"  trigger     (string)  what event precedes the reward or resource loss\n"
            f"  confidence  (float)   0.0–1.0\n"
            f"  generality  (string)  universal | class | instance\n"
            f"    universal = true for any 2D game with an agent + target\n"
            f"    class     = true for games with similar structure (e.g. intercept/dodge)\n"
            f"    instance  = specific to this particular environment\n\n"
            f"Aim for 3–6 hypotheses. Reference specific step numbers or abstract field values.\n\n"
            f"---\n\n"
            f"## {label} Log\n\n"
            f"{log}\n"
        )

    # ------------------------------------------------------------------
    # File handshake + polling
    # ------------------------------------------------------------------

    def _write_request_file(self, prompt: str) -> Path:
        ts = int(time.time())
        req_path = MEMORY_DIR / f"hyp_request_{ts}.md"
        req_path.write_text(prompt, encoding="utf-8")
        return req_path

    def _validate_hypotheses(self, hypotheses: list) -> List[str]:
        """Return list of error strings; empty = valid."""
        errors = []
        for i, h in enumerate(hypotheses):
            missing = REQUIRED_HYPOTHESIS_KEYS - set(h.keys())
            if missing:
                errors.append(f"hypothesis[{i}] missing keys: {missing}")
        return errors

    def _parse_ack_path(self, raw_stdout: str) -> Optional[Path]:
        """Extract confirmed path from 'ACK: WRITTEN <path>' in stdout."""
        for line in (raw_stdout or "").splitlines():
            if line.strip().startswith("ACK: WRITTEN"):
                parts = line.strip().split("ACK: WRITTEN", 1)
                if len(parts) == 2:
                    return Path(parts[1].strip())
        return None

    def _poll_for_file(self, out_path: Path, timeout: int = 180, interval: int = 5) -> dict:
        """Poll for valid JSON at out_path. Returns parsed dict or {}."""
        deadline = time.time() + timeout
        print(f"  Polling {out_path.name} (timeout={timeout}s)...", flush=True)
        while time.time() < deadline:
            if out_path.exists():
                try:
                    data = json.loads(out_path.read_text(encoding="utf-8").strip())
                    print("  ✅ Response file received.")
                    return data
                except json.JSONDecodeError:
                    pass  # File mid-write or empty; keep polling
            time.sleep(interval)
        print(f"  ⏰ Timeout: no valid file within {timeout}s.")
        return {}

    # ------------------------------------------------------------------
    # Library integration
    # ------------------------------------------------------------------

    def _add_hypotheses_to_library(self, hypotheses: list):
        """Validate, label, and ingest hypotheses into the RuleLibrary."""
        errors = self._validate_hypotheses(hypotheses)
        if errors:
            print(f"  ⚠️  Schema warnings:\n" + "\n".join(f"    {e}" for e in errors))

        for hyp in hypotheses:
            desc = (hyp.get("description") or hyp.get("text")
                    or hyp.get("label") or "")
            if not desc:
                obj = hyp.get("object") or hyp.get("feature") or "element"
                trigger = hyp.get("trigger") or "certain conditions"
                direction = hyp.get("direction") or "maximize"
                desc = f"{direction.capitalize()} {obj} — triggered by: {trigger}"

            rule = DiscoveredRule(
                id=hyp.get("id", f"rule_{int(time.time()*1000)}"),
                description=desc,
                feature=hyp.get("feature") or hyp.get("object") or "unknown",
                direction=hyp.get("direction", "maximize"),
                source="offline_batch",
                environment_context=self.game_id,
                confidence=float(hyp.get("confidence", 0.5))
            )
            self.library.add_rule(rule)
            print(f"  + [{rule.feature}] ({rule.confidence:.0%}) {rule.description}")

        self.save_library()
        lib_path = rules_path(self.game_id)
        print(f"\n  Saved {len(hypotheses)} rules → {lib_path}")

    # ------------------------------------------------------------------
    # Manual injection (for when Tetra's reply came to chat instead)
    # ------------------------------------------------------------------

    def ingest_response(self, raw_json: str):
        """Parse and ingest a raw JSON string pasted from Tetra's chat reply."""
        try:
            data = json.loads(raw_json.strip())
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            return
        hypotheses = data.get("hypotheses", [])
        if not hypotheses:
            print("⚠️  No 'hypotheses' key in provided JSON.")
            return
        print(f"✅ Injecting {len(hypotheses)} hypotheses from manual paste.")
        self._add_hypotheses_to_library(hypotheses)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_log(self, log_path: str):
        """Read a game log, request hypotheses from Tetra, ingest results."""
        print(f"Loading log from {log_path}...")
        try:
            data = json.loads(Path(log_path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            print("❌ Log file not found.")
            return

        traj = data.get("trajectory", [])
        if not traj:
            print("No trajectory data found.")
            return

        compressed = self.compress_trajectory(traj)
        n_key = len(compressed.splitlines())
        print(f"Trajectory: {len(traj)} steps → {n_key} key events ({n_key/len(traj):.1%})")

        # Where Tetra should write its JSON response (absolute path)
        ts = int(time.time())
        out_path = (MEMORY_DIR / f"hypotheses_{ts}.json").resolve()

        prompt = self._build_prompt(compressed, out_path)
        req_path = self._write_request_file(prompt)

        print(f"\n📄 Request file  : {req_path.name}")
        print(f"📥 Output expected: {out_path.name}")
        print(f"🔒 Blind label   : {self.blind_label}  (game_id hidden from Tetra)")


        # Short CLI message — doesn't dump the whole log into chat context
        short_msg = (
            f"Offline batch for {self.game_id}. "
            f"Read request from memory file: {req_path.name}. "
            f"Write JSON response to: {out_path}. "
            f"Confirm with: ACK: WRITTEN {out_path}"
        )
        response = self.bridge.send_observation(episode=data.get("episodes", 1),
                                                 observation=short_msg)

        # Check for ACK token in stdout (fastest path)
        ack_path = self._parse_ack_path(getattr(response, "raw", ""))
        if ack_path and ack_path.exists():
            print(f"  ✅ ACK token received, reading from: {ack_path}")
            out_path = ack_path  # honour actual path Tetra wrote to

        # Poll for file (handles async write lag)
        result = self._poll_for_file(out_path, timeout=180)
        hypotheses = result.get("hypotheses", [])

        # Fallback: Tetra replied via stdout rather than file
        if not hypotheses and getattr(response, "hypotheses", None):
            print("  ℹ️  Using stdout response as file fallback.")
            hypotheses = response.hypotheses

        if not hypotheses:
            print("\n❌ No hypotheses received.")
            print(f"   If Tetra replied in chat, save the JSON to a file and run:")
            print(f"   python -m throng4.llm_policy.offline_generator --inject <path>")
            return

        # Validate schema — retry once with a repair prompt if invalid
        errors = self._validate_hypotheses(hypotheses)
        if errors:
            print(f"\n⚠️  Schema errors in response, sending repair request...")
            repair_msg = (
                f"Your previous response to {req_path.name} had schema errors: "
                f"{errors}. Please rewrite to {out_path} with the correct fields: "
                f"id, description, object, feature, direction, trigger, confidence. "
                f"Confirm: ACK: WRITTEN {out_path}"
            )
            self.bridge.send_observation(episode=1, observation=repair_msg)
            result = self._poll_for_file(out_path, timeout=120)
            hypotheses = result.get("hypotheses", hypotheses)  # use original if retry also fails

        print(f"\n✅ {len(hypotheses)} hypotheses ingested.")
        self._add_hypotheses_to_library(hypotheses)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Offline LLM Generator")
    parser.add_argument("--log", type=str, default="atari_offline_log.json",
                        help="Path to JSON gameplay log")
    parser.add_argument("--inject", type=str, default=None,
                        help="Path to JSON file with Tetra's manual reply")
    parser.add_argument("--game", type=str, default="ALE/Breakout-v5",
                        help="Game ID (e.g. ALE/Breakout-v5)")
    args = parser.parse_args()

    gen = OfflineGenerator(game_id=args.game)
    if args.inject:
        p = Path(args.inject)
        if not p.exists():
            print(f"❌ File not found: {p}")
        else:
            gen.ingest_response(p.read_text(encoding="utf-8"))
    else:
        gen.process_log(args.log)
