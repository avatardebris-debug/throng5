"""
brain_state.py — Serializable brain state for save/load/checkpoint.

Saves and restores the ENTIRE brain state using torch.save():
  - TorchDQN weights + optimizer + epsilon (via TorchDQN.save/load)
  - CNN encoder (included in TorchDQN state)
  - WorldModel (encoder + heads + optimizer + training stats)
  - CuriosityModule (MLP weights + visit counts)
  - MetaController (slot stats + collapse state)
  - Amygdala threat model + Hippocampus buffer + Motor heuristics
  - Training statistics and episode counts

Format: Single .brain file (PyTorch checkpoint format)

Usage:
    from brain.scaling.brain_state import BrainState, AutoCheckpointer

    # Save
    BrainState.save(brain, "checkpoints/run42_ep1000.brain")

    # Load (into existing brain)
    BrainState.load("checkpoints/run42_ep1000.brain", brain)

    # Auto-checkpoint during training
    saver = AutoCheckpointer(brain, interval_episodes=100)
    saver.maybe_checkpoint()
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BrainState:
    """
    Serialize and deserialize the entire WholeBrain state.
    Uses torch.save when available, falls back to numpy .npz.
    """

    VERSION = "2.0"

    @staticmethod
    def save(brain, filepath: str) -> Dict[str, Any]:
        """
        Save the entire brain state to a file.
        Returns metadata about the save.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state: Dict[str, Any] = {
            "_version": BrainState.VERSION,
            "_timestamp": time.time(),
            "_step_count": brain._step_count,
            "_episode_count": brain._episode_count,
            "n_features": brain.n_features,
            "n_actions": brain.n_actions,
        }

        # ── TorchDQN (full state including CNN) ───────────────────────
        if hasattr(brain.striatum, '_torch_dqn') and brain.striatum._torch_dqn is not None:
            dqn = brain.striatum._torch_dqn
            state["torch_dqn"] = {
                "online_net": dqn.online_net.state_dict(),
                "target_net": dqn.target_net.state_dict(),
                "optimizer": dqn.optimizer.state_dict(),
                "epsilon": dqn.epsilon,
                "total_updates": dqn._total_updates,
                "total_steps": dqn._total_steps,
            }
            if dqn.cnn is not None:
                state["torch_dqn"]["cnn"] = dqn.cnn.state_dict()

        # ── Striatum numpy fallback ───────────────────────────────────
        if hasattr(brain.striatum, '_W1'):
            state["striatum_numpy"] = {
                "W1": brain.striatum._W1,
                "b1": brain.striatum._b1,
                "W2": brain.striatum._W2,
                "b2": brain.striatum._b2,
                "epsilon": getattr(brain.striatum, '_epsilon', 0.15),
                "total_updates": getattr(brain.striatum, '_total_updates', 0),
            }

        # ── WorldModel ────────────────────────────────────────────────
        bg = brain.basal_ganglia
        if hasattr(bg, '_world_model') and bg._world_model is not None:
            state["world_model"] = bg._world_model.save_state()

        # ── CuriosityModule ───────────────────────────────────────────
        curiosity = brain.curiosity
        state["curiosity"] = {
            "W1": curiosity._W1,
            "b1": curiosity._b1,
            "W2": curiosity._W2,
            "b2": curiosity._b2,
            "total_pred_error": curiosity._total_pred_error,
            "pred_steps": curiosity._pred_steps,
            "reward_mean": curiosity._reward_mean,
            "reward_var": curiosity._reward_var,
            "reward_count": curiosity._reward_count,
            "visit_counts": dict(curiosity._visit_counts),
            "unique_states": list(curiosity._unique_states),
            "total_visits": curiosity._total_visits,
        }

        # ── MetaController ────────────────────────────────────────────
        if brain.meta_controller is not None:
            mc = brain.meta_controller
            mc_state = {
                "is_collapsed": mc._is_collapsed,
                "locked_learner": mc._locked_learner,
                "total_steps": mc._total_steps,
                "collapse_count": mc._collapse_count,
                "active_learner_name": brain._active_learner_name,
                "slots": {},
            }
            for name, slot in mc._slots.items():
                mc_state["slots"][name] = {
                    "alpha": slot.alpha,
                    "beta_param": slot.beta_param,
                    "total_reward": slot.total_reward,
                    "total_selections": slot.total_selections,
                    "total_steps": slot.total_steps,
                    "rewards": list(slot.rewards),
                }
            state["meta_controller"] = mc_state

        # ── Amygdala/Thalamus ─────────────────────────────────────────
        amygdala = brain.amygdala
        state["amygdala"] = {
            "W1": amygdala._W1,
            "b1": amygdala._b1,
            "W2": amygdala._W2,
            "b2": amygdala._b2,
        }

        # ── Hippocampus ───────────────────────────────────────────────
        hippocampus = brain.hippocampus
        # Cap at 10k transitions to keep file size reasonable
        transitions = list(hippocampus._transitions)[-10000:]
        state["hippocampus"] = {
            "transitions": transitions,
            "total_stored": hippocampus._total_stored,
        }

        # ── Motor Cortex ──────────────────────────────────────────────
        motor = brain.motor
        state["motor"] = {
            "heuristics": dict(motor._heuristics),
            "total_actions": motor._total_actions,
            "fallback_count": motor._fallback_count,
        }

        # ── Basal Ganglia (context) ───────────────────────────────────
        state["basal_ganglia"] = {
            "steps_since_dream": bg._steps_since_dream,
        }

        # ── Write ─────────────────────────────────────────────────────
        if TORCH_AVAILABLE:
            torch.save(state, str(filepath))
        else:
            np.savez_compressed(str(filepath), state=state)

        size_mb = os.path.getsize(str(filepath)) / (1024 * 1024)
        meta = {
            "filepath": str(filepath),
            "size_mb": round(size_mb, 2),
            "step_count": brain._step_count,
            "episode_count": brain._episode_count,
            "timestamp": state["_timestamp"],
        }
        return meta

    @staticmethod
    def load(filepath: str, brain=None):
        """
        Load brain state from a checkpoint file.

        If brain is provided, loads weights into existing brain.
        If brain is None, creates a new WholeBrain with saved params.

        Returns the brain instance.
        """
        if TORCH_AVAILABLE:
            state = torch.load(str(filepath), map_location="cpu", weights_only=False)
        else:
            loaded = np.load(str(filepath), allow_pickle=True)
            state = loaded["state"].item()

        # Determine saved config
        n_features = state.get("n_features", 84)
        n_actions = state.get("n_actions", 18)
        has_torch_dqn = "torch_dqn" in state

        if brain is None:
            from brain.orchestrator import WholeBrain
            brain = WholeBrain(
                n_features=n_features,
                n_actions=n_actions,
                use_torch=has_torch_dqn,
                enable_logging=False,
            )

        brain._step_count = state.get("_step_count", 0)
        brain._episode_count = state.get("_episode_count", 0)

        # ── TorchDQN ──────────────────────────────────────────────────
        dqn_state = state.get("torch_dqn")
        if dqn_state and hasattr(brain.striatum, '_torch_dqn') and brain.striatum._torch_dqn is not None:
            dqn = brain.striatum._torch_dqn
            dqn.online_net.load_state_dict(dqn_state["online_net"])
            dqn.target_net.load_state_dict(dqn_state["target_net"])
            dqn.optimizer.load_state_dict(dqn_state["optimizer"])
            dqn.epsilon = dqn_state.get("epsilon", dqn.epsilon)
            dqn._total_updates = dqn_state.get("total_updates", 0)
            dqn._total_steps = dqn_state.get("total_steps", 0)
            if dqn.cnn is not None and "cnn" in dqn_state:
                dqn.cnn.load_state_dict(dqn_state["cnn"])

        # ── Striatum numpy ────────────────────────────────────────────
        s = state.get("striatum_numpy", {})
        if s and hasattr(brain.striatum, '_W1'):
            brain.striatum._W1 = np.asarray(s["W1"], dtype=np.float32)
            brain.striatum._b1 = np.asarray(s["b1"], dtype=np.float32)
            brain.striatum._W2 = np.asarray(s["W2"], dtype=np.float32)
            brain.striatum._b2 = np.asarray(s["b2"], dtype=np.float32)
            brain.striatum._epsilon = s.get("epsilon", 0.15)
            brain.striatum._total_updates = s.get("total_updates", 0)

        # ── WorldModel ────────────────────────────────────────────────
        wm_state = state.get("world_model")
        if wm_state and hasattr(brain.basal_ganglia, '_world_model') and brain.basal_ganglia._world_model is not None:
            brain.basal_ganglia._world_model.load_state(wm_state)

        # ── CuriosityModule ───────────────────────────────────────────
        c = state.get("curiosity", {})
        if c:
            cur = brain.curiosity
            cur._W1 = np.asarray(c["W1"], dtype=np.float32)
            cur._b1 = np.asarray(c["b1"], dtype=np.float32)
            cur._W2 = np.asarray(c["W2"], dtype=np.float32)
            cur._b2 = np.asarray(c["b2"], dtype=np.float32)
            cur._total_pred_error = c.get("total_pred_error", 0)
            cur._pred_steps = c.get("pred_steps", 0)
            cur._reward_mean = c.get("reward_mean", 0)
            cur._reward_var = c.get("reward_var", 1)
            cur._reward_count = c.get("reward_count", 0)
            from collections import defaultdict
            cur._visit_counts = defaultdict(int, c.get("visit_counts", {}))
            cur._unique_states = set(c.get("unique_states", []))
            cur._total_visits = c.get("total_visits", 0)

        # ── MetaController ────────────────────────────────────────────
        mc_state = state.get("meta_controller", {})
        if mc_state and brain.meta_controller is not None:
            mc = brain.meta_controller
            mc._is_collapsed = mc_state.get("is_collapsed", False)
            mc._locked_learner = mc_state.get("locked_learner")
            mc._total_steps = mc_state.get("total_steps", 0)
            mc._collapse_count = mc_state.get("collapse_count", 0)
            brain._active_learner_name = mc_state.get("active_learner_name", "default")
            from collections import deque
            for name, slot_data in mc_state.get("slots", {}).items():
                if name in mc._slots:
                    slot = mc._slots[name]
                    slot.alpha = slot_data.get("alpha", 1.0)
                    slot.beta_param = slot_data.get("beta_param", 1.0)
                    slot.total_reward = slot_data.get("total_reward", 0)
                    slot.total_selections = slot_data.get("total_selections", 0)
                    slot.total_steps = slot_data.get("total_steps", 0)
                    slot.rewards = deque(slot_data.get("rewards", []), maxlen=500)

        # ── Amygdala ──────────────────────────────────────────────────
        a = state.get("amygdala", {})
        if a:
            brain.amygdala._W1 = np.asarray(a["W1"], dtype=np.float32)
            brain.amygdala._b1 = np.asarray(a["b1"], dtype=np.float32)
            brain.amygdala._W2 = np.asarray(a["W2"], dtype=np.float32)
            brain.amygdala._b2 = np.asarray(a["b2"], dtype=np.float32)

        # ── Hippocampus ───────────────────────────────────────────────
        h = state.get("hippocampus", {})
        if h:
            brain.hippocampus._transitions = h.get("transitions", [])
            brain.hippocampus._total_stored = h.get("total_stored", 0)

        # ── Motor Cortex ──────────────────────────────────────────────
        m = state.get("motor", {})
        if m:
            brain.motor._heuristics = m.get("heuristics", {})
            brain.motor._total_actions = m.get("total_actions", 0)
            brain.motor._fallback_count = m.get("fallback_count", 0)

        # ── Basal Ganglia ─────────────────────────────────────────────
        bg = state.get("basal_ganglia", {})
        if bg:
            brain.basal_ganglia._steps_since_dream = bg.get("steps_since_dream", 0)

        return brain

    @staticmethod
    def info(filepath: str) -> Dict[str, Any]:
        """Read metadata from a checkpoint without fully loading."""
        if TORCH_AVAILABLE:
            state = torch.load(str(filepath), map_location="cpu", weights_only=False)
        else:
            loaded = np.load(str(filepath), allow_pickle=True)
            state = loaded["state"].item()

        return {
            "version": state.get("_version"),
            "timestamp": state.get("_timestamp"),
            "step_count": state.get("_step_count"),
            "episode_count": state.get("_episode_count"),
            "n_features": state.get("n_features"),
            "n_actions": state.get("n_actions"),
            "has_torch_dqn": "torch_dqn" in state,
            "has_world_model": "world_model" in state,
            "has_meta_controller": "meta_controller" in state,
            "size_mb": round(os.path.getsize(str(filepath)) / (1024 * 1024), 2),
        }


class AutoCheckpointer:
    """
    Automatic checkpointing during training.

    Saves at regular episode intervals and keeps the N most recent.
    """

    def __init__(
        self,
        brain,
        checkpoint_dir: str = "experiments/checkpoints",
        interval_episodes: int = 100,
        keep_last: int = 5,
    ):
        self.brain = brain
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval_episodes
        self.keep_last = keep_last
        self._last_checkpoint_ep = 0
        self._history: list = []

    def maybe_checkpoint(self) -> Optional[Dict]:
        """Check if it's time to save, and do so if needed."""
        ep = self.brain._episode_count
        if ep - self._last_checkpoint_ep >= self.interval:
            return self.checkpoint()
        return None

    def checkpoint(self) -> Dict[str, Any]:
        """Force a checkpoint now."""
        ep = self.brain._episode_count
        step = self.brain._step_count
        filename = f"brain_ep{ep:06d}_s{step:08d}.brain"
        filepath = self.checkpoint_dir / filename

        meta = BrainState.save(self.brain, str(filepath))
        self._history.append(str(filepath))
        self._last_checkpoint_ep = ep

        # Prune old checkpoints
        while len(self._history) > self.keep_last:
            old = self._history.pop(0)
            try:
                os.remove(old)
            except OSError:
                pass

        return meta
