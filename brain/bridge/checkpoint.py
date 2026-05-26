"""
Checkpoint format and loaders for Puffer-pretrained RainbowDQN weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from brain.orchestrator import WholeBrain


CHECKPOINT_FORMAT = "throng5_rainbow_dqn_v1"


def save_bridge_checkpoint(
    dqn,
    path: Union[str, Path],
    *,
    game_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save RainbowDQN state plus bridge metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "game_id": game_id,
        "n_features": dqn.n_features,
        "n_actions": dqn.n_actions,
        "online_net": dqn.online_net.state_dict(),
        "target_net": dqn.target_net.state_dict(),
        "optimizer": dqn.optimizer.state_dict(),
        "total_updates": dqn._total_updates,
        "total_steps": dqn._total_steps,
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_into_striatum(striatum, path: Union[str, Path], device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a bridge checkpoint into an existing Striatum RainbowDQN.

    Striatum must already have ``use_torch=True`` (``_torch_dqn`` not None).
    """
    if striatum._torch_dqn is None:
        raise RuntimeError(
            "Striatum has no Torch/Rainbow backend. Create WholeBrain with use_torch=True."
        )
    path = Path(path)
    state = torch.load(path, map_location=device or striatum._torch_dqn.device, weights_only=False)
    if state.get("format") and state["format"] != CHECKPOINT_FORMAT:
        raise ValueError(f"Unknown checkpoint format: {state.get('format')}")

    dqn = striatum._torch_dqn
    if state["n_features"] != dqn.n_features or state["n_actions"] != dqn.n_actions:
        raise ValueError(
            f"Shape mismatch: checkpoint ({state['n_features']}, {state['n_actions']}) "
            f"vs striatum ({dqn.n_features}, {dqn.n_actions})"
        )

    dqn.load_state_dict(state)
    return {
        "path": str(path),
        "game_id": state.get("game_id"),
        "total_updates": dqn._total_updates,
        "total_steps": dqn._total_steps,
        "stats": dqn.stats(),
    }


def load_into_brain(
    brain: WholeBrain,
    path: Union[str, Path],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Load pretrained weights into ``brain.striatum``."""
    return load_into_striatum(brain.striatum, path, device=device)


def checkpoint_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Read ``game_id``, ``n_features``, ``n_actions`` without loading weights."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "format": state.get("format"),
        "game_id": state.get("game_id"),
        "n_features": state.get("n_features"),
        "n_actions": state.get("n_actions"),
        "total_updates": state.get("total_updates", 0),
        "total_steps": state.get("total_steps", 0),
    }
