"""
transfer.py — Transfer learned representations between games.

Provides utilities for:
1. CNN feature transfer: copy early conv layers, reinitialize game-specific head
2. DQN shared-layer transfer: copy value stream, reinitialize action head
3. Full brain warm-start: carry all compatible weights

Usage:
    from brain.learning.transfer import transfer_cnn, transfer_dqn_shared

    transfer_cnn(source_brain, target_brain, freeze_layers=2)
    transfer_dqn_shared(source_brain, target_brain)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def transfer_cnn(
    source_brain,
    target_brain,
    freeze_layers: int = 2,
) -> dict:
    """
    Transfer CNN encoder weights from one brain to another.

    Copies all conv layers, freezes the first N, and reinitializes
    the final FC layer for the new task.

    Args:
        source_brain: WholeBrain instance to copy FROM
        target_brain: WholeBrain instance to copy TO
        freeze_layers: Number of early conv layers to freeze (0 = no freezing)

    Returns:
        Report dict with transfer details.
    """
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "PyTorch not available"}

    source_cnn = _get_cnn(source_brain)
    target_cnn = _get_cnn(target_brain)

    if source_cnn is None or target_cnn is None:
        return {"status": "skipped", "reason": "One or both brains lack CNN encoder"}

    # Copy all matching parameters
    source_state = source_cnn.state_dict()
    target_state = target_cnn.state_dict()

    transferred = []
    skipped = []

    for key in source_state:
        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key].clone()
            transferred.append(key)
        else:
            skipped.append(key)

    target_cnn.load_state_dict(target_state)

    # Freeze early layers
    frozen = []
    if freeze_layers > 0:
        frozen = _freeze_cnn_layers(target_cnn, freeze_layers)

    return {
        "status": "success",
        "transferred": transferred,
        "skipped": skipped,
        "frozen": frozen,
        "n_transferred": len(transferred),
        "n_frozen": len(frozen),
    }


def transfer_dqn_shared(
    source_brain,
    target_brain,
    transfer_value_stream: bool = True,
    reinit_action_head: bool = True,
) -> dict:
    """
    Transfer DQN shared layers between brains.

    The DuelingDQN has a shared feature extractor, a value stream,
    and an advantage stream. This transfers the shared layers and
    optionally the value stream, but reinitializes the advantage
    stream (action head) since action spaces may differ.

    Args:
        source_brain: WholeBrain to copy FROM
        target_brain: WholeBrain to copy TO
        transfer_value_stream: Also transfer the value stream
        reinit_action_head: Reinitialize advantage stream with new init

    Returns:
        Report dict.
    """
    if not TORCH_AVAILABLE:
        return {"status": "skipped", "reason": "PyTorch not available"}

    source_dqn = _get_dqn(source_brain)
    target_dqn = _get_dqn(target_brain)

    if source_dqn is None or target_dqn is None:
        return {"status": "skipped", "reason": "One or both brains lack TorchDQN"}

    source_net = source_dqn.online_net
    target_net = target_dqn.online_net

    transferred = []
    skipped = []

    source_state = source_net.state_dict()
    target_state = target_net.state_dict()

    for key in source_state:
        # Skip advantage stream if reinitializing action head
        if reinit_action_head and "advantage" in key:
            skipped.append(key)
            continue
        # Skip value stream if not transferring
        if not transfer_value_stream and "value" in key:
            skipped.append(key)
            continue

        if key in target_state and source_state[key].shape == target_state[key].shape:
            target_state[key] = source_state[key].clone()
            transferred.append(key)
        else:
            skipped.append(key)

    target_net.load_state_dict(target_state)

    # Also sync target network
    target_dqn.target_net.load_state_dict(target_net.state_dict())

    return {
        "status": "success",
        "transferred": transferred,
        "skipped": skipped,
        "n_transferred": len(transferred),
    }


def transfer_full(source_brain, target_brain, freeze_cnn_layers: int = 2) -> dict:
    """
    Full warm-start: transfer both CNN and DQN shared layers.

    Ideal for game-to-game curriculum transfer where both use CNN.
    """
    cnn_result = transfer_cnn(source_brain, target_brain, freeze_layers=freeze_cnn_layers)
    dqn_result = transfer_dqn_shared(source_brain, target_brain)

    return {
        "cnn": cnn_result,
        "dqn": dqn_result,
    }


# ── Internal Helpers ──────────────────────────────────────────────────

def _get_cnn(brain):
    """Extract CNN encoder from brain."""
    if hasattr(brain, 'sensory') and hasattr(brain.sensory, '_cnn'):
        return brain.sensory._cnn
    if hasattr(brain, 'striatum') and hasattr(brain.striatum, '_torch_dqn'):
        dqn = brain.striatum._torch_dqn
        if dqn is not None and dqn.cnn is not None:
            return dqn.cnn
    return None


def _get_dqn(brain):
    """Extract TorchDQN from brain."""
    if hasattr(brain, 'striatum') and hasattr(brain.striatum, '_torch_dqn'):
        return brain.striatum._torch_dqn
    return None


def _freeze_cnn_layers(cnn, n_layers: int) -> list:
    """Freeze the first N conv layers of a CNN."""
    frozen = []
    conv_count = 0

    for name, param in cnn.named_parameters():
        if "conv" in name.lower() or "bn" in name.lower():
            if conv_count < n_layers * 2:  # weight + bias per layer
                param.requires_grad = False
                frozen.append(name)
                conv_count += 1
        elif conv_count > 0 and conv_count < n_layers * 2:
            # Also freeze batch norm parameters associated with early layers
            param.requires_grad = False
            frozen.append(name)

    return frozen
