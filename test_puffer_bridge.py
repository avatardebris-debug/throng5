"""Smoke test: serial pretrain + load into WholeBrain (no PufferLib required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bootstrap_paths import ensure_throng_paths

ensure_throng_paths()

import numpy as np

from brain.bridge import TrainConfig, load_into_brain
from brain.bridge.puffer_dqn_trainer import PufferDQNTrainer
from brain.orchestrator import WholeBrain


def test_feature_env_reset_single_adapter_call():
    """FeatureAtariEnv resets the adapter exactly once (no super().reset double-call)."""
    from unittest.mock import MagicMock, patch

    from brain.bridge.puffer_feature_env import FeatureAtariEnv

    mock_adapter = MagicMock()
    mock_adapter.env.action_space.n = 4
    af = MagicMock()
    af.to_vector.return_value = np.zeros(84, dtype=np.float32)
    mock_adapter.get_abstract_features.return_value = af
    mock_adapter.reset.return_value = np.zeros(128, dtype=np.float32)

    with patch("brain.bridge.puffer_feature_env.AtariAdapter", return_value=mock_adapter):
        env = FeatureAtariEnv(game_id="ALE/Breakout-v5")
    env.reset(seed=42)
    assert mock_adapter.reset.call_count == 1
    env.close()


def test_serial_pretrain_and_load():
    ckpt = ROOT / "checkpoints" / "test_puffer_bridge.pt"
    cfg = TrainConfig(
        game_id="ALE/Breakout-v5",
        total_steps=200,
        num_envs=1,
        use_puffer_vector=False,
        log_interval=100,
        export_path=str(ckpt),
        train_interval=2,
    )
    stats = PufferDQNTrainer(cfg).train()
    assert stats["total_updates"] > 0, "expected gradient updates"
    assert ckpt.is_file(), "checkpoint not written"

    brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False, use_torch=True)
    info = load_into_brain(brain, ckpt)
    assert info["total_updates"] > 0
    brain.close()
    print(f"[PASS] serial pretrain updates={stats['total_updates']} load={info['path']}")


if __name__ == "__main__":
    test_feature_env_reset_single_adapter_call()
    test_serial_pretrain_and_load()
    print("ALL PUFFER BRIDGE TESTS PASSED")
