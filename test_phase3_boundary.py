"""Phase 3 boundary cleanup tests (no PufferLib required)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent

from bootstrap_paths import ensure_throng_paths

ensure_throng_paths()


def test_nes_adapter_create_env_on_class():
    from brain.environments.nes.feature_adapter import NESAdapter

    assert hasattr(NESAdapter, "create_env")
    assert "staticmethod" in str(type(NESAdapter.__dict__["create_env"]))


def test_nes_adapter_no_import_monkeypatch():
    import brain.environments.nes_adapter as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert "NESAdapter.create_env =" not in src


def test_montezuma_torch_smoke():
    from brain.games.montezuma.runner.constants import GAME_ID, N_ACTIONS
    from brain.orchestrator import WholeBrain

    brain = WholeBrain(
        n_features=84,
        n_actions=N_ACTIONS,
        session_name="montezuma_torch_smoke",
        enable_logging=False,
        use_torch=True,
    )
    assert brain.striatum._torch_dqn is not None
    assert GAME_ID == "ALE/MontezumaRevenge-v5"

    rng = np.random.RandomState(7)
    prev_action = 0
    for i in range(20):
        obs = rng.randn(84).astype(np.float32)
        out = brain.step(
            obs,
            prev_action=prev_action,
            reward=0.0,
            done=(i == 19),
        )
        assert 0 <= int(out["action"]) < N_ACTIONS
        prev_action = int(out["action"])

    brain.close()


def test_feature_atari_env_uses_step_action():
    from brain.bridge.puffer_feature_env import FeatureAtariEnv
    from brain.learning.abstract_features import AbstractFeature

    mock_adapter = MagicMock()
    mock_adapter.env.action_space.n = 4
    vec_a = np.arange(84, dtype=np.float32)
    vec_b = np.arange(84, dtype=np.float32) + 1.0

    def _af(action: int) -> AbstractFeature:
        data = vec_a if action == 0 else vec_b
        af = MagicMock()
        af.to_vector.return_value = data.copy()
        return af

    mock_adapter.get_abstract_features.side_effect = _af
    mock_adapter.reset.return_value = np.zeros(128, dtype=np.float32)
    mock_adapter.step.return_value = (np.zeros(128), 1.0, False, {})

    with patch("brain.bridge.puffer_feature_env.AtariAdapter", return_value=mock_adapter):
        env = FeatureAtariEnv(game_id="ALE/Breakout-v5")

    obs0, _ = env.reset(seed=0)
    assert mock_adapter.reset.call_count == 1
    mock_adapter.get_abstract_features.assert_called_with(action=0)
    np.testing.assert_array_equal(obs0, vec_a)

    obs1, _, _, _, _ = env.step(2)
    mock_adapter.step.assert_called_with(2)
    mock_adapter.get_abstract_features.assert_called_with(action=2)
    np.testing.assert_array_equal(obs1, vec_b)
    env.close()


def test_bridge_roundtrip_mock_features():
    from brain.bridge import load_into_brain
    from brain.bridge.checkpoint import save_bridge_checkpoint
    from brain.bridge.puffer_dqn_trainer import PufferDQNTrainer
    from brain.learning.torch_dqn import RainbowDQN, TORCH_AVAILABLE
    from brain.orchestrator import WholeBrain

    if not TORCH_AVAILABLE:
        print("[SKIP] bridge_roundtrip_mock — PyTorch not available")
        return

    ckpt = ROOT / "checkpoints" / "test_phase3_roundtrip.pt"
    dqn = RainbowDQN(n_features=84, n_actions=4, device="cpu", batch_size=8)
    rng = np.random.RandomState(1)
    batch = [
        (rng.randn(84).astype(np.float32), 1, 0.1, rng.randn(84).astype(np.float32), False)
        for _ in range(8)
    ]
    for _ in range(3):
        dqn.train_step(batch=batch)
    save_bridge_checkpoint(dqn, ckpt, game_id="mock")

    brain = WholeBrain(n_features=84, n_actions=4, enable_logging=False, use_torch=True)
    info = load_into_brain(brain, ckpt)
    assert info["total_updates"] >= 3

    rng = np.random.RandomState(0)
    prev = 0
    for i in range(30):
        obs = rng.randn(84).astype(np.float32)
        r = brain.step(obs, prev_action=prev, reward=0.01, done=False)
        prev = int(r["action"])
    brain.close()
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    test_nes_adapter_create_env_on_class()
    test_nes_adapter_no_import_monkeypatch()
    test_montezuma_torch_smoke()
    test_feature_atari_env_uses_step_action()
    test_bridge_roundtrip_mock_features()
    print("ALL PHASE3 BOUNDARY TESTS PASSED")
