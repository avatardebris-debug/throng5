"""
sensory_cortex.py — Perception and Feature Extraction Region.

Responsible for:
  - Converting raw environment observations into structured features
  - Environment fingerprinting (action space, obs dimensionality, reward characteristics)
  - Producing the 84-dim abstract feature vector consumed by other regions
  - Broadcasting perceptions to all other brain regions
  - [NEW] Optional CNN encoder for learned visual features from pixels

This wraps the existing abstract_features.py and adapter system.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import numpy as np

from brain.message_bus import MessageBus, Priority
from brain.regions.base_region import BrainRegion

# Optional PyTorch for CNN encoder
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Projection matrix (deterministic fallback for non-CNN mode) ───────

_PROJ_CACHE: Dict[tuple, np.ndarray] = {}

def _get_random_projection(input_dim: int, output_dim: int) -> np.ndarray:
    """Cached random projection matrix for fixed-size downsampling."""
    key = (input_dim, output_dim)
    if key not in _PROJ_CACHE:
        rng = np.random.RandomState(42)
        mat = rng.randn(input_dim, output_dim).astype(np.float32)
        mat /= np.sqrt(input_dim)
        _PROJ_CACHE[key] = mat
    return _PROJ_CACHE[key]


class SensoryCortex(BrainRegion):
    """
    Perception and feature extraction brain region.

    Converts raw observations into a structured 84-dim abstract vector
    and broadcasts to the rest of the brain.

    Modes:
      - No adapter, no CNN:  random projection (fast fallback)
      - Adapter:             adapter.make_features() (game-specific)
      - CNN (use_cnn=True):  learned 3-layer conv -> 84-dim features
      - FFT+CNN (use_fft=True): 2D FFT compression -> CNN (55x less data)

    Fast path: runs every frame, < 1ms (random proj) or ~2ms (CNN).
    """

    def __init__(
        self,
        bus: MessageBus,
        adapter=None,
        n_features: int = 84,
        use_cnn: bool = False,
        use_fft: bool = False,
        fft_top_k: int = 128,
        frame_stack: int = 4,
    ):
        super().__init__("sensory_cortex", bus)
        self._adapter = adapter
        self._n_features = n_features
        self._last_features: Optional[np.ndarray] = None
        self._last_raw_obs = None
        self._env_fingerprint: Dict[str, Any] = {}

        # ── FFT compression ───────────────────────────────────────────
        self._use_fft = use_fft
        self._fft_top_k = fft_top_k  # Keep top K frequency components

        # ── CNN encoder ───────────────────────────────────────────────
        self._use_cnn = use_cnn and TORCH_AVAILABLE
        self._cnn = None
        self._cnn_device = "cpu"
        self._frame_stack = frame_stack
        self._frame_buffer: deque = deque(maxlen=frame_stack)
        self._cnn_initialized = False
        self._cnn_head: Optional[Any] = None

        if self._use_cnn:
            self._init_cnn()

        # ── Entity GNN (graph-based relational encoding) ──────────────
        self._gnn = None
        self._gnn_dim = 32  # Global embedding size
        try:
            from brain.networks.entity_gnn import EntityGNN
            self._gnn = EntityGNN(d_entity=16, d_global=self._gnn_dim)
        except ImportError:
            pass

    def _init_cnn(self) -> None:
        """Initialize the CNN encoder for pixel observations."""
        if not TORCH_AVAILABLE:
            self._use_cnn = False
            return

        self._cnn_device = "cuda" if torch.cuda.is_available() else "cpu"
        # CNN takes frame_stack grayscale channels -> n_features
        self._cnn = nn.Sequential(
            nn.Conv2d(self._frame_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ).to(self._cnn_device)

        # Linear head is lazy-initialized on first frame (unknown spatial dims)
        self._cnn_head = None
        self._cnn_initialized = True

    def _preprocess_frame(self, obs: np.ndarray) -> np.ndarray:
        """Convert raw frame to grayscale, resize, optionally FFT compress."""
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 3 and obs.shape[2] == 3:
            # RGB -> grayscale
            gray = 0.299 * obs[:, :, 0] + 0.587 * obs[:, :, 1] + 0.114 * obs[:, :, 2]
        elif obs.ndim == 3:
            gray = obs[:, :, 0]
        elif obs.ndim == 2:
            gray = obs
        else:
            return obs.flatten()[:84]

        # Resize to 84x84 using simple strided sampling
        h, w = gray.shape
        row_idx = np.linspace(0, h - 1, 84).astype(int)
        col_idx = np.linspace(0, w - 1, 84).astype(int)
        resized = gray[np.ix_(row_idx, col_idx)]

        # Normalize to [0, 1]
        if resized.max() > 1.0:
            resized = resized / 255.0

        # FFT compression: keep only top-K frequency components
        if self._use_fft:
            resized = self._fft_compress(resized)

        return resized

    def _fft_compress(self, frame: np.ndarray) -> np.ndarray:
        """
        Compress frame using 2D FFT.

        Process:
          1. Apply 2D FFT (84x84 -> 84x84 complex coefficients)
          2. Sort coefficients by magnitude
          3. Zero out all but top-K (keep edges, shapes, large structures)
          4. Inverse FFT -> compressed spatial image

        Result: same 84x84 shape but with high-frequency noise removed.
        The CNN sees a cleaner, information-dense image.

        Compression ratio: keeps top_k/7056 = ~1.8% of frequency data.
        """
        # Forward FFT
        fft_2d = np.fft.fft2(frame)
        fft_shifted = np.fft.fftshift(fft_2d)  # Center low frequencies

        # Get magnitudes and find threshold
        magnitudes = np.abs(fft_shifted)
        if magnitudes.size <= self._fft_top_k:
            return frame  # Frame already small enough

        # Keep only top-K coefficients by magnitude
        threshold = np.partition(magnitudes.ravel(), -self._fft_top_k)[-self._fft_top_k]
        mask = magnitudes >= threshold
        fft_filtered = fft_shifted * mask

        # Inverse FFT -> compressed spatial image
        fft_unshifted = np.fft.ifftshift(fft_filtered)
        compressed = np.real(np.fft.ifft2(fft_unshifted))

        # Re-normalize to [0, 1]
        cmin, cmax = compressed.min(), compressed.max()
        if cmax > cmin:
            compressed = (compressed - cmin) / (cmax - cmin)
        else:
            compressed = np.zeros_like(compressed)

        return compressed.astype(np.float32)

    def _fft_features(self, obs: np.ndarray) -> np.ndarray:
        """
        Pure FFT feature extraction (no CNN).

        Converts frame to frequency domain and uses the top-K
        magnitudes directly as the feature vector.
        Fastest path for pixel obs — no learned parameters.
        """
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = 0.299 * obs[:, :, 0] + 0.587 * obs[:, :, 1] + 0.114 * obs[:, :, 2]
        elif obs.ndim == 3:
            gray = obs[:, :, 0]
        elif obs.ndim == 2:
            gray = obs
        else:
            return self._random_projection_encode(obs)

        # Normalize
        if gray.max() > 1.0:
            gray = gray / 255.0

        # 2D FFT
        fft_2d = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_2d)
        magnitudes = np.abs(fft_shifted).ravel()

        # Sort by magnitude, take top n_features
        top_indices = np.argsort(magnitudes)[-self._n_features:]
        features = magnitudes[top_indices]

        # Normalize
        fmax = features.max()
        if fmax > 0:
            features = features / fmax

        return features.astype(np.float32)

    def _cnn_encode(self, obs: np.ndarray) -> np.ndarray:
        """
        Encode pixel observation through CNN (inference path).

        Uses torch.no_grad() for speed. For training, use
        encode_for_training() which preserves gradients.
        """
        frame = self._preprocess_frame(obs)

        # Build frame stack
        if len(self._frame_buffer) == 0:
            for _ in range(self._frame_stack):
                self._frame_buffer.append(frame)
        else:
            self._frame_buffer.append(frame)

        # Stack frames -> (frame_stack, 84, 84)
        stacked = np.stack(list(self._frame_buffer), axis=0)

        with torch.no_grad():
            x = torch.from_numpy(stacked).unsqueeze(0).to(self._cnn_device)
            conv_out = self._cnn(x)  # (1, flat_dim)

            # Lazy-init linear head
            if self._cnn_head is None:
                flat_dim = conv_out.shape[1]
                self._cnn_head = nn.Linear(flat_dim, self._n_features).to(self._cnn_device)

            features = self._cnn_head(conv_out)  # (1, n_features)
            return features.squeeze(0).cpu().numpy()

    def encode_for_training(self, preprocessed_frames: np.ndarray) -> 'torch.Tensor':
        """
        Encode preprocessed frames WITH gradients (training path).

        Args:
            preprocessed_frames: (batch, frame_stack, 84, 84) numpy array

        Returns:
            (batch, n_features) tensor WITH gradient graph attached.
            This allows DQN loss to backprop through the CNN.
        """
        if self._cnn is None:
            raise RuntimeError("CNN not initialized")

        x = torch.from_numpy(preprocessed_frames).float().to(self._cnn_device)
        conv_out = self._cnn(x)  # (batch, flat_dim)

        # Lazy-init head if needed
        if self._cnn_head is None:
            flat_dim = conv_out.shape[1]
            self._cnn_head = nn.Linear(flat_dim, self._n_features).to(self._cnn_device)

        return self._cnn_head(conv_out)  # (batch, n_features) — WITH gradients

    def get_last_preprocessed(self) -> Optional[np.ndarray]:
        """Return last frame stack as (frame_stack, 84, 84) for replay storage."""
        if len(self._frame_buffer) < self._frame_stack:
            return None
        return np.stack(list(self._frame_buffer), axis=0)

    def _random_projection_encode(self, obs: np.ndarray) -> np.ndarray:
        """Fallback: random projection for non-CNN mode."""
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim >= 2:
            # Pixel obs -> flatten and project
            flat = obs.flatten()
            if len(flat) > self._n_features:
                proj = _get_random_projection(len(flat), self._n_features)
                features = flat @ proj
            else:
                features = flat
        else:
            features = obs.flatten()

        # Pad/truncate
        if len(features) < self._n_features:
            features = np.pad(features, (0, self._n_features - len(features)))
        elif len(features) > self._n_features:
            features = features[:self._n_features]

        # Normalize
        std = np.std(features)
        if std > 0:
            features = features / (std * 3)
        return np.clip(features, -5, 5).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────

    def set_adapter(self, adapter) -> None:
        """Set the environment adapter (Atari, Tetris, etc.)."""
        self._adapter = adapter

    def set_fingerprint(self, fingerprint: Dict[str, Any]) -> None:
        """Set the environment fingerprint."""
        self._env_fingerprint = fingerprint

    def reset_frame_buffer(self) -> None:
        """Clear frame buffer on episode reset."""
        self._frame_buffer.clear()

    def get_cnn_parameters(self):
        """Return CNN parameters for optimizer (used by Striatum)."""
        if self._cnn is None:
            return []
        params = list(self._cnn.parameters())
        if self._cnn_head is not None:
            params += list(self._cnn_head.parameters())
        return params

    def freeze_cnn_layers(self, n_layers: int = 2) -> list:
        """
        Freeze first N conv layers for transfer learning.

        Early layers learn universal features (edges, textures) that
        transfer well. Later layers are game-specific.

        Returns list of frozen parameter names.
        """
        if self._cnn is None:
            return []
        frozen = []
        conv_count = 0
        for name, param in self._cnn.named_parameters():
            if "conv" in name.lower():
                if conv_count < n_layers:
                    param.requires_grad = False
                    frozen.append(name)
                conv_count += 1
            elif conv_count > 0 and conv_count <= n_layers:
                # Freeze associated batch norm / bias
                param.requires_grad = False
                frozen.append(name)
        return frozen

    def unfreeze_all(self) -> None:
        """Unfreeze all CNN parameters (for fine-tuning)."""
        if self._cnn is None:
            return
        for param in self._cnn.parameters():
            param.requires_grad = True
        if self._cnn_head is not None:
            for param in self._cnn_head.parameters():
                param.requires_grad = True

    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from observation and broadcast.

        Expected inputs:
            obs: raw observation from environment
            action: action that was taken (for feature construction)
            reward: reward received
            done: episode done flag
        """
        obs = inputs.get("obs")
        action = inputs.get("action", 0)
        reward = inputs.get("reward", 0.0)
        done = inputs.get("done", False)

        features = None

        # Path 1: CNN encoder (learned visual features, optionally FFT-compressed)
        if self._use_cnn and obs is not None and np.asarray(obs).ndim >= 2:
            features = self._cnn_encode(obs)

        # Path 2: Pure FFT features (no CNN, fast frequency-domain encoding)
        elif self._use_fft and obs is not None and np.asarray(obs).ndim >= 2:
            features = self._fft_features(obs)

        # Path 3: Adapter (game-specific features)
        elif self._adapter is not None and obs is not None:
            try:
                features = self._adapter.make_features(action)
            except Exception:
                features = self._last_features

        # Path 4: Random projection fallback
        elif obs is not None:
            features = self._random_projection_encode(obs)

        # Path 4: Use last features
        if features is None:
            features = self._last_features

        # ── GNN enrichment (if ObjectGraph provided) ──────────────────
        object_graph = inputs.get("object_graph")
        gnn_embedding = None
        if self._gnn is not None and object_graph is not None:
            try:
                _, global_emb = self._gnn.forward(object_graph)
                gnn_embedding = global_emb  # (d_global,)
                # Blend GNN embedding into features
                if features is not None:
                    # Replace last gnn_dim dims of features with GNN output
                    gnn_dim = min(len(global_emb), len(features))
                    features = features.copy()
                    features[-gnn_dim:] = global_emb[:gnn_dim]
            except Exception:
                pass

        self._last_features = features
        self._last_raw_obs = obs

        # Reset frame buffer on episode boundary
        if done:
            self.reset_frame_buffer()

        # Build perception packet
        perception = {
            "features": features,
            "obs": obs,
            "reward": reward,
            "done": done,
            "action": action,
            "env_fingerprint": self._env_fingerprint,
            "gnn_embedding": gnn_embedding,
        }

        # Broadcast perception to all regions
        self.broadcast(
            msg_type="perception",
            payload={
                "features": features.tolist() if features is not None else None,
                "reward": reward,
                "done": done,
            },
        )

        return perception

    def learn(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Sensory cortex learns through CNN encoder (if active)."""
        # CNN gradients flow through Striatum's DQN loss
        return {"loss": 0.0}

    def report(self) -> Dict[str, Any]:
        base = super().report()
        cnn_info = {}
        if self._use_cnn and self._cnn is not None:
            n_params = sum(p.numel() for p in self._cnn.parameters())
            if self._cnn_head is not None:
                n_params += sum(p.numel() for p in self._cnn_head.parameters())
            cnn_info = {
                "cnn_params": n_params,
                "cnn_device": self._cnn_device,
                "frame_stack": self._frame_stack,
            }

        # Determine encoder name
        if self._use_cnn and self._use_fft:
            enc = "fft+cnn"
        elif self._use_cnn:
            enc = "cnn"
        elif self._use_fft:
            enc = "fft"
        elif self._adapter:
            enc = "adapter"
        else:
            enc = "random_projection"

        return {
            **base,
            "has_adapter": self._adapter is not None,
            "has_features": self._last_features is not None,
            "feature_dim": len(self._last_features) if self._last_features is not None else 0,
            "encoder": enc,
            "fft_top_k": self._fft_top_k if self._use_fft else None,
            **cnn_info,
            "env_fingerprint": self._env_fingerprint,
        }
