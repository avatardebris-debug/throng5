"""
CompressedStateEncoder — Converts observations to lightweight representations.

For the dreamer to simulate fast, it needs compact state representations:
  - Tetris: board → binary grid (1=block, 0=empty) + piece ID
  - Atari: frame → downsampled binary presence map
  - Generic: observation → quantized low-dimensional vector

The compressed state enables 10x+ faster simulation compared to raw observations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum


class EncodingMode(Enum):
    """How to compress the observation."""
    BINARY_GRID = "binary_grid"      # Tetris-style: 1s and 0s
    DOWNSAMPLED = "downsampled"      # Atari-style: spatial downsampling
    QUANTIZED = "quantized"          # Generic: quantize to fewer levels


@dataclass
class CompressedState:
    """A compressed observation ready for fast simulation."""
    data: np.ndarray              # The compressed representation
    original_shape: Tuple         # Shape of original observation
    encoding_mode: EncodingMode
    metadata: dict = field(default_factory=dict)  # Piece ID, etc.

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def compression_ratio(self) -> float:
        original_size = 1
        for d in self.original_shape:
            original_size *= d
        return original_size / max(self.size, 1)


class CompressedStateEncoder:
    """
    Converts environment observations to compact representations.

    Automatically detects or is configured with the encoding mode.
    Tracks compression statistics for calibration.
    """

    def __init__(self,
                 mode: EncodingMode = EncodingMode.QUANTIZED,
                 grid_shape: Optional[Tuple[int, int]] = None,
                 downsample_shape: Tuple[int, int] = (8, 8),
                 n_quantize_levels: int = 4,
                 threshold: float = 0.5):
        """
        Args:
            mode: Encoding strategy
            grid_shape: For BINARY_GRID mode, the (rows, cols) of the game board
            downsample_shape: For DOWNSAMPLED mode, target spatial dims
            n_quantize_levels: For QUANTIZED mode, number of discrete levels
            threshold: Binarization threshold for BINARY_GRID mode
        """
        self.mode = mode
        self.grid_shape = grid_shape
        self.downsample_shape = downsample_shape
        self.n_quantize_levels = n_quantize_levels
        self.threshold = threshold

        # Calibration stats
        self._encode_count = 0
        self._total_compression_ratio = 0.0
        self._reconstruction_errors = []

    def encode(self, observation: np.ndarray,
               metadata: Optional[dict] = None) -> CompressedState:
        """
        Compress an observation.

        Args:
            observation: Raw observation from environment
            metadata: Optional extra info (piece ID, game state, etc.)

        Returns:
            CompressedState ready for dreamer simulation
        """
        original_shape = observation.shape

        if self.mode == EncodingMode.BINARY_GRID:
            compressed = self._encode_binary_grid(observation)
        elif self.mode == EncodingMode.DOWNSAMPLED:
            compressed = self._encode_downsampled(observation)
        else:
            compressed = self._encode_quantized(observation)

        result = CompressedState(
            data=compressed,
            original_shape=original_shape,
            encoding_mode=self.mode,
            metadata=metadata or {},
        )

        self._encode_count += 1
        self._total_compression_ratio += result.compression_ratio

        return result

    def decode(self, compressed: CompressedState) -> np.ndarray:
        """
        Approximate reconstruction of the original observation.

        Not perfect — this is lossy compression. Used for calibration.
        """
        if compressed.encoding_mode == EncodingMode.BINARY_GRID:
            return self._decode_binary_grid(compressed)
        elif compressed.encoding_mode == EncodingMode.DOWNSAMPLED:
            return self._decode_downsampled(compressed)
        else:
            return self._decode_quantized(compressed)

    def calibrate(self, observations: np.ndarray) -> dict:
        """
        Measure encode→decode reconstruction quality on sample data.

        Args:
            observations: Array of shape (n_samples, *obs_shape)

        Returns:
            Dict with mean_error, max_error, compression_ratio
        """
        errors = []
        for obs in observations:
            compressed = self.encode(obs)
            reconstructed = self.decode(compressed)

            # Reshape if needed
            if reconstructed.shape != obs.shape:
                reconstructed = reconstructed.flatten()[:obs.size]
                obs_flat = obs.flatten()[:reconstructed.size]
            else:
                obs_flat = obs.flatten()
                reconstructed = reconstructed.flatten()

            error = np.mean(np.abs(obs_flat - reconstructed))
            errors.append(error)

        self._reconstruction_errors = errors

        return {
            'mean_error': float(np.mean(errors)),
            'max_error': float(np.max(errors)),
            'std_error': float(np.std(errors)),
            'avg_compression_ratio': self.avg_compression_ratio,
            'n_samples': len(observations),
        }

    # ── Encoding implementations ──────────────────────────────

    def _encode_binary_grid(self, obs: np.ndarray) -> np.ndarray:
        """Convert to binary grid: values above threshold → 1, else → 0."""
        if self.grid_shape and obs.ndim == 1:
            # Reshape flat observation into grid if shape is known
            grid_size = self.grid_shape[0] * self.grid_shape[1]
            if obs.size >= grid_size:
                grid = obs[:grid_size].reshape(self.grid_shape)
            else:
                grid = obs.reshape(-1)
                return (grid > self.threshold).astype(np.float32)
            return (grid > self.threshold).astype(np.float32)

        # Already 2D or higher
        return (obs > self.threshold).astype(np.float32)

    def _encode_downsampled(self, obs: np.ndarray) -> np.ndarray:
        """Downsample to target spatial dimensions using block averaging."""
        if obs.ndim == 1:
            # 1D: simple decimation
            target_size = self.downsample_shape[0] * self.downsample_shape[1]
            if obs.size <= target_size:
                return obs.copy().astype(np.float32)

            indices = np.linspace(0, obs.size - 1, target_size, dtype=int)
            return obs[indices].astype(np.float32)

        if obs.ndim >= 2:
            h, w = obs.shape[:2]
            th, tw = self.downsample_shape

            # Block averaging
            bh = max(1, h // th)
            bw = max(1, w // tw)

            result = np.zeros(self.downsample_shape, dtype=np.float32)
            for i in range(th):
                for j in range(tw):
                    block = obs[i*bh:min((i+1)*bh, h),
                                j*bw:min((j+1)*bw, w)]
                    if block.size > 0:
                        result[i, j] = np.mean(block)

            return result

        return obs.copy().astype(np.float32)

    def _encode_quantized(self, obs: np.ndarray) -> np.ndarray:
        """Quantize to n_levels discrete values."""
        obs_flat = obs.flatten().astype(np.float64)

        if obs_flat.size == 0:
            return obs_flat.astype(np.float32)

        # Normalize to [0, 1]
        vmin, vmax = obs_flat.min(), obs_flat.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(obs_flat, dtype=np.float32)

        normalized = (obs_flat - vmin) / (vmax - vmin)

        # Quantize
        quantized = np.round(normalized * (self.n_quantize_levels - 1))
        quantized = quantized / (self.n_quantize_levels - 1)

        # Scale back
        result = quantized * (vmax - vmin) + vmin
        return result.astype(np.float32)

    # ── Decoding implementations ──────────────────────────────

    def _decode_binary_grid(self, compressed: CompressedState) -> np.ndarray:
        """Decode binary grid back to approximate observation."""
        return compressed.data.flatten().astype(np.float32)

    def _decode_downsampled(self, compressed: CompressedState) -> np.ndarray:
        """Upsample back to original shape (nearest-neighbor)."""
        original_size = 1
        for d in compressed.original_shape:
            original_size *= d

        if compressed.data.ndim == 1:
            target_size = original_size
            indices = np.linspace(0, compressed.data.size - 1,
                                  target_size, dtype=int)
            return compressed.data[indices]

        # 2D upsampling
        th, tw = compressed.data.shape[:2]
        oh, ow = compressed.original_shape[:2]

        result = np.zeros(compressed.original_shape[:2], dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                si = min(int(i * th / oh), th - 1)
                sj = min(int(j * tw / ow), tw - 1)
                result[i, j] = compressed.data[si, sj]

        return result

    def _decode_quantized(self, compressed: CompressedState) -> np.ndarray:
        """Quantized data is already in original space, just reshape."""
        return compressed.data.reshape(compressed.original_shape)

    # ── Properties ────────────────────────────────────────────

    @property
    def avg_compression_ratio(self) -> float:
        if self._encode_count == 0:
            return 0.0
        return self._total_compression_ratio / self._encode_count

    def summary(self) -> str:
        lines = [
            f"CompressedStateEncoder ({self.mode.value}):",
            f"  Encodes performed: {self._encode_count}",
            f"  Avg compression ratio: {self.avg_compression_ratio:.1f}x",
        ]
        if self._reconstruction_errors:
            lines.append(
                f"  Avg reconstruction error: "
                f"{np.mean(self._reconstruction_errors):.4f}"
            )
        return "\n".join(lines)
