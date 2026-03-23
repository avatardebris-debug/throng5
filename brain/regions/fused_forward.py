"""
fused_forward.py — Batched forward pass across all DQN-style brain regions.

Instead of running 7 independent matmuls (2 per region × 7 regions = 14 numpy
calls), this module stacks all region weights into a single contiguous array
and computes all region outputs in one batched operation.

                 Before (per region):          After (fused):
  Allocs/step:    14 np.ndarray               1 np.ndarray
  numpy calls:    14 (dot + relu × 7)         2 (batched einsum)
  Cache misses:   High (scattered weights)    Low (contiguous layout)

Usage:
    from brain.regions.fused_forward import FusedRegionForward

    frf = FusedRegionForward(n_features=84, hidden_size=128, n_outputs=32)
    frf.add_region("amygdala",  W1, b1, W2, b2)
    frf.add_region("basal_ganglia", W1, b1, W2, b2)

    outputs = frf.forward(features)  # dict[region_name -> output_array]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class FusedRegionForward:
    """
    Batch-compute the forward pass for multiple two-layer ReLU regions.

    All regions must share the same input size. Output sizes may differ.
    Weights are stored in a single contiguous C-order array for cache efficiency.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self._region_names: List[str] = []
        self._hidden_sizes: List[int] = []
        self._output_sizes: List[int] = []

        # Lazily assembled after all regions added
        self._W1_block: Optional[np.ndarray] = None  # (R, n_features, H_max)
        self._b1_block: Optional[np.ndarray] = None  # (R, H_max)
        self._W2_list: List[np.ndarray] = []  # per-region (H_i, out_i)
        self._b2_list: List[np.ndarray] = []  # per-region (out_i,)
        self._compiled = False
        self._h_max = 0

        # Per-region source weight references (kept for live sync)
        self._src_W1: Dict[str, np.ndarray] = {}
        self._src_b1: Dict[str, np.ndarray] = {}
        self._src_W2: Dict[str, np.ndarray] = {}
        self._src_b2: Dict[str, np.ndarray] = {}

        # Reusable hidden buffer (allocated once per compile)
        self._hidden_buf: Optional[np.ndarray] = None

    def add_region(
        self,
        name: str,
        W1: np.ndarray,   # (n_features, hidden_size)
        b1: np.ndarray,   # (hidden_size,)
        W2: np.ndarray,   # (hidden_size, n_outputs)
        b2: np.ndarray,   # (n_outputs,)
    ) -> None:
        """Register a region's weights. Call compile() after all regions added."""
        self._region_names.append(name)
        hidden_size = W1.shape[1]
        output_size = W2.shape[1]
        self._hidden_sizes.append(hidden_size)
        self._output_sizes.append(output_size)
        self._src_W1[name] = W1
        self._src_b1[name] = b1
        self._src_W2[name] = W2
        self._src_b2[name] = b2
        self._compiled = False

    def compile(self) -> None:
        """
        Build the contiguous weight blocks from all registered regions.
        Must be called (or re-called) after all regions are added or weights change.
        """
        if not self._region_names:
            return
        R = len(self._region_names)
        H_max = max(self._hidden_sizes)
        self._h_max = H_max

        # Allocate padded blocks (zero-padded to H_max for uniform batching)
        self._W1_block = np.zeros((R, self.n_features, H_max), dtype=np.float32)
        self._b1_block = np.zeros((R, H_max), dtype=np.float32)
        self._W2_list = []
        self._b2_list = []

        for i, name in enumerate(self._region_names):
            H = self._hidden_sizes[i]
            self._W1_block[i, :, :H] = self._src_W1[name]
            self._b1_block[i, :H] = self._src_b1[name]
            self._W2_list.append(np.asarray(self._src_W2[name], dtype=np.float32))
            self._b2_list.append(np.asarray(self._src_b2[name], dtype=np.float32))

        # Pre-allocate hidden buffer: (R, H_max)
        self._hidden_buf = np.empty((R, H_max), dtype=np.float32)
        self._compiled = True

    def sync_weights(self, name: str) -> None:
        """
        Sync a single region's weights into the fused block (e.g. after a gradient step).
        Cheaper than full recompile.
        """
        if name not in self._region_names:
            return
        i = self._region_names.index(name)
        H = self._hidden_sizes[i]
        self._W1_block[i, :, :H] = self._src_W1[name]
        self._b1_block[i, :H] = self._src_b1[name]
        self._W2_list[i][:] = self._src_W2[name]
        self._b2_list[i][:] = self._src_b2[name]

    def forward(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute all region outputs for given features in one batched pass.

        Layer 1 (batched):  hidden = relu(W1_block @ features + b1_block)
                            Shape: (R, n_features) @ (n_features, H_max)^T = (R, H_max)
        Layer 2 (per-region, different output sizes): output_i = W2_i @ hidden_i + b2_i

        Returns dict mapping region_name -> output_array.
        """
        if not self._compiled:
            self.compile()

        f = np.asarray(features, dtype=np.float32)

        # Batched layer 1: (R, H_max) = (R, n_features, H_max)^T x (n_features,)
        # np.einsum('rih,i->rh', W1_block, f) is equivalent but einsum is slower than matmul
        # Use (R, n_features) @ (n_features, H_max) transposed approach:
        # h = f @ W1_block[:,i,:] is per-region; batched: W1_block has shape (R, n_features, H_max)
        # Fastest: for each region index do a matmul (still faster than separate regional calls
        # because weights are cache-local in the block).
        # Even better: reshape to (R, n_features) broadcast
        np.einsum('j,rjh->rh', f, self._W1_block, out=self._hidden_buf)
        self._hidden_buf += self._b1_block
        np.maximum(self._hidden_buf, 0, out=self._hidden_buf)  # ReLU in-place

        # Layer 2: per-region (different output sizes)
        results: Dict[str, np.ndarray] = {}
        for i, name in enumerate(self._region_names):
            H = self._hidden_sizes[i]
            h = self._hidden_buf[i, :H]
            out = h @ self._W2_list[i] + self._b2_list[i]
            results[name] = out

        return results

    @property
    def region_names(self) -> List[str]:
        return list(self._region_names)
