"""
sparse_regrowth.py — Dynamic Sparse Neurogenesis / Pruning (Phase 5).

Bio-inspired implementation of Nash Equilibrium Neurogenesis from throng2,
adapted for ANN weight matrices:

  Pruning (Nash Equilibrium pruning):
    Synapses with low magnitude AND low gradient signal are dead → zeroed
    "No gradient through you = you are not needed"

  Regrowth (Hebbian neurogenesis):
    Zero-weight positions with high gradient signal regenerate with small
    random values — the network is "discovering" it needs a neuron there
    "Strong signal through a dead pathway = birth a new synapse"

  Nash Equilibrium:
    Converges when no weight wants to change state:
    - Alive weights with gradient stay alive
    - Dead weights with no gradient stay dead
    - Alive weights with no gradient get pruned next cycle
    - Dead weights with gradient get regrown next cycle

This is equivalent to the "start large, trim to best" idea AND the
"grow where needed" idea simultaneously, without needing two training phases.

Operates on named parameter groups ('shared' layers in RainbowDQNNet by default).
Runs every `prune_interval` train steps. Zero overhead at inference.

Usage:
    sr = SparseRegrowth(model=online_net, prune_interval=500)
    # In train loop:
    sr.maybe_step(train_step_count)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SparseRegrowth:
    """
    Dynamic sparse training: prune dead weights, regrow where gradient demands.

    Parameters
    ----------
    model : nn.Module
        The neural network to apply sparse regrowth to.
    target_layers : list of str
        Name substrings to match in model.named_parameters().
        Default: ['shared'] — matches the shared trunk of RainbowDQNNet.
    prune_interval : int
        How many train steps between prune/grow cycles (default: 500).
    prune_mag_thresh : float
        Weight magnitude below this → candidate for pruning (default: 0.005).
    prune_grad_thresh : float
        Gradient magnitude below this → confirmed dead (default: 1e-6).
        Both conditions must hold simultaneously to prune.
    grow_grad_thresh : float
        Gradient magnitude above this at a zeroed position → regrow (default: 0.05).
    grow_std : float
        Std of random init for regrown weights (default: 0.01 — small, Hebbian birth).
    max_sparsity : float
        Hard cap: never prune more than this fraction of weights (default: 0.5).
        Prevents catastrophic collapse.
    min_sparsity : float
        Don't bother running if sparsity already below this (default: 0.0).
    """

    def __init__(
        self,
        model: "nn.Module",
        target_layers: Optional[List[str]] = None,
        prune_interval: int = 500,
        prune_mag_thresh: float = 0.005,
        prune_grad_thresh: float = 1e-6,
        grow_grad_thresh: float = 0.05,
        grow_std: float = 0.01,
        max_sparsity: float = 0.50,
        min_sparsity: float = 0.0,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for SparseRegrowth")

        self.model = model
        self.target_layers = target_layers or ["shared"]
        self.prune_interval = prune_interval
        self.prune_mag_thresh = prune_mag_thresh
        self.prune_grad_thresh = prune_grad_thresh
        self.grow_grad_thresh = grow_grad_thresh
        self.grow_std = grow_std
        self.max_sparsity = max_sparsity
        self.min_sparsity = min_sparsity

        # Stats
        self._total_steps: int = 0
        self._cycle_count: int = 0
        self._n_pruned_total: int = 0
        self._n_grown_total: int = 0
        self._last_sparsity: float = 0.0

        # RigL-style: save gradient snapshot before pruning so next cycle
        # can use it to identify where regrowth is needed.
        # Backprop skips zeroed weights (∂L/∂w = x * ∂L/∂y = 0 when w=0),
        # so we need the pre-prune gradient as a proxy signal.
        self._grad_snapshot: dict = {}  # name -> tensor saved last cycle

    def _matches_target(self, name: str) -> bool:
        """True if the parameter name matches any target layer substring."""
        return any(t in name for t in self.target_layers)

    def _count_total_params(self) -> int:
        """Count all matching parameters."""
        total = 0
        for name, param in self.model.named_parameters():
            if self._matches_target(name) and param.requires_grad:
                total += param.numel()
        return total

    def _current_sparsity(self) -> float:
        """Fraction of matching parameters that are exactly zero."""
        n_zero = 0
        n_total = 0
        for name, param in self.model.named_parameters():
            if self._matches_target(name) and param.requires_grad:
                n_zero += int((param.data == 0.0).sum().item())
                n_total += param.numel()
        if n_total == 0:
            return 0.0
        return n_zero / n_total

    def step(self) -> dict:
        """
        Execute one prune + grow cycle.

        Returns dict with stats from this cycle.
        """
        if not TORCH_AVAILABLE:
            return {}

        n_pruned = 0
        n_grown = 0
        n_total = 0

        # First compute current sparsity to enforce max_sparsity cap
        cur_sparsity = self._current_sparsity()

        for name, param in self.model.named_parameters():
            if not self._matches_target(name) or not param.requires_grad:
                continue
            if param.grad is None:
                continue

            data = param.data
            grad = param.grad.data.abs()
            n_total += param.numel()

            with torch.no_grad():
                # ── Pruning ─────────────────────────────────────────────
                # Dead = small magnitude AND tiny gradient (no signal through it)
                if cur_sparsity < self.max_sparsity:
                    dead_mask = (
                        (data.abs() < self.prune_mag_thresh)
                        & (grad < self.prune_grad_thresh)
                        & (data != 0.0)  # don't count already-zeroed
                    )
                    # Enforce max_sparsity: how many more can we prune?
                    n_currently_zero = int((data == 0.0).sum().item())
                    max_prune = int(self.max_sparsity * n_total) - n_currently_zero
                    if max_prune > 0:
                        n_can_prune = min(int(dead_mask.sum().item()), max_prune)
                        if n_can_prune > 0:
                            # Prune lowest-magnitude among dead candidates
                            candidates = dead_mask.nonzero(as_tuple=False)
                            if len(candidates) > n_can_prune:
                                mags = data.abs()[dead_mask]
                                _, idx = mags.topk(n_can_prune, largest=False)
                                for i in idx:
                                    data[candidates[i][0], candidates[i][1]] = 0.0
                            else:
                                data[dead_mask] = 0.0
                            n_pruned += n_can_prune

                # ── Regrowth (RigL-style) ─────────────────────────────────
                # Use gradient snapshot from PREVIOUS cycle as regrowth signal.
                # Backprop gives zero gradient at zeroed positions (∂L/∂w=x*δ=0
                # because w=0 → no activation flows). The snapshot captures
                # the gradient BEFORE we zeroed it — this is the proxy for
                # "what signal would flow here if the synapse existed?"
                if name in self._grad_snapshot:
                    snap_grad = self._grad_snapshot[name]
                    zero_mask = (data == 0.0)
                    grow_mask = zero_mask & (snap_grad > self.grow_grad_thresh)
                    n_grow = int(grow_mask.sum().item())
                    if n_grow > 0:
                        data[grow_mask] = (
                            torch.randn_like(data[grow_mask]) * self.grow_std
                        )
                        n_grown += n_grow

                # Save current gradient snapshot (used by next cycle for regrowth)
                self._grad_snapshot[name] = grad.clone()

        self._cycle_count += 1
        self._n_pruned_total += n_pruned
        self._n_grown_total += n_grown
        self._last_sparsity = self._current_sparsity()

        return {
            "cycle": self._cycle_count,
            "n_pruned": n_pruned,
            "n_grown": n_grown,
            "sparsity": round(self._last_sparsity, 4),
        }

    def maybe_step(self, train_step: int) -> Optional[dict]:
        """
        Call every training step. Triggers a prune/grow cycle every
        `prune_interval` steps.

        Returns cycle stats dict if a cycle ran, else None.
        """
        self._total_steps += 1
        if self._total_steps % self.prune_interval == 0:
            return self.step()
        return None

    def report(self) -> dict:
        return {
            "total_steps": self._total_steps,
            "cycles": self._cycle_count,
            "n_pruned_total": self._n_pruned_total,
            "n_grown_total": self._n_grown_total,
            "sparsity": round(self._last_sparsity, 4),
            "max_sparsity_cap": self.max_sparsity,
        }


# ── Standalone smoke test ─────────────────────────────────────────────────


if __name__ == "__main__":
    """
    Build a toy 3-layer MLP, run 2000 gradient steps.
    Verify: pruning events happened, regrowth events happened, no NaN in weights.

    Note: in a freshly-initialized net, thresholds must match weight scale.
    RainbowDQNNet in production uses these at ~10k steps where many weights
    have drifted to 0 through Adam's L2 decay + gradient signal drop.
    """
    import sys

    if not TORCH_AVAILABLE:
        print("PyTorch not available — skip")
        sys.exit(0)

    import torch
    import torch.nn as nn

    rng = np.random.default_rng(42)

    # Toy model matching RainbowDQNNet shared structure
    class ToyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(16, 64), nn.LayerNorm(64), nn.ReLU(),
                nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            )
            self.head = nn.Linear(32, 4)

        def forward(self, x):
            return self.head(self.shared(x))

    net = ToyNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sr = SparseRegrowth(
        model=net,
        target_layers=["shared"],
        prune_interval=50,          # more frequent cycles in short test
        prune_mag_thresh=0.05,      # broader: anything < 0.05 is a candidate
        prune_grad_thresh=1e-3,     # realistic: drop if grad < 0.001
        grow_grad_thresh=0.001,     # Adam normalizes grads; raw grads ~0.001-0.01
        max_sparsity=0.40,
    )

    cycles = []
    for step in range(2000):
        x = torch.randn(32, 16)
        y = torch.randint(0, 4, (32,))
        loss = nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        result = sr.maybe_step(step + 1)
        if result:
            cycles.append(result)

    rpt = sr.report()
    print(f"\nSparseRegrowth smoke: cycles={rpt['cycles']}, "
          f"sparsity={rpt['sparsity']:.2%}, "
          f"pruned={rpt['n_pruned_total']}, grown={rpt['n_grown_total']}")

    # Check no NaN in weights
    nan_found = any(
        torch.isnan(p.data).any()
        for p in net.parameters()
    )

    # Pass bar: pruning must have happened (some activity detected),
    # regrowth must have happened (Hebbian cycle working), no NaN.
    # Sparsity may be low if regrowth continuously fills pruned sites.
    pruning_active = rpt["n_pruned_total"] > 0
    regrowth_active = rpt["n_grown_total"] > 0
    nan_ok = not nan_found

    print(f"pruning_active={pruning_active}, regrowth_active={regrowth_active}, nan_ok={nan_ok}")
    if pruning_active and regrowth_active and nan_ok:
        print("PASS")
        sys.exit(0)
    else:
        print(f"FAIL  pruned={rpt['n_pruned_total']}  grown={rpt['n_grown_total']}  nan={nan_found}")
        sys.exit(1)
