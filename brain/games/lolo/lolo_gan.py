"""
lolo_gan.py — GAN for procedural Lolo puzzle generation.

Pure numpy implementation. Generator outputs 13x11 grids with tile types
and enemy positions. Discriminator scores puzzle quality (appropriately
difficult, not too easy, not unsolvable).

Architecture:
  Generator:  z(32) + tier(8) → 40 → 256 → 512 → 143*9  (13*11 grid, 9 tile channels)
  Discriminator: 143*9 → 512 → 256 → 1 (quality score)

Tile channels (9): EMPTY, ROCK, TREE, HEART, EMERALD, CHEST, EXIT, WATER, ENEMY
Post-processing enforces exactly 1 player, 1 chest, 1 exit, border walls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from brain.games.lolo.lolo_simulator import (
    Enemy, EnemyType, LoloSimulator, Tile,
)


# Tile channel indices for the GAN output
_CHAN = {
    "EMPTY": 0, "ROCK": 1, "TREE": 2, "HEART": 3,
    "EMERALD": 4, "CHEST": 5, "EXIT": 6, "WATER": 7, "ENEMY": 8,
}
N_CHANNELS = 9
GRID_H, GRID_W = 13, 11
GRID_CELLS = GRID_H * GRID_W  # 143


def _relu(x):
    return np.maximum(0, x)

def _relu_grad(x):
    return (x > 0).astype(np.float32)

def _leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def _leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha).astype(np.float32)

def _sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))

def _softmax_2d(logits):
    """Softmax over last axis (channel dimension)."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)

def _gumbel_softmax(logits, temperature=1.0):
    """Differentiable categorical sampling."""
    g = -np.log(-np.log(np.random.uniform(1e-10, 1.0, logits.shape) + 1e-10) + 1e-10)
    y = _softmax_2d((logits + g) / max(temperature, 0.01))
    return y


class LoloGenerator:
    """
    Neural network that generates Lolo puzzle grids.

    Input:  z(32) + tier_onehot(8) = 40
    Output: (143, 9) logits — per-cell distribution over tile types

    Architecture: 40 → 256(ReLU) → 512(ReLU) → 143*9(reshape)
    """

    def __init__(self, z_dim: int = 32, lr: float = 0.0002):
        self.z_dim = z_dim
        self.lr = lr

        # Weights (He initialization)
        self.W1 = np.random.randn(256, 40).astype(np.float32) * np.sqrt(2.0 / 40)
        self.b1 = np.zeros(256, np.float32)
        self.W2 = np.random.randn(512, 256).astype(np.float32) * np.sqrt(2.0 / 256)
        self.b2 = np.zeros(512, np.float32)
        self.W3 = np.random.randn(GRID_CELLS * N_CHANNELS, 512).astype(np.float32) * np.sqrt(2.0 / 512)
        self.b3 = np.zeros(GRID_CELLS * N_CHANNELS, np.float32)

        # Adam state
        self._adam = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                      for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, z: np.ndarray, tier_onehot: np.ndarray, temperature: float = 1.0):
        """Forward pass. Returns (grid_probs, cache for backprop)."""
        x = np.concatenate([z, tier_onehot])  # (40,)

        # Layer 1
        h1_pre = self.W1 @ x + self.b1
        h1 = _leaky_relu(h1_pre)

        # Layer 2
        h2_pre = self.W2 @ h1 + self.b2
        h2 = _leaky_relu(h2_pre)

        # Output layer
        logits_flat = self.W3 @ h2 + self.b3  # (143*9,)
        logits = logits_flat.reshape(GRID_CELLS, N_CHANNELS)

        # Gumbel-softmax for differentiable sampling
        probs = _gumbel_softmax(logits, temperature)

        cache = {"x": x, "h1_pre": h1_pre, "h1": h1,
                 "h2_pre": h2_pre, "h2": h2, "logits": logits}
        return probs, cache

    def backward(self, d_probs: np.ndarray, cache: dict):
        """Backward pass. Returns gradients dict."""
        # d_probs: (143, 9) — gradient from discriminator

        # Output layer
        d_logits_flat = d_probs.reshape(-1)  # (143*9,)
        dW3 = np.outer(d_logits_flat, cache["h2"])
        db3 = d_logits_flat

        # Layer 2
        dh2 = self.W3.T @ d_logits_flat
        dh2 *= _leaky_relu_grad(cache["h2_pre"])
        dW2 = np.outer(dh2, cache["h1"])
        db2 = dh2

        # Layer 1
        dh1 = self.W2.T @ dh2
        dh1 *= _leaky_relu_grad(cache["h1_pre"])
        dW1 = np.outer(dh1, cache["x"])
        db1 = dh1

        return {"W1": dW1, "b1": db1, "W2": dW2,
                "b2": db2, "W3": dW3, "b3": db3}

    def update(self, grads: dict):
        """Adam update."""
        beta1, beta2, eps = 0.5, 0.999, 1e-8
        params = self._params()

        for name, p in params.items():
            adam = self._adam[name]
            adam["t"] += 1
            g = np.clip(grads[name], -1.0, 1.0)

            adam["m"] = beta1 * adam["m"] + (1 - beta1) * g
            adam["v"] = beta2 * adam["v"] + (1 - beta2) * g**2

            m_hat = adam["m"] / (1 - beta1**adam["t"])
            v_hat = adam["v"] / (1 - beta2**adam["t"])

            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def generate_grid(self, tier: int = 1, temperature: float = 0.8) -> np.ndarray:
        """Generate a raw grid (143, 9) probability matrix."""
        z = np.random.randn(self.z_dim).astype(np.float32)
        tier_oh = np.zeros(8, np.float32)
        tier_oh[min(tier - 1, 7)] = 1.0
        probs, _ = self.forward(z, tier_oh, temperature)
        return probs


class LoloDiscriminator:
    """
    Scores puzzle quality: high for appropriately difficult, low for too-easy or unsolvable.

    Input:  (143, 9) one-hot grid flattened = 1287
    Output: scalar score in [0, 1]

    Architecture: 1287 → 512(LeakyReLU) → 256(LeakyReLU) → 1(sigmoid)
    """

    def __init__(self, lr: float = 0.0002):
        self.lr = lr
        inp = GRID_CELLS * N_CHANNELS  # 1287

        self.W1 = np.random.randn(512, inp).astype(np.float32) * np.sqrt(2.0 / inp)
        self.b1 = np.zeros(512, np.float32)
        self.W2 = np.random.randn(256, 512).astype(np.float32) * np.sqrt(2.0 / 512)
        self.b2 = np.zeros(256, np.float32)
        self.W3 = np.random.randn(1, 256).astype(np.float32) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(1, np.float32)

        self._adam = {k: {"m": np.zeros_like(v), "v": np.zeros_like(v), "t": 0}
                      for k, v in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "W3": self.W3, "b3": self.b3}

    def forward(self, grid_probs: np.ndarray):
        """Forward pass. Returns (score, cache)."""
        x = grid_probs.reshape(-1)  # (1287,)

        h1_pre = self.W1 @ x + self.b1
        h1 = _leaky_relu(h1_pre)

        h2_pre = self.W2 @ h1 + self.b2
        h2 = _leaky_relu(h2_pre)

        out_pre = self.W3 @ h2 + self.b3
        score = _sigmoid(out_pre[0])

        cache = {"x": x, "h1_pre": h1_pre, "h1": h1,
                 "h2_pre": h2_pre, "h2": h2, "out_pre": out_pre, "score": score}
        return score, cache

    def backward(self, d_score: float, cache: dict):
        """Backward through discriminator."""
        # Sigmoid gradient
        s = cache["score"]
        d_out = np.array([d_score * s * (1 - s)], np.float32)

        dW3 = np.outer(d_out, cache["h2"])
        db3 = d_out

        dh2 = self.W3.T @ d_out
        dh2 = dh2.flatten() * _leaky_relu_grad(cache["h2_pre"])
        dW2 = np.outer(dh2, cache["h1"])
        db2 = dh2

        dh1 = self.W2.T @ dh2
        dh1 *= _leaky_relu_grad(cache["h1_pre"])
        dW1 = np.outer(dh1, cache["x"])
        db1 = dh1

        # Gradient w.r.t. input (for generator training)
        d_input = self.W1.T @ dh1  # (1287,)
        d_grid = d_input.reshape(GRID_CELLS, N_CHANNELS)

        return {"W1": dW1, "b1": db1, "W2": dW2,
                "b2": db2, "W3": dW3, "b3": db3}, d_grid

    def update(self, grads: dict):
        """Adam update."""
        beta1, beta2, eps = 0.5, 0.999, 1e-8
        params = self._params()
        for name, p in params.items():
            adam = self._adam[name]
            adam["t"] += 1
            g = np.clip(grads[name], -1.0, 1.0)
            adam["m"] = beta1 * adam["m"] + (1 - beta1) * g
            adam["v"] = beta2 * adam["v"] + (1 - beta2) * g**2
            m_hat = adam["m"] / (1 - beta1**adam["t"])
            v_hat = adam["v"] / (1 - beta2**adam["t"])
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)


# ═══════════════════════════════════════════════════════════════════════
# GAN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

# Enemy types available per tier
_TIER_ENEMIES = {
    1: [],
    2: [],
    3: [EnemyType.SNAKEY, EnemyType.LEEPER],
    4: [EnemyType.ROCKY, EnemyType.ALMA],
    5: [EnemyType.ROCKY, EnemyType.ALMA, EnemyType.MEDUSA],
    6: [EnemyType.ROCKY, EnemyType.ALMA, EnemyType.MEDUSA, EnemyType.DON_MEDUSA],
}


class LoloGAN:
    """
    Full GAN orchestrator for puzzle generation.

    Generates puzzles, post-processes into valid LoloSimulator instances,
    and trains Generator/Discriminator from solve results.
    """

    def __init__(self, z_dim: int = 32, lr: float = 0.0002):
        self.G = LoloGenerator(z_dim=z_dim, lr=lr)
        self.D = LoloDiscriminator(lr=lr)
        self.rng = np.random.RandomState(42)

        # Solved puzzle bank — generator learns to imitate these
        self.solved_bank: List[np.ndarray] = []  # (143, 9) one-hot grids
        self._pretrain_steps = 0

        # Training stats
        self._gen_count = 0
        self._d_losses: List[float] = []
        self._g_losses: List[float] = []

    def add_solved(self, grid_probs: np.ndarray) -> None:
        """Add a solved puzzle grid to the imitation bank."""
        self.solved_bank.append(grid_probs.copy())
        if len(self.solved_bank) > 2000:
            self.solved_bank = self.solved_bank[-2000:]

    def pretrain_from_solved(self, epochs: int = 50, batch_size: int = 16) -> Dict[str, float]:
        """
        Supervised pre-training: generator learns to reconstruct solved puzzles.

        For each batch, picks random solved grids as targets and trains the
        generator to output grids that match them (MSE loss). This teaches
        the generator what solvable puzzles look like BEFORE adversarial training.

        Returns dict with training metrics.
        """
        if len(self.solved_bank) < 2:
            return {"pretrain_loss": 0.0, "steps": 0, "bank_size": len(self.solved_bank)}

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            # Sample a random batch of solved puzzles as targets
            indices = self.rng.choice(len(self.solved_bank),
                                      size=min(batch_size, len(self.solved_bank)),
                                      replace=False)

            for idx in indices:
                target = self.solved_bank[idx]  # (143, 9) one-hot
                loss = self._pretrain_step(target)
                total_loss += loss
                steps += 1
                self._pretrain_steps += 1

        avg_loss = total_loss / max(steps, 1)
        return {"pretrain_loss": float(avg_loss), "steps": steps,
                "bank_size": len(self.solved_bank)}

    def generate(self, tier: int = 1, temperature: float = 0.8) -> Optional[LoloSimulator]:
        """
        Generate one puzzle. Returns LoloSimulator or None if postprocessing fails.
        """
        probs = self.G.generate_grid(tier, temperature)
        sim = self._postprocess(probs, tier)
        self._gen_count += 1
        return sim

    def _postprocess(self, probs: np.ndarray, tier: int) -> Optional[LoloSimulator]:
        """
        Convert (143, 9) probability grid → valid LoloSimulator.

        Enforces:
          - Border walls (row 0, row 12, col 0, col 10)
          - Exactly 1 chest, 1 exit
          - At least 2 hearts
          - Player placed on empty cell
          - Enemies placed from ENEMY channel
        """
        # Argmax to get discrete tile per cell
        grid_flat = np.argmax(probs, axis=-1)  # (143,)
        grid = grid_flat.reshape(GRID_H, GRID_W).astype(np.uint8)

        # Map GAN channels → Tile enum
        chan_to_tile = {
            0: Tile.EMPTY, 1: Tile.ROCK, 2: Tile.TREE, 3: Tile.HEART,
            4: Tile.EMERALD, 5: Tile.CHEST, 6: Tile.EXIT, 7: Tile.WATER,
            8: Tile.EMPTY,  # Enemy channel → empty tile (enemies placed separately)
        }
        for r in range(GRID_H):
            for c in range(GRID_W):
                grid[r, c] = chan_to_tile.get(int(grid[r, c]), Tile.EMPTY)

        # ── 1. Enforce border walls ──
        grid[0, :] = Tile.ROCK
        grid[GRID_H - 1, :] = Tile.ROCK
        grid[:, 0] = Tile.ROCK
        grid[:, GRID_W - 1] = Tile.ROCK

        # ── 2. Collect interior empty cells ──
        interior = []
        for r in range(1, GRID_H - 1):
            for c in range(1, GRID_W - 1):
                if grid[r, c] == Tile.EMPTY:
                    interior.append((r, c))
        self.rng.shuffle(interior)

        if len(interior) < 4:
            return None

        # ── 3. Ensure exactly 1 chest ──
        chests = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r, c] == Tile.CHEST]
        if len(chests) == 0:
            r, c = interior.pop()
            grid[r, c] = Tile.CHEST
        elif len(chests) > 1:
            for cr, cc in chests[1:]:
                grid[cr, cc] = Tile.EMPTY

        # ── 4. Ensure exactly 1 exit ──
        exits = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r, c] == Tile.EXIT]
        if len(exits) == 0:
            if interior:
                r, c = interior.pop()
                grid[r, c] = Tile.EXIT
            else:
                return None
        elif len(exits) > 1:
            for er, ec in exits[1:]:
                grid[er, ec] = Tile.EMPTY

        # ── 5. Ensure >= 2 hearts ──
        hearts = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if grid[r, c] == Tile.HEART]
        while len(hearts) < 2 and interior:
            r, c = interior.pop()
            grid[r, c] = Tile.HEART
            hearts.append((r, c))

        if len(hearts) < 2:
            return None

        # ── 6. Place player on empty cell ──
        player_placed = False
        # Refresh interior empties
        interior = [(r, c) for r in range(1, GRID_H - 1) for c in range(1, GRID_W - 1)
                     if grid[r, c] == Tile.EMPTY]
        self.rng.shuffle(interior)
        if interior:
            pr, pc = interior.pop()
            grid[pr, pc] = Tile.PLAYER
            player_placed = True
        if not player_placed:
            return None

        # ── 7. Place enemies from ENEMY channel ──
        enemies: List[Enemy] = []
        enemy_types = _TIER_ENEMIES.get(tier, [])
        if enemy_types:
            # Find cells where GAN wanted enemies (from original probs)
            enemy_probs = probs[:, _CHAN["ENEMY"]].reshape(GRID_H, GRID_W)
            # Get top-k enemy positions from interior empties
            interior = [(r, c) for r in range(1, GRID_H - 1) for c in range(1, GRID_W - 1)
                         if grid[r, c] == Tile.EMPTY]
            if interior:
                scores = [enemy_probs[r, c] for r, c in interior]
                n_enemies = min(len(interior), max(1, int(np.sum(np.array(scores) > 0.3))))
                n_enemies = min(n_enemies, 4)  # Cap at 4
                top_indices = np.argsort(scores)[-n_enemies:]
                for idx in top_indices:
                    r, c = interior[idx]
                    etype = self.rng.choice(enemy_types)
                    enemies.append(Enemy(etype=etype, row=r, col=c))

        # ── 8. Build simulator ──
        magic_hearts = set()
        if tier >= 4:
            for hr, hc in hearts:
                if self.rng.random() < 0.5:
                    magic_hearts.add((hr, hc))

        try:
            sim = LoloSimulator(grid, enemies, magic_hearts)
            return sim
        except Exception:
            return None

    def train_step(
        self,
        good_puzzles: List[np.ndarray],
        bad_puzzles: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Balanced GAN training step.

        Balance mechanisms:
          1. Loss-ratio gating: skip D training if D is already too strong
          2. 2:1 G:D step ratio: G gets twice the updates (harder task)
          3. Running average tracks balance health
        """
        d_loss = 0.0
        g_loss = 0.0

        # ── Check balance: should we train D? ──
        # If D is much stronger than G (low D-loss, high G-loss), skip D
        avg_d = float(np.mean(self._d_losses[-20:])) if len(self._d_losses) >= 5 else 999.0
        avg_g = float(np.mean(self._g_losses[-20:])) if len(self._g_losses) >= 5 else 999.0
        train_d = (avg_d >= avg_g * 0.5) or len(self._d_losses) < 10  # Always train early

        # ── Train Discriminator (if not too strong) ──
        if train_d:
            for grid in good_puzzles:
                score, cache = self.D.forward(grid)
                d_real_loss = -np.log(score + 1e-10)
                d_loss += d_real_loss
                d_grads, _ = self.D.backward(1.0 / (score + 1e-10), cache)
                self.D.update(d_grads)

            for grid in bad_puzzles:
                score, cache = self.D.forward(grid)
                d_fake_loss = -np.log(1 - score + 1e-10)
                d_loss += d_fake_loss
                d_grads, _ = self.D.backward(-1.0 / (1 - score + 1e-10), cache)
                self.D.update(d_grads)
        else:
            # Still compute D loss for tracking, just don't update weights
            for grid in good_puzzles:
                score, _ = self.D.forward(grid)
                d_loss += -np.log(score + 1e-10)
            for grid in bad_puzzles:
                score, _ = self.D.forward(grid)
                d_loss += -np.log(1 - score + 1e-10)

        # ── Train Generator (2:1 ratio — G gets more steps) ──
        n_gen = max(len(good_puzzles) * 2, 4)  # 2× multiplier
        for _ in range(n_gen):
            z = np.random.randn(self.G.z_dim).astype(np.float32)
            tier_oh = np.zeros(8, np.float32)
            tier_oh[0] = 1.0

            probs, g_cache = self.G.forward(z, tier_oh, temperature=0.8)
            score, d_cache = self.D.forward(probs)

            gen_loss = -np.log(score + 1e-10)
            g_loss += gen_loss

            _, d_grid_grad = self.D.backward(1.0 / (score + 1e-10), d_cache)
            g_grads = self.G.backward(d_grid_grad, g_cache)
            self.G.update(g_grads)

        n_total = max(len(good_puzzles) + len(bad_puzzles), 1)
        d_loss /= n_total
        g_loss /= max(n_gen, 1)

        self._d_losses.append(d_loss)
        self._g_losses.append(g_loss)

        return {"d_loss": float(d_loss), "g_loss": float(g_loss),
                "d_trained": train_d, "balance": round(avg_d / max(avg_g, 0.01), 2)}

    def _pretrain_step(self, target: np.ndarray) -> float:
        """
        One supervised pre-training step: generate grid, compute MSE vs target.

        Args:
            target: (143, 9) one-hot grid of a solved puzzle

        Returns: MSE loss value
        """
        z = np.random.randn(self.G.z_dim).astype(np.float32)
        tier_oh = np.zeros(8, np.float32)
        tier_oh[0] = 1.0

        probs, cache = self.G.forward(z, tier_oh, temperature=1.0)

        # MSE loss: ||output - target||^2
        diff = probs - target  # (143, 9)
        loss = float(np.mean(diff ** 2))

        # Gradient: d(MSE)/d(output) = 2 * (output - target) / N
        d_output = 2.0 * diff / diff.size

        # Backprop through generator
        grads = self.G.backward(d_output, cache)
        self.G.update(grads)

        return loss

    def report(self) -> Dict[str, Any]:
        return {
            "generated": self._gen_count,
            "solved_bank": len(self.solved_bank),
            "pretrain_steps": self._pretrain_steps,
            "d_loss_avg": float(np.mean(self._d_losses[-50:])) if self._d_losses else 0,
            "g_loss_avg": float(np.mean(self._g_losses[-50:])) if self._g_losses else 0,
            "g_params": sum(p.size for p in self.G._params().values()),
            "d_params": sum(p.size for p in self.D._params().values()),
        }
