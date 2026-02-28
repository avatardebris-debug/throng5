"""
throng4/meta_policy/meta_adapter.py
=====================================
Phase 2: Classifies an environment by type and returns tuned DreamerEngine
parameters. Falls back to defaults if no known principle matches.

EnvClass: (stochastic: bool) x (sparse: bool)
  deterministic + sparse  -> Montezuma, GridWorld mazes, Sokoban
  stochastic   + sparse   -> FrozenLake, taxi, partial-obs mazes
  deterministic + dense   -> CartPole, Acrobot, LunarLander
  stochastic   + dense    -> Breakout, Space Invaders, Atari shooting games
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EnvClass:
    stochastic: bool
    sparse: bool

    @property
    def label(self) -> str:
        s = "stochastic" if self.stochastic else "deterministic"
        r = "sparse" if self.sparse else "dense"
        return f"{s}_{r}"


@dataclass
class AdapterParams:
    """Parameters emitted by MetaAdapter for a given EnvClass."""
    dream_interval:    int    # steps between dream cycles
    advisory_rate:     float  # fraction of greedy steps that follow dreamer advice
    promote_threshold: int    # option uses before promotion eligibility
    epsilon_decay:     float  # per-episode decay rate
    gamma:             float  # discount factor
    label:             str    = ""
    principle_id:      Optional[str] = None  # which PrincipleStore entry drove this


# ── Default parameter sets per environment class ───────────────────────
#
# Rationale:
#   Deterministic + sparse: agent *should* converge aggressively once it finds
#     a winning path. Frequent dreaming helps plan ahead in sparse-signal
#     environments. High advisory rate bootstraps from dream teacher early.
#
#   Stochastic + sparse: need maximum exploration diversity; wide epsilon decay,
#     moderate dreaming (world model itself is noisier so dream quality lower).
#
#   Deterministic + dense: reward signal is rich, world model calibrates fast
#     so dreams are reliable but less needed; low advisory rate lets RL lead.
#
#   Stochastic + dense: noisy reward + noisy dynamics = dreaming is least
#     reliable; minimal dreaming, rely mostly on replay buffer diversity.

_DEFAULTS: dict[EnvClass, AdapterParams] = {
    EnvClass(stochastic=False, sparse=True): AdapterParams(
        dream_interval    = 5,
        advisory_rate     = 0.40,
        promote_threshold = 20,
        epsilon_decay     = 0.995,
        gamma             = 0.97,
        label             = "deterministic_sparse",
    ),
    EnvClass(stochastic=True, sparse=True): AdapterParams(
        dream_interval    = 8,
        advisory_rate     = 0.30,
        promote_threshold = 30,
        epsilon_decay     = 0.990,
        gamma             = 0.95,
        label             = "stochastic_sparse",
    ),
    EnvClass(stochastic=False, sparse=False): AdapterParams(
        dream_interval    = 30,
        advisory_rate     = 0.05,
        promote_threshold = 10,
        epsilon_decay     = 0.998,
        gamma             = 0.99,
        label             = "deterministic_dense",
    ),
    EnvClass(stochastic=True, sparse=False): AdapterParams(
        dream_interval    = 20,
        advisory_rate     = 0.10,
        promote_threshold = 50,
        epsilon_decay     = 0.992,
        gamma             = 0.97,
        label             = "stochastic_dense",
    ),
}


class MetaAdapter:
    """
    Map environment metadata dicts (from round_robin_runner.GAME_ROSTER)
    or EnvironmentFingerprints to AdapterParams.

    Priority:
      1. PrincipleStore (Tetra-authored principles with evidence)
      2. _DEFAULTS table
    """

    @staticmethod
    def classify_from_metadata(cfg: dict) -> EnvClass:
        """Classify from a GameRoster entry dict."""
        return EnvClass(
            stochastic = bool(cfg.get("stochastic", False)),
            sparse     = cfg.get("reward_type", "dense") == "sparse",
        )

    @staticmethod
    def classify_from_fingerprint(fp) -> EnvClass:
        """
        Classify from an EnvironmentFingerprint.
        fp.reward_variance   > 0.3  => stochastic
        fp.reward_frequency  < 0.05 => sparse
        """
        stochastic = getattr(fp, "reward_variance",  0.0) > 0.3
        sparse     = getattr(fp, "reward_frequency", 1.0) < 0.05
        return EnvClass(stochastic=stochastic, sparse=sparse)

    @staticmethod
    def params_for(cfg_or_fp) -> AdapterParams:
        """
        Main entry point. Accepts either a GameRoster dict or a fingerprint.
        Checks PrincipleStore first, falls back to _DEFAULTS.
        """
        if isinstance(cfg_or_fp, dict):
            env_class = MetaAdapter.classify_from_metadata(cfg_or_fp)
        else:
            env_class = MetaAdapter.classify_from_fingerprint(cfg_or_fp)

        # Try PrincipleStore
        try:
            from throng4.meta_policy.principle_store import PrincipleStore
            params = PrincipleStore.get_params(env_class)
            if params is not None:
                return params
        except ImportError:
            pass

        return _DEFAULTS.get(env_class, _DEFAULTS[EnvClass(stochastic=False, sparse=True)])
