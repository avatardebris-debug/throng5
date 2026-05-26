"""
Option-Critic integrated context for Striatum.

Builds elite + present + dream context vectors, manages OC warmup,
and feeds transitions into OptionCritic.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from brain.learning.option_critic import OptionCritic

if TYPE_CHECKING:
    from brain.regions.striatum import Striatum

_log = logging.getLogger(__name__)

_OC_CTX_EXPECTED = (AttributeError, TypeError, ValueError, RuntimeError, ImportError)


def _oc_ctx_strict() -> bool:
    return os.environ.get("THRONG_STRICT_OC_CTX", "").lower() in ("1", "true", "yes")


def _oc_ctx_part(part: str, build, fallback):
    """Build one OC ctx segment; degrade gracefully unless strict debug is enabled."""
    try:
        return build()
    except _OC_CTX_EXPECTED as exc:
        _log.debug("oc ctx %s: %s", part, exc)
        return fallback()
    except Exception:
        if _oc_ctx_strict():
            raise
        _log.warning("oc ctx %s failed unexpectedly", part, exc_info=True)
        return fallback()


class OptionCriticContext:
    """OC state, context building, and observe/update wiring for Striatum."""

    def __init__(self, owner: "Striatum") -> None:
        self._owner = owner
        self._option_critic: Optional[OptionCritic] = None
        self._oc_last_state: Optional[np.ndarray] = None
        self._oc_last_option: Optional[int] = None
        self._oc_last_action: Optional[int] = None

        self._ctx_elite_buf = None
        self._ctx_dreamer = None
        self._ctx_cache: Optional[np.ndarray] = None
        self._ctx_step: int = 0
        self._oc_ctx_interval: int = 4
        self._ctx_embed_dim: int = 8
        self._ctx_dream_dim: int = 8

    @property
    def option_critic(self) -> Optional[OptionCritic]:
        return self._option_critic

    def enable(self, n_options: int = 4) -> None:
        """
        Activate Option-Critic. Auto-scales min_updates from recent episode length.
        """
        owner = self._owner
        try:
            avg_steps = 200
            if hasattr(owner, "_episode_step_counts") and len(owner._episode_step_counts) > 0:
                avg_steps = int(np.mean(list(owner._episode_step_counts)[-20:]))

            warmup = max(100, 3 * avg_steps)

            self._option_critic = OptionCritic(
                n_options=n_options,
                n_actions=owner.n_actions,
                n_features=owner.n_features,
                gamma=owner._gamma,
                min_updates=warmup,
            )
        except Exception:
            self._option_critic = None

    def set_context_sources(
        self,
        elite_buf,
        dreamer,
        elite_embed_dim: int = 8,
        dream_dim: int = 8,
    ) -> None:
        """Wire past+future context sources for integrated prediction input."""
        owner = self._owner
        self._ctx_elite_buf = elite_buf
        self._ctx_dreamer = dreamer
        self._ctx_embed_dim = elite_embed_dim
        self._ctx_dream_dim = dream_dim

        if self._option_critic is not None:
            ctx_dim = elite_embed_dim + owner.n_features + dream_dim
            try:
                self._option_critic.set_context_mode(ctx_dim)
            except Exception:
                pass

    def oc_input(self, features: np.ndarray) -> np.ndarray:
        """Unified Option-Critic state vector for train and inference."""
        features_arr = np.asarray(features, dtype=np.float32)
        if self._ctx_elite_buf is not None or self._ctx_dreamer is not None:
            return self.build_ctx(features_arr)
        return features_arr

    def maybe_update_warmup(self) -> None:
        """Refresh OC min_updates from recent mean episode length."""
        owner = self._owner
        if self._option_critic is None or len(owner._episode_step_counts) == 0:
            return
        avg_steps = int(np.mean(list(owner._episode_step_counts)[-20:]))
        warmup = max(100, 3 * avg_steps)
        self._option_critic.set_min_updates(warmup)

    def build_ctx(self, features: np.ndarray) -> np.ndarray:
        """
        Build integrated context: [elite_embedding | features | dream_features].
        """
        owner = self._owner
        self._ctx_step += 1

        if self._ctx_cache is not None and self._ctx_step % self._oc_ctx_interval != 0:
            ed = self._ctx_embed_dim
            self._ctx_cache[ed: ed + owner.n_features] = features
            return self._ctx_cache

        parts = []
        ed = self._ctx_embed_dim

        def _elite_build():
            if self._ctx_elite_buf is not None and hasattr(
                self._ctx_elite_buf, "summary_embedding",
            ):
                return self._ctx_elite_buf.summary_embedding(ed).astype(np.float32)
            return np.zeros(ed, dtype=np.float32)

        parts.append(_oc_ctx_part("elite", _elite_build, lambda: np.zeros(ed, dtype=np.float32)))
        parts.append(features.astype(np.float32))

        dd = self._ctx_dream_dim

        def _dream_build():
            if self._ctx_dreamer is not None and hasattr(self._ctx_dreamer, "dream_latent"):
                dream = np.asarray(self._ctx_dreamer.dream_latent(features), dtype=np.float32)
                if len(dream) >= dd:
                    return dream[:dd]
                return np.pad(dream, (0, dd - len(dream)))
            return np.zeros(dd, dtype=np.float32)

        parts.append(_oc_ctx_part("dreamer", _dream_build, lambda: np.zeros(dd, dtype=np.float32)))

        ctx = np.concatenate(parts, axis=0)
        self._ctx_cache = ctx
        return ctx

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Feed a transition into OptionCritic. NO-OP if OC not enabled."""
        if self._option_critic is None:
            return
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        oc_state = self.oc_input(state_arr)
        oc_next = self.oc_input(next_state_arr)
        option = self._oc_last_option
        if option is None:
            option = self._option_critic.select_option(oc_state, explore=True)
            self._oc_last_option = option
        try:
            self._option_critic.update(
                oc_state, option, action, reward, oc_next, done,
            )
            self._option_critic.should_terminate(next_state_arr, option)
        except Exception as exc:
            _log.debug("oc_observe failed: %s", exc)

    def reset_cache(self) -> None:
        """Clear integrated context cache (tests / parity checks)."""
        self._ctx_cache = None
        self._ctx_step = 0

    def reset_episode(self) -> None:
        self._oc_last_state = None
        self._oc_last_option = None
        self._oc_last_action = None
        self.reset_cache()
        if self._option_critic is not None:
            self._option_critic.reset_episode()
        self.maybe_update_warmup()

    def select_action(
        self,
        features_arr: np.ndarray,
        explore: bool,
        forward_fn,
    ) -> Optional[Dict[str, Any]]:
        """
        Option-guided action when OC is ready. Returns None if ε-greedy path should run.
        """
        if self._option_critic is None or not self._option_critic.is_ready or not explore:
            return None

        oc = self._option_critic
        oc_input = self.oc_input(features_arr)
        if oc.current_option is None:
            option = oc.select_option(oc_input, explore=True)
        else:
            option = oc.current_option
        action = oc.intra_option_action(oc_input, option, explore=True)
        self._oc_last_state = features_arr.copy()
        self._oc_last_option = option
        self._oc_last_action = action

        if self._owner._torch_dqn is not None:
            _, q_values = self._owner._torch_dqn.select_action(features_arr, explore=False)
        else:
            q_values = forward_fn(features_arr)

        return {
            "action": action,
            "q_values": q_values,
            "epsilon": 0.0,
            "backend": "option_critic",
            "option": option,
        }

    def record_step_state(
        self,
        features_arr: np.ndarray,
        option: int,
        action: int,
    ) -> None:
        self._oc_last_state = features_arr.copy()
        self._oc_last_option = option
        self._oc_last_action = action
