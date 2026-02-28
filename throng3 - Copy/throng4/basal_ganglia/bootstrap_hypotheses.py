"""
bootstrap_hypotheses.py — 3-tier hypothesis bootstrapping for novel games.

Tier 1 (always): Universal priors — argmin/argmax/flat, game-agnostic.
Tier 2 (offline): Transfer from ExperimentDB — best performers from past games,
                  adapted to the new action space.
Tier 3 (online):  Tetra-seeded — LLM suggests game-specific hypotheses.
                  Only used if bridge is provided and online.

Usage (simplest):
    hypotheses = bootstrap_hypotheses("breakout", n_actions=18)

Usage (with DB transfer):
    with ExperimentDB() as db:
        hypotheses = bootstrap_hypotheses("breakout", n_actions=18, db=db)

Usage (full, with Tetra):
    hypotheses = bootstrap_hypotheses(
        "breakout", n_actions=18, db=db, bridge=tetra_bridge,
        game_description="Breakout: bounce ball to break bricks, avoid losing ball"
    )
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from throng4.storage.experiment_db import ExperimentDB
    from throng4.llm_policy.openclaw_bridge import OpenClawBridge

from throng4.basal_ganglia.dreamer_engine import Hypothesis


# ─────────────────────────────────────────────────────────────────────────────
# Game similarity — used to weight transferred hypotheses
# ─────────────────────────────────────────────────────────────────────────────

# Abstract game feature tags. Each game gets a set of tags.
# Similarity = Jaccard(tags_A, tags_B).
GAME_TAGS: dict[str, set[str]] = {
    "tetris":    {"spatial", "placement", "gravity", "line_clear", "height",
                  "discrete_action", "board_2d", "sequential"},
    "breakout":  {"spatial", "ball_physics", "destruction", "height",
                  "discrete_action", "board_2d", "sequential"},
    "gridworld": {"spatial", "navigation", "discrete_action", "board_2d",
                  "sequential", "goal_reach"},
    "cartpole":  {"continuous_state", "balance", "discrete_action",
                  "sequential", "physics"},
    "pong":      {"spatial", "ball_physics", "discrete_action", "board_2d",
                  "sequential", "competitive"},
    "snake":     {"spatial", "navigation", "discrete_action", "board_2d",
                  "sequential", "growth", "avoidance"},
    "pacman":    {"spatial", "navigation", "discrete_action", "board_2d",
                  "sequential", "avoidance", "collection"},
}


def game_similarity(game_a: str, game_b: str) -> float:
    """
    Jaccard similarity between two games' feature tag sets.

    Returns 0.0–1.0. Unknown games get an empty tag set (similarity=0).
    """
    tags_a = GAME_TAGS.get(game_a.lower(), set())
    tags_b = GAME_TAGS.get(game_b.lower(), set())
    if not tags_a or not tags_b:
        return 0.0
    intersection = len(tags_a & tags_b)
    union = len(tags_a | tags_b)
    return intersection / union if union > 0 else 0.0


def register_game(game: str, tags: set[str]) -> None:
    """Register a new game's feature tags for similarity matching."""
    GAME_TAGS[game.lower()] = tags


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Universal priors (always available, no DB needed)
# ─────────────────────────────────────────────────────────────────────────────

def _universal_priors(n_actions: int) -> List[Hypothesis]:
    """
    Three game-agnostic priors that work for any discrete-action environment.

    These are the same strategies as minimize_height / maximize_lines /
    build_flat in Tetris, but named abstractly so they transfer cleanly.

      minimize_cost:   argmin of first-quarter state features
                       (good for: height, danger, risk)
      maximize_gain:   argmax of first-quarter state features
                       (good for: reward potential, density, progress)
      minimize_variance: argmin of abs-diff (smoothness / flatness)
                         (good for: stability, consistency)
    """
    q = max(4, n_actions // 4)

    minimize_cost = Hypothesis(
        id=1,
        name="minimize_cost",
        action_selector=lambda s, q=q: int(np.argmin(s[:min(q, s.size)])),
        description="Universal: minimize first-quarter state (cost/risk/height)",
    )

    maximize_gain = Hypothesis(
        id=2,
        name="maximize_gain",
        action_selector=lambda s, q=q: int(np.argmax(s[:min(q, s.size)])),
        description="Universal: maximize first-quarter state (gain/reward/density)",
    )

    def _flat_selector(s, q=q):
        chunk = s[:min(q, s.size)]
        diffs = np.abs(np.diff(chunk))
        return int(np.argmin(diffs)) if diffs.size > 0 else 0

    minimize_variance = Hypothesis(
        id=3,
        name="minimize_variance",
        action_selector=_flat_selector,
        description="Universal: minimize state variance (flatness/stability)",
    )

    return [minimize_cost, maximize_gain, minimize_variance]


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — Transfer from ExperimentDB
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_db_hypothesis(row: dict, new_id: int, n_actions: int,
                         source_game: str, target_game: str,
                         similarity: float) -> Hypothesis:
    """
    Adapt a DB hypothesis row to a new game's action space.

    The strategy (argmin/argmax/etc.) is preserved; only the action space
    size (n_actions) is rebound. The name gets a transfer suffix so you
    can trace provenance.
    """
    q = max(4, n_actions // 4)
    name = row.get("name", f"transferred_{new_id}")
    description = row.get("description", "")

    # Infer strategy from name/description keywords
    name_lower = name.lower()
    desc_lower = description.lower()

    if any(k in name_lower or k in desc_lower
           for k in ("max", "gain", "lines", "density", "reward")):
        sel = lambda s, q=q: int(np.argmax(s[:min(q, s.size)]))
        strategy = "maximize"
    elif any(k in name_lower or k in desc_lower
             for k in ("flat", "variance", "bump", "smooth", "diff")):
        def sel(s, q=q):
            chunk = s[:min(q, s.size)]
            diffs = np.abs(np.diff(chunk))
            return int(np.argmin(diffs)) if diffs.size > 0 else 0
        strategy = "minimize_variance"
    else:
        # Default: minimize (most conservative transfer)
        sel = lambda s, q=q: int(np.argmin(s[:min(q, s.size)]))
        strategy = "minimize"

    clean_name = name.split("_v")[0]  # strip generation suffix
    transferred_name = f"{clean_name}@{source_game[:4]}"

    return Hypothesis(
        id=new_id,
        name=transferred_name,
        action_selector=sel,
        description=(
            f"Transferred from {source_game} (sim={similarity:.2f}, "
            f"win={row.get('win_rate', 0):.0%}): {strategy}"
        ),
    )


def _tier2_from_db(target_game: str, n_actions: int,
                   db: "ExperimentDB",
                   max_transfer: int = 3,
                   min_similarity: float = 0.2) -> List[Hypothesis]:
    """
    Pull best hypotheses from DB, filter by game similarity, adapt to new game.
    """
    rows = db.get_top_hypotheses(
        limit=20,
        min_win_rate=0.2,
        min_evidence=10,
        exclude_game=target_game,
    )

    if not rows:
        return []

    adapted = []
    seen_strategies: set[str] = set()
    next_id = 10  # IDs 1-3 reserved for universal priors

    for row in rows:
        source_game = row.get("game", "")
        sim = game_similarity(target_game, source_game) if source_game else 0.3

        if sim < min_similarity:
            continue

        # Deduplicate by strategy type
        name_lower = row.get("name", "").lower()
        desc_lower = row.get("description", "").lower()
        if any(k in name_lower or k in desc_lower
               for k in ("max", "gain")):
            strategy_key = "maximize"
        elif any(k in name_lower or k in desc_lower
                 for k in ("flat", "variance", "bump")):
            strategy_key = "minimize_variance"
        else:
            strategy_key = "minimize"

        if strategy_key in seen_strategies:
            continue  # Don't add duplicate strategies
        seen_strategies.add(strategy_key)

        adapted.append(_adapt_db_hypothesis(
            row, next_id, n_actions, source_game, target_game, sim
        ))
        next_id += 1

        if len(adapted) >= max_transfer:
            break

    return adapted


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Tetra-seeded (online only)
# ─────────────────────────────────────────────────────────────────────────────

def _tier3_from_tetra(target_game: str, n_actions: int,
                      bridge: "OpenClawBridge",
                      game_description: str = "") -> List[Hypothesis]:
    """
    Ask Tetra to suggest game-specific hypotheses.

    Returns empty list if Tetra is unavailable or returns nothing useful.
    """
    try:
        prompt = (
            f"Game: {target_game}. "
            f"{game_description} "
            f"Action space size: {n_actions}. "
            f"Suggest 2 concrete hypotheses in the format: "
            f"'When [state condition], prefer [action type]'. "
            f"Keep them simple and testable."
        )
        response = bridge.send_observation(
            episode=0,
            observation=prompt,
            context={"game": target_game, "n_actions": n_actions},
        )

        if not response.success or not response.hypotheses:
            return []

        # Convert Tetra's text hypotheses to Hypothesis objects
        # Tetra returns dicts with 'name', 'description', 'action_selector_type'
        tetra_hyps = []
        next_id = 20  # IDs 10-19 reserved for DB transfers
        q = max(4, n_actions // 4)

        for h in response.hypotheses[:2]:
            sel_type = h.get("action_selector_type", "minimize")
            if sel_type == "maximize":
                sel = lambda s, q=q: int(np.argmax(s[:min(q, s.size)]))
            elif sel_type == "minimize_variance":
                def sel(s, q=q):
                    chunk = s[:min(q, s.size)]
                    diffs = np.abs(np.diff(chunk))
                    return int(np.argmin(diffs)) if diffs.size > 0 else 0
            else:
                sel = lambda s, q=q: int(np.argmin(s[:min(q, s.size)]))

            tetra_hyps.append(Hypothesis(
                id=next_id,
                name=h.get("name", f"tetra_{next_id}"),
                action_selector=sel,
                description=f"Tetra-seeded: {h.get('description', '')}",
            ))
            next_id += 1

        return tetra_hyps

    except Exception:
        return []  # Tetra offline — silently fall back to tiers 1+2


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_hypotheses(
    game: str,
    n_actions: int,
    db: Optional["ExperimentDB"] = None,
    bridge: Optional["OpenClawBridge"] = None,
    game_description: str = "",
    max_total: int = 6,
    verbose: bool = True,
) -> List[Hypothesis]:
    """
    Bootstrap a hypothesis pool for any game, with or without prior knowledge.

    Tier 1 (always):  3 universal priors — minimize_cost, maximize_gain,
                      minimize_variance. These work for any discrete-action env.
    Tier 2 (offline): Up to 3 hypotheses transferred from ExperimentDB,
                      filtered by game similarity and adapted to n_actions.
    Tier 3 (online):  Up to 2 Tetra-suggested hypotheses (skipped if offline).

    Args:
        game:             Game identifier (e.g. "breakout", "tetris").
        n_actions:        Size of the action space.
        db:               ExperimentDB instance for tier-2 transfer.
                          If None, only tier-1 priors are returned.
        bridge:           OpenClawBridge for tier-3 Tetra suggestions.
                          If None or offline, tier-3 is skipped silently.
        game_description: Human-readable game description for Tetra.
        max_total:        Cap on total hypotheses returned.
        verbose:          Print bootstrap summary.

    Returns:
        List of Hypothesis objects ready to pass to DreamerEngine.dream().
    """
    hypotheses: List[Hypothesis] = []

    # ── Tier 1: Universal priors ──────────────────────────────────────────
    priors = _universal_priors(n_actions)
    hypotheses.extend(priors)
    if verbose:
        print(f"  [Bootstrap] Tier 1: {len(priors)} universal priors")

    # ── Tier 2: DB transfer ───────────────────────────────────────────────
    if db is not None:
        transferred = _tier2_from_db(
            target_game=game,
            n_actions=n_actions,
            db=db,
            max_transfer=max_total - len(hypotheses),
        )
        hypotheses.extend(transferred)
        if verbose and transferred:
            names = [h.name for h in transferred]
            print(f"  [Bootstrap] Tier 2: {len(transferred)} transferred "
                  f"from DB: {names}")
        elif verbose:
            print(f"  [Bootstrap] Tier 2: no DB matches above similarity threshold")

    # ── Tier 3: Tetra ─────────────────────────────────────────────────────
    if bridge is not None and len(hypotheses) < max_total:
        tetra = _tier3_from_tetra(
            target_game=game,
            n_actions=n_actions,
            bridge=bridge,
            game_description=game_description,
        )
        hypotheses.extend(tetra[:max_total - len(hypotheses)])
        if verbose and tetra:
            print(f"  [Bootstrap] Tier 3: {len(tetra)} Tetra-seeded hypotheses")
        elif verbose:
            print(f"  [Bootstrap] Tier 3: Tetra offline or no suggestions")

    hypotheses = hypotheses[:max_total]

    if verbose:
        print(f"  [Bootstrap] Total: {len(hypotheses)} hypotheses for '{game}' "
              f"(n_actions={n_actions})")

    return hypotheses
