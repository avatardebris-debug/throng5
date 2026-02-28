"""
landmark_graph.py — Directed graph of discovered game states.

Landmarks are coarsely-hashed notable states (rooms, checkpoints, items).
Edges are proven action chains connecting them. Planning = graph search.

The graph builds automatically from:
  - Rehearsal Loop proven chains → edges
  - BottleneckTracker frontier → new landmarks
  - Dead-end detector → pruned landmarks

Usage:
    graph = LandmarkGraph()
    graph.add_landmark(features, label="skull_room")
    graph.add_edge(from_hash, to_hash, chain=[2,3,1,0], confidence=9.0)
    path = graph.plan_route(current_hash, goal_hash)
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Landmark:
    """A notable state in the game world."""
    state_hash: int
    features: Optional[np.ndarray] = None
    label: str = ""
    visits: int = 0
    deaths: int = 0
    reward_seen: float = 0.0
    is_dead_end: bool = False
    is_goal: bool = False
    is_trap: bool = False
    discovered_step: int = 0

    @property
    def danger_score(self) -> float:
        return self.deaths / max(self.visits, 1)


@dataclass
class Edge:
    """A proven route between two landmarks."""
    from_hash: int
    to_hash: int
    actions: List[int]
    confidence: float = 0.09
    success_rate: float = 0.0
    traversals: int = 0
    failures: int = 0
    avg_reward: float = 0.0
    is_irreversible: bool = False

    @property
    def cost(self) -> float:
        """Lower = better route. Factors in reliability and danger."""
        if self.success_rate < 0.01:
            return float("inf")
        base = len(self.actions)  # Prefer shorter chains
        reliability = 1.0 / max(self.success_rate, 0.01)
        return base * reliability


class LandmarkGraph:
    """
    Directed graph of discovered game states connected by action chains.

    Planning = shortest-path search on this graph.
    Automatically built from Rehearsal Loop's proven chains.
    """

    def __init__(self, n_buckets: int = 256):
        self._n_buckets = n_buckets
        self._landmarks: Dict[int, Landmark] = {}
        self._edges: Dict[int, List[Edge]] = defaultdict(list)  # from_hash → edges
        self._reverse_edges: Dict[int, List[Edge]] = defaultdict(list)  # to_hash → edges

        # State hashing (shared normalization)
        self._running_mean = np.zeros(8, dtype=np.float64)
        self._running_var = np.ones(8, dtype=np.float64)
        self._n_seen: int = 0

    def _hash_state(self, features: np.ndarray) -> int:
        """Coarse state hash — matches BottleneckTracker/ActionChainStore."""
        features = np.asarray(features, dtype=np.float32).flatten()
        k = min(8, len(features))
        top_k = features[:k]

        self._n_seen += 1
        alpha = min(0.01, 1.0 / self._n_seen)
        self._running_mean[:k] += alpha * (top_k.astype(np.float64) - self._running_mean[:k])
        diff = top_k.astype(np.float64) - self._running_mean[:k]
        self._running_var[:k] += alpha * (diff ** 2 - self._running_var[:k])

        std = np.sqrt(self._running_var[:k] + 1e-8)
        normalized = (top_k - self._running_mean[:k].astype(np.float32)) / std.astype(np.float32)
        bins = np.clip(((normalized + 3) / 6 * 8).astype(int), 0, 7)
        return hash(bins.tobytes()) % self._n_buckets

    # ── Landmark Management ──────────────────────────────────────────

    def add_landmark(
        self,
        features: np.ndarray,
        label: str = "",
        is_goal: bool = False,
        step: int = 0,
    ) -> int:
        """Add or update a landmark. Returns state hash."""
        h = self._hash_state(features)
        if h not in self._landmarks:
            self._landmarks[h] = Landmark(
                state_hash=h,
                features=np.asarray(features, dtype=np.float32).copy(),
                label=label or f"L{h}",
                is_goal=is_goal,
                discovered_step=step,
            )
        lm = self._landmarks[h]
        lm.visits += 1
        if is_goal:
            lm.is_goal = True
        if label:
            lm.label = label
        return h

    def mark_dead_end(self, state_hash: int) -> None:
        """Mark a landmark as a dead end (unwinnable from here)."""
        if state_hash in self._landmarks:
            self._landmarks[state_hash].is_dead_end = True

    def mark_trap(self, state_hash: int) -> None:
        """Mark a landmark as a trap (reward but leads to dead end)."""
        if state_hash in self._landmarks:
            self._landmarks[state_hash].is_trap = True

    def record_death(self, features: np.ndarray) -> int:
        """Record a death at this landmark."""
        h = self._hash_state(features)
        if h in self._landmarks:
            self._landmarks[h].deaths += 1
        return h

    def get_landmark(self, state_hash: int) -> Optional[Landmark]:
        return self._landmarks.get(state_hash)

    def get_nearest_landmark(self, features: np.ndarray) -> Optional[Landmark]:
        """Find the landmark closest to the given features."""
        h = self._hash_state(features)
        return self._landmarks.get(h)

    # ── Edge Management ──────────────────────────────────────────────

    def add_edge(
        self,
        from_features: np.ndarray,
        to_features: np.ndarray,
        actions: List[int],
        confidence: float = 0.09,
        success_rate: float = 0.0,
        is_irreversible: bool = False,
    ) -> Tuple[int, int]:
        """
        Add a directed edge (route) between two landmarks.

        Creates landmarks if they don't exist.
        Updates existing edge if this route is better.
        """
        from_hash = self.add_landmark(from_features)
        to_hash = self.add_landmark(to_features)

        # Check for existing edge
        for edge in self._edges[from_hash]:
            if edge.to_hash == to_hash:
                # Update if better
                if confidence > edge.confidence:
                    edge.actions = list(actions)
                    edge.confidence = confidence
                    edge.success_rate = success_rate
                    edge.is_irreversible = is_irreversible
                edge.traversals += 1
                return from_hash, to_hash

        # New edge
        edge = Edge(
            from_hash=from_hash,
            to_hash=to_hash,
            actions=list(actions),
            confidence=confidence,
            success_rate=success_rate,
            is_irreversible=is_irreversible,
        )
        self._edges[from_hash].append(edge)
        self._reverse_edges[to_hash].append(edge)
        return from_hash, to_hash

    def get_edges_from(self, state_hash: int) -> List[Edge]:
        """Get all outgoing edges from a landmark."""
        return self._edges.get(state_hash, [])

    def get_edges_to(self, state_hash: int) -> List[Edge]:
        """Get all incoming edges to a landmark (for backward chaining)."""
        return self._reverse_edges.get(state_hash, [])

    # ── Planning (Dijkstra) ──────────────────────────────────────────

    def plan_route(
        self,
        from_features: np.ndarray,
        to_features: np.ndarray,
        avoid_dead_ends: bool = True,
        avoid_traps: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find shortest path from current state to goal state.

        Returns list of plan steps:
            [{"from": hash, "to": hash, "actions": [...], "cost": float}, ...]
        Returns None if no path exists.
        """
        from_hash = self._hash_state(from_features)
        to_hash = self._hash_state(to_features)

        if from_hash not in self._landmarks or to_hash not in self._landmarks:
            return None

        # Dijkstra's algorithm
        dist: Dict[int, float] = {from_hash: 0.0}
        prev: Dict[int, Tuple[int, Edge]] = {}
        visited: Set[int] = set()
        heap = [(0.0, from_hash)]

        while heap:
            cost, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == to_hash:
                break

            for edge in self._edges.get(current, []):
                neighbor = edge.to_hash

                # Skip dead ends and traps
                if avoid_dead_ends and neighbor in self._landmarks:
                    if self._landmarks[neighbor].is_dead_end:
                        continue
                if avoid_traps and neighbor in self._landmarks:
                    if self._landmarks[neighbor].is_trap:
                        continue

                new_cost = cost + edge.cost
                if neighbor not in dist or new_cost < dist[neighbor]:
                    dist[neighbor] = new_cost
                    prev[neighbor] = (current, edge)
                    heapq.heappush(heap, (new_cost, neighbor))

        if to_hash not in prev and from_hash != to_hash:
            return None

        # Reconstruct path
        path = []
        current = to_hash
        while current in prev:
            from_node, edge = prev[current]
            path.append({
                "from": from_node,
                "to": current,
                "actions": edge.actions,
                "cost": edge.cost,
                "confidence": edge.confidence,
                "irreversible": edge.is_irreversible,
            })
            current = from_node

        path.reverse()
        return path

    def get_reachable(self, from_features: np.ndarray) -> List[int]:
        """Get all landmarks reachable from the given state (BFS)."""
        from_hash = self._hash_state(from_features)
        visited = set()
        queue = [from_hash]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in self._edges.get(current, []):
                if edge.to_hash not in visited:
                    queue.append(edge.to_hash)
        return list(visited)

    def get_goals(self) -> List[Landmark]:
        """Get all landmarks marked as goals."""
        return [lm for lm in self._landmarks.values() if lm.is_goal]

    # ── Import from Rehearsal ────────────────────────────────────────

    def import_proven_chains(self, chain_store) -> int:
        """
        Import proven action chains from ActionChainStore as edges.

        Each proven chain becomes an edge from its start state to the
        end state (estimated). Returns number of edges imported.
        """
        imported = 0
        for chain in chain_store._chains.values():
            if chain.tier in ("real", "proven") and chain.actions:
                # We have the start state hash but need end state
                # For now, create a synthetic end-state landmark
                from_hash = chain.state_hash
                # End state ≈ hash of (start_hash + action_count)
                to_hash = (from_hash + len(chain.actions)) % self._n_buckets

                if from_hash not in self._landmarks:
                    self._landmarks[from_hash] = Landmark(
                        state_hash=from_hash, label=f"chain_start_{from_hash}",
                    )
                if to_hash not in self._landmarks:
                    self._landmarks[to_hash] = Landmark(
                        state_hash=to_hash, label=f"chain_end_{to_hash}",
                    )

                self.add_edge(
                    from_features=np.zeros(8),  # Will use hash directly
                    to_features=np.zeros(8),
                    actions=chain.actions,
                    confidence=chain.confidence,
                    success_rate=chain.success_rate,
                )
                imported += 1
        return imported

    # ── Reporting ────────────────────────────────────────────────────

    def report(self) -> Dict[str, Any]:
        total_edges = sum(len(e) for e in self._edges.values())
        dead_ends = sum(1 for lm in self._landmarks.values() if lm.is_dead_end)
        traps = sum(1 for lm in self._landmarks.values() if lm.is_trap)
        goals = sum(1 for lm in self._landmarks.values() if lm.is_goal)
        return {
            "landmarks": len(self._landmarks),
            "edges": total_edges,
            "dead_ends": dead_ends,
            "traps": traps,
            "goals": goals,
        }
