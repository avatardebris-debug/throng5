"""
spatial_mapper.py — Learned spatial graph of navigable regions.

Replaces hardcoded room_constants.py with a learned spatial map.
Builds a graph of "locations" from observed state transitions
without any game-specific knowledge.

The mapper:
  1. Clusters observed states into discrete "locations"
  2. Builds edges between locations based on observed transitions
  3. Tracks rewards, threats, and visit counts per location
  4. Provides path-planning queries for the Prefrontal Cortex

Usage:
    from brain.environments.spatial_mapper import SpatialMapper

    mapper = SpatialMapper()
    mapper.observe(features, action=3, reward=1.0, next_features=next_f)
    loc = mapper.current_location(features)
    path = mapper.shortest_path(loc, goal_loc)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class Location:
    """A discovered location (cluster of similar states)."""
    loc_id: int
    centroid: np.ndarray             # Mean state features
    n_samples: int = 0               # How many times visited
    total_reward: float = 0.0
    total_threat: float = 0.0
    discovery_step: int = 0
    last_visit_step: int = 0
    is_terminal: bool = False        # Episode ended here


@dataclass
class Edge:
    """A transition between two locations."""
    from_loc: int
    to_loc: int
    action: int
    count: int = 0
    avg_reward: float = 0.0
    success_rate: float = 1.0        # Fraction of times this transition worked


class SpatialMapper:
    """
    Learned spatial map from state transitions.

    Uses online clustering to discover discrete locations,
    and tracks transitions between them to build a navigable graph.
    """

    def __init__(
        self,
        n_features: int = 84,
        max_locations: int = 500,
        merge_threshold: float = 0.5,  # States closer than this merge into one location
        min_samples_to_confirm: int = 3,
    ):
        self.n_features = n_features
        self.max_locations = max_locations
        self.merge_threshold = merge_threshold
        self.min_samples = min_samples_to_confirm

        self._locations: Dict[int, Location] = {}
        self._edges: Dict[Tuple[int, int, int], Edge] = {}  # (from, to, action) -> Edge
        self._next_id = 0
        self._step = 0

        self._prev_loc_id: Optional[int] = None

    def observe(
        self,
        features: np.ndarray,
        action: int = 0,
        reward: float = 0.0,
        next_features: Optional[np.ndarray] = None,
        done: bool = False,
        threat: float = 0.0,
    ) -> int:
        """
        Observe a state transition. Returns the current location ID.

        Updates the spatial graph with:
        - New or updated location for current state
        - Edge from previous location to current (if applicable)
        """
        self._step += 1
        features = np.asarray(features, dtype=np.float32).flatten()[:self.n_features]

        # Find or create location for current state
        loc_id = self._assign_location(features, done)

        # Update location stats
        loc = self._locations[loc_id]
        loc.n_samples += 1
        loc.total_reward += reward
        loc.total_threat += threat
        loc.last_visit_step = self._step

        # Create edge from previous location
        if self._prev_loc_id is not None and self._prev_loc_id != loc_id:
            edge_key = (self._prev_loc_id, loc_id, action)
            if edge_key not in self._edges:
                self._edges[edge_key] = Edge(
                    from_loc=self._prev_loc_id, to_loc=loc_id, action=action
                )
            edge = self._edges[edge_key]
            edge.count += 1
            # Update running average reward
            edge.avg_reward = (
                edge.avg_reward * (edge.count - 1) + reward
            ) / edge.count

        # Handle next_features if provided (for forward-looking edge)
        if next_features is not None and not done:
            next_features = np.asarray(next_features, dtype=np.float32).flatten()[:self.n_features]
            next_loc_id = self._assign_location(next_features, False)
            if next_loc_id != loc_id:
                edge_key = (loc_id, next_loc_id, action)
                if edge_key not in self._edges:
                    self._edges[edge_key] = Edge(
                        from_loc=loc_id, to_loc=next_loc_id, action=action
                    )
                self._edges[edge_key].count += 1

        self._prev_loc_id = loc_id if not done else None
        return loc_id

    def current_location(self, features: np.ndarray) -> int:
        """Get location ID for a given state (without modifying the graph)."""
        features = np.asarray(features, dtype=np.float32).flatten()[:self.n_features]
        return self._find_nearest(features)

    def neighbors(self, loc_id: int) -> List[Tuple[int, int, float]]:
        """Get (neighbor_id, action, avg_reward) tuples for outgoing edges."""
        result = []
        for (from_id, to_id, action), edge in self._edges.items():
            if from_id == loc_id:
                result.append((to_id, action, edge.avg_reward))
        return result

    def shortest_path(self, start: int, goal: int) -> List[Tuple[int, int]]:
        """
        Find shortest path from start to goal location.

        Returns list of (location_id, action) pairs.
        Uses Dijkstra with edge weights inversely proportional
        to success rate and visit count.
        """
        if start == goal:
            return []
        if start not in self._locations or goal not in self._locations:
            return []

        # Build adjacency from edges
        adj: Dict[int, List[Tuple[int, int, float]]] = defaultdict(list)
        for (from_id, to_id, action), edge in self._edges.items():
            weight = 1.0 / (edge.success_rate * max(edge.count, 1) + 0.1)
            adj[from_id].append((to_id, action, weight))

        # Dijkstra
        dist: Dict[int, float] = {start: 0.0}
        prev: Dict[int, Tuple[int, int]] = {}  # loc -> (prev_loc, action)
        heap = [(0.0, start)]
        visited: Set[int] = set()

        while heap:
            d, u = heappop(heap)
            if u in visited:
                continue
            visited.add(u)

            if u == goal:
                break

            for v, action, weight in adj.get(u, []):
                new_dist = d + weight
                if v not in dist or new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = (u, action)
                    heappush(heap, (new_dist, v))

        # Reconstruct path
        if goal not in prev:
            return []

        path = []
        node = goal
        while node in prev:
            p, action = prev[node]
            path.append((node, action))
            node = p
        path.reverse()
        return path

    def high_reward_locations(self, top_n: int = 5) -> List[Tuple[int, float]]:
        """Return locations with highest average reward."""
        scored = []
        for loc_id, loc in self._locations.items():
            if loc.n_samples >= self.min_samples:
                avg_reward = loc.total_reward / loc.n_samples
                scored.append((loc_id, avg_reward))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]

    def unexplored_locations(self, top_n: int = 5) -> List[Tuple[int, int]]:
        """Return least-visited locations (exploration targets)."""
        scored = [(loc_id, loc.n_samples) for loc_id, loc in self._locations.items()]
        scored.sort(key=lambda x: x[1])
        return scored[:top_n]

    def frontier_locations(self, top_n: int = 5) -> List[int]:
        """Return locations with fewer outgoing edges (map boundary)."""
        out_degree: Dict[int, int] = defaultdict(int)
        for (from_id, _, _) in self._edges:
            out_degree[from_id] += 1
        frontier = [
            (loc_id, out_degree.get(loc_id, 0))
            for loc_id in self._locations
            if self._locations[loc_id].n_samples >= self.min_samples
        ]
        frontier.sort(key=lambda x: x[1])
        return [loc_id for loc_id, _ in frontier[:top_n]]

    # ── Internal ──────────────────────────────────────────────────────

    def _assign_location(self, features: np.ndarray, is_terminal: bool) -> int:
        """Find nearest existing location or create a new one."""
        nearest_id = self._find_nearest(features)

        if nearest_id >= 0:
            loc = self._locations[nearest_id]
            dist = np.linalg.norm(features - loc.centroid[:len(features)])
            if dist < self.merge_threshold:
                # Update centroid (running mean)
                alpha = 1.0 / (loc.n_samples + 1)
                loc.centroid[:len(features)] = (
                    (1 - alpha) * loc.centroid[:len(features)] + alpha * features
                )
                if is_terminal:
                    loc.is_terminal = True
                return nearest_id

        # Create new location
        if len(self._locations) >= self.max_locations:
            self._merge_closest()

        loc_id = self._next_id
        self._next_id += 1
        centroid = np.zeros(self.n_features, dtype=np.float32)
        centroid[:len(features)] = features
        self._locations[loc_id] = Location(
            loc_id=loc_id,
            centroid=centroid,
            discovery_step=self._step,
            is_terminal=is_terminal,
        )
        return loc_id

    def _find_nearest(self, features: np.ndarray) -> int:
        """Find the nearest existing location by L2 distance."""
        if not self._locations:
            return -1

        best_id = -1
        best_dist = float("inf")
        for loc_id, loc in self._locations.items():
            dist = np.linalg.norm(features - loc.centroid[:len(features)])
            if dist < best_dist:
                best_dist = dist
                best_id = loc_id
        return best_id

    def _merge_closest(self) -> None:
        """Merge the two closest locations to make room for new ones."""
        if len(self._locations) < 2:
            return

        ids = list(self._locations.keys())
        best_pair = None
        best_dist = float("inf")

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = np.linalg.norm(
                    self._locations[ids[i]].centroid
                    - self._locations[ids[j]].centroid
                )
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (ids[i], ids[j])

        if best_pair is None:
            return

        keep, remove = best_pair
        loc_keep = self._locations[keep]
        loc_remove = self._locations[remove]

        # Merge statistics
        total = loc_keep.n_samples + loc_remove.n_samples
        if total > 0:
            w = loc_keep.n_samples / total
            loc_keep.centroid = w * loc_keep.centroid + (1 - w) * loc_remove.centroid
        loc_keep.n_samples = total
        loc_keep.total_reward += loc_remove.total_reward
        loc_keep.total_threat += loc_remove.total_threat

        # Redirect edges
        new_edges = {}
        for key, edge in self._edges.items():
            from_id, to_id, action = key
            if from_id == remove:
                from_id = keep
            if to_id == remove:
                to_id = keep
            new_key = (from_id, to_id, action)
            if from_id != to_id:  # No self-loops
                if new_key not in new_edges:
                    new_edges[new_key] = Edge(from_loc=from_id, to_loc=to_id, action=action)
                new_edges[new_key].count += edge.count
        self._edges = new_edges

        del self._locations[remove]

    # ── Reporting ─────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "n_locations": len(self._locations),
            "n_edges": len(self._edges),
            "total_observations": self._step,
            "confirmed_locations": sum(
                1 for loc in self._locations.values()
                if loc.n_samples >= self.min_samples
            ),
            "terminal_locations": sum(
                1 for loc in self._locations.values() if loc.is_terminal
            ),
        }
