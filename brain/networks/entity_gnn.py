"""
entity_gnn.py — Lightweight Graph Neural Network for game entity reasoning.

Takes an ObjectGraph (entities + relations) and produces:
  - Per-entity embeddings (node features after message passing)
  - Global graph embedding (pooled readout for the full scene)

The GNN enables relational reasoning: "block A is blocking path to key B"
or "enemy C is between player and exit D". This structured representation
makes the causal model, dead-end detector, and subgoal planner dramatically
more effective than raw pixel features.

Architecture:
  1. Entity properties → node features (linear projection)
  2. Relations → edge features
  3. N rounds of message passing:
     - Aggregate neighbor messages (sum)
     - Update node embedding via MLP
  4. Global readout: mean-pool + max-pool → d_global vector

Pure numpy implementation — no PyTorch dependency.

Usage:
    from brain.planning.object_graph import ObjectGraph
    from brain.networks.entity_gnn import EntityGNN

    gnn = EntityGNN(d_entity=16, d_global=32)
    graph = ObjectGraph()
    graph.add_entity("player", properties={"x": 67, "y": 120}, category="agent")
    graph.add_entity("key", properties={"x": 200, "y": 80}, category="item")
    graph.add_relation("player", "near", "key")

    node_emb, global_emb = gnn.forward(graph)
    # node_emb: (N, d_entity)   — per-entity features
    # global_emb: (d_global,)   — whole-scene summary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Relation type encoding ────────────────────────────────────────────────

RELATION_TYPES = [
    "near", "far", "adjacent", "above", "below",
    "left_of", "right_of", "blocks", "requires",
    "contains", "on_top_of", "threatens", "unknown",
]
_REL_TO_IDX = {r: i for i, r in enumerate(RELATION_TYPES)}

# ── Category encoding ────────────────────────────────────────────────────

CATEGORIES = [
    "agent", "enemy", "item", "block", "door", "wall",
    "platform", "hazard", "projectile", "object",
]
_CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """He initialization for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class EntityGNN:
    """
    Message-passing GNN over game entity graph.

    Converts ObjectGraph → fixed-size feature vector for downstream modules.
    Pure numpy — runs fast even without GPU.
    """

    def __init__(
        self,
        d_entity: int = 16,
        d_edge: int = 8,
        d_global: int = 32,
        n_rounds: int = 2,
        d_raw_node: int = None,  # Auto-computed from features
    ):
        self.d_entity = d_entity
        self.d_edge = d_edge
        self.d_global = d_global
        self.n_rounds = n_rounds

        # Raw node features: category one-hot + position (x, y) + property flags
        self._d_raw_node = d_raw_node or (len(CATEGORIES) + 2 + 8)  # ~20
        self._d_raw_edge = len(RELATION_TYPES) + 2  # type one-hot + distance + angle

        # ── Learnable weights ──────────────────────────────────────────
        # Node encoder: raw features → d_entity
        self.w_node_enc = _he_init(self._d_raw_node, d_entity)
        self.b_node_enc = np.zeros(d_entity, dtype=np.float32)

        # Edge encoder: raw edge → d_edge
        self.w_edge_enc = _he_init(self._d_raw_edge, d_edge)
        self.b_edge_enc = np.zeros(d_edge, dtype=np.float32)

        # Message MLP (per round): [source_emb || edge_emb] → d_entity
        self.msg_weights = []
        for _ in range(n_rounds):
            w = _he_init(d_entity + d_edge, d_entity)
            b = np.zeros(d_entity, dtype=np.float32)
            self.msg_weights.append((w, b))

        # Update MLP (per round): [node_emb || aggregated_msg] → d_entity
        self.upd_weights = []
        for _ in range(n_rounds):
            w = _he_init(d_entity * 2, d_entity)
            b = np.zeros(d_entity, dtype=np.float32)
            self.upd_weights.append((w, b))

        # Global readout: [mean_pool || max_pool] → d_global
        self.w_global = _he_init(d_entity * 2, d_global)
        self.b_global = np.zeros(d_global, dtype=np.float32)

        # Stats
        self._forward_count = 0

    def _encode_node(self, entity) -> np.ndarray:
        """Convert entity properties to raw feature vector."""
        features = np.zeros(self._d_raw_node, dtype=np.float32)

        # Category one-hot
        cat_idx = _CAT_TO_IDX.get(entity.category, len(CATEGORIES) - 1)
        if cat_idx < len(CATEGORIES):
            features[cat_idx] = 1.0

        # Position (normalized to [0, 1] assuming 256x256 game space)
        pos = entity.position
        if pos is not None:
            x, y = pos
            features[len(CATEGORIES)] = x / 256.0
            features[len(CATEGORIES) + 1] = y / 256.0

        # Extra property flags (up to 8)
        props = entity.properties
        prop_offset = len(CATEGORIES) + 2
        prop_keys = ["alive", "collected", "opened", "visible", "moving",
                     "dangerous", "pushable", "locked"]
        for i, key in enumerate(prop_keys):
            if i + prop_offset < self._d_raw_node:
                val = props.get(key, 0)
                features[i + prop_offset] = float(val) if val else 0.0

        return features

    def _encode_edge(self, relation, source_entity, target_entity) -> np.ndarray:
        """Convert relation to raw edge feature vector."""
        features = np.zeros(self._d_raw_edge, dtype=np.float32)

        # Relation type one-hot
        rel_idx = _REL_TO_IDX.get(relation.relation_type, len(RELATION_TYPES) - 1)
        if rel_idx < len(RELATION_TYPES):
            features[rel_idx] = 1.0

        # Distance between entities (normalized)
        pos_s = source_entity.position if source_entity else None
        pos_t = target_entity.position if target_entity else None
        if pos_s and pos_t:
            dx = pos_t[0] - pos_s[0]
            dy = pos_t[1] - pos_s[1]
            dist = np.sqrt(dx**2 + dy**2)
            features[len(RELATION_TYPES)] = dist / 256.0
            # Angle (normalized to [-1, 1])
            features[len(RELATION_TYPES) + 1] = np.arctan2(dy, dx) / np.pi

        return features

    def forward(self, graph) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run GNN forward pass on an ObjectGraph.

        Args:
            graph: ObjectGraph instance with entities and relations

        Returns:
            node_embeddings: (N, d_entity) — per-entity features
            global_embedding: (d_global,) — whole-scene summary
        """
        entities = list(graph._entities.values())
        if not entities:
            return (
                np.zeros((0, self.d_entity), dtype=np.float32),
                np.zeros(self.d_global, dtype=np.float32),
            )

        N = len(entities)
        entity_names = [e.name for e in entities]
        name_to_idx = {name: i for i, name in enumerate(entity_names)}

        # ── 1. Encode nodes ────────────────────────────────────────────
        raw_nodes = np.stack([self._encode_node(e) for e in entities])  # (N, d_raw)
        node_emb = _relu(raw_nodes @ self.w_node_enc + self.b_node_enc)  # (N, d_entity)

        # ── 2. Build edge list and encode edges ────────────────────────
        edges = []  # (src_idx, tgt_idx, edge_features)
        for rel in graph._relations:
            src_idx = name_to_idx.get(rel.source)
            tgt_idx = name_to_idx.get(rel.target)
            if src_idx is not None and tgt_idx is not None:
                src_entity = graph._entities.get(rel.source)
                tgt_entity = graph._entities.get(rel.target)
                edge_feat = self._encode_edge(rel, src_entity, tgt_entity)
                edges.append((src_idx, tgt_idx, edge_feat))

        # Encode edges
        if edges:
            edge_feats = np.stack([e[2] for e in edges])  # (E, d_raw_edge)
            edge_emb = _relu(edge_feats @ self.w_edge_enc + self.b_edge_enc)  # (E, d_edge)
        else:
            edge_emb = np.zeros((0, self.d_edge), dtype=np.float32)

        # ── 3. Message passing rounds ──────────────────────────────────
        for r in range(self.n_rounds):
            msg_w, msg_b = self.msg_weights[r]
            upd_w, upd_b = self.upd_weights[r]

            # Compute messages along each edge
            agg = np.zeros((N, self.d_entity), dtype=np.float32)

            for e_idx, (src, tgt, _) in enumerate(edges):
                # Message: [source_embedding || edge_embedding] → d_entity
                if e_idx < len(edge_emb):
                    msg_input = np.concatenate([node_emb[src], edge_emb[e_idx]])
                else:
                    msg_input = np.concatenate([
                        node_emb[src],
                        np.zeros(self.d_edge, dtype=np.float32),
                    ])
                msg = _relu(msg_input @ msg_w + msg_b)
                agg[tgt] += msg  # Sum aggregation

            # Update: [node_embedding || aggregated_messages] → new_embedding
            update_input = np.concatenate([node_emb, agg], axis=1)  # (N, 2*d_entity)
            node_emb = _relu(update_input @ upd_w + upd_b)  # (N, d_entity)

        # ── 4. Global readout ──────────────────────────────────────────
        mean_pool = np.mean(node_emb, axis=0)  # (d_entity,)
        max_pool = np.max(node_emb, axis=0)    # (d_entity,)
        readout = np.concatenate([mean_pool, max_pool])  # (2*d_entity,)
        global_emb = _relu(readout @ self.w_global + self.b_global)  # (d_global,)

        self._forward_count += 1

        return node_emb, global_emb

    def update_weights(
        self,
        node_emb: np.ndarray,
        global_emb: np.ndarray,
        target_signal: np.ndarray,
        lr: float = 0.001,
    ) -> float:
        """
        Simple gradient update on global readout layer.

        For now, only trains the readout; message passing weights
        are learned more slowly through downstream loss.
        """
        # Reconstruct readout input
        if node_emb.shape[0] == 0:
            return 0.0

        mean_pool = np.mean(node_emb, axis=0)
        max_pool = np.max(node_emb, axis=0)
        readout = np.concatenate([mean_pool, max_pool])

        # Simple MSE on global output
        error = global_emb - target_signal.flatten()[:self.d_global]
        loss = float(np.mean(error ** 2))

        # Gradient for global readout
        d_global = error  # d(loss)/d(output) for MSE
        d_readout = d_global @ self.w_global.T * (readout > 0).astype(np.float32)

        self.w_global -= lr * np.outer(readout, d_global)
        self.b_global -= lr * d_global

        return loss

    def report(self) -> Dict[str, Any]:
        return {
            "d_entity": self.d_entity,
            "d_edge": self.d_edge,
            "d_global": self.d_global,
            "n_rounds": self.n_rounds,
            "forward_count": self._forward_count,
            "n_relation_types": len(RELATION_TYPES),
            "n_categories": len(CATEGORIES),
        }
