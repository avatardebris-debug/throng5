"""
object_graph.py — Relational graph of game entities.

Represents the game world as a graph of objects with typed relationships:
    [Player] --at--> (67, 120)
    [Key]    --at--> (200, 80)
    [Door]   --requires--> [Key]
    [Dragon] --blocks--> path_to([Key])
    [Rock]   --can_block--> [Dragon]

Built from RAMSemanticMapper's entity groups and the CausalModel's
learned effects. The LLMStrategy module formats this graph into
natural language for Tetra to reason about.

Usage:
    graph = ObjectGraph()
    graph.add_entity("player", {"x": 67, "y": 120, "lives": 3})
    graph.add_entity("key", {"x": 200, "y": 80, "collected": False})
    graph.add_relation("door", "requires", "key")
    description = graph.describe()
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Entity:
    """A game object with properties."""
    name: str
    category: str = "object"          # player, item, enemy, obstacle, goal
    properties: Dict[str, Any] = field(default_factory=dict)
    ram_addresses: List[int] = field(default_factory=list)
    last_seen_step: int = 0
    discovered_step: int = 0

    @property
    def position(self) -> Optional[Tuple[int, int]]:
        x = self.properties.get("x", self.properties.get("pos_x"))
        y = self.properties.get("y", self.properties.get("pos_y"))
        if x is not None and y is not None:
            return (int(x), int(y))
        return None

    def distance_to(self, other: "Entity") -> float:
        p1 = self.position
        p2 = other.position
        if p1 is None or p2 is None:
            return float("inf")
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


@dataclass
class Relation:
    """A typed relationship between two entities."""
    source: str           # Entity name
    relation_type: str    # "at", "requires", "blocks", "near", "can_reach"
    target: str           # Entity name or value
    confidence: float = 1.0
    learned: bool = False  # True if discovered by observation, False if hardcoded
    observations: int = 0


class ObjectGraph:
    """
    Relational graph of game entities and their relationships.

    Provides:
    - Entity management (add, update, remove)
    - Relation tracking (spatial, causal, requirement)
    - Natural language description for LLM consumption
    - Auto-generated spatial relations from positions
    """

    def __init__(self):
        self._entities: Dict[str, Entity] = {}
        self._relations: List[Relation] = []
        self._step: int = 0

    # ── Entity Management ────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        properties: Optional[Dict[str, Any]] = None,
        category: str = "object",
        ram_addresses: Optional[List[int]] = None,
    ) -> Entity:
        """Add or update a game entity."""
        if name in self._entities:
            entity = self._entities[name]
            if properties:
                entity.properties.update(properties)
            entity.last_seen_step = self._step
            return entity

        entity = Entity(
            name=name,
            category=category,
            properties=properties or {},
            ram_addresses=ram_addresses or [],
            discovered_step=self._step,
            last_seen_step=self._step,
        )
        self._entities[name] = entity
        return entity

    def remove_entity(self, name: str) -> None:
        if name in self._entities:
            del self._entities[name]
            self._relations = [
                r for r in self._relations
                if r.source != name and r.target != name
            ]

    def get_entity(self, name: str) -> Optional[Entity]:
        return self._entities.get(name)

    def get_entities_by_category(self, category: str) -> List[Entity]:
        return [e for e in self._entities.values() if e.category == category]

    # ── Relation Management ──────────────────────────────────────────

    def add_relation(
        self,
        source: str,
        relation_type: str,
        target: str,
        confidence: float = 1.0,
        learned: bool = False,
    ) -> None:
        """Add a relationship between entities."""
        # Check for existing relation
        for r in self._relations:
            if r.source == source and r.relation_type == relation_type and r.target == target:
                r.confidence = max(r.confidence, confidence)
                r.observations += 1
                return

        self._relations.append(Relation(
            source=source,
            relation_type=relation_type,
            target=target,
            confidence=confidence,
            learned=learned,
        ))

    def get_relations(self, entity_name: str) -> List[Relation]:
        """Get all relations involving this entity."""
        return [
            r for r in self._relations
            if r.source == entity_name or r.target == entity_name
        ]

    def get_blockers(self, target_entity: str) -> List[str]:
        """What entities are blocking access to the target?"""
        return [
            r.source for r in self._relations
            if r.target == target_entity and r.relation_type == "blocks"
        ]

    def get_requirements(self, entity_name: str) -> List[str]:
        """What does this entity require?"""
        return [
            r.target for r in self._relations
            if r.source == entity_name and r.relation_type == "requires"
        ]

    # ── Auto-update from RAM ─────────────────────────────────────────

    def update_from_ram(
        self,
        ram: Any,
        mapper: Any = None,
    ) -> None:
        """
        Update entity positions from RAM using semantic mapper.

        mapper: RAMSemanticMapper instance
        """
        import numpy as np
        ram = np.asarray(ram, dtype=np.uint8).flatten()
        self._step += 1

        if mapper is None:
            return

        # Update entities from entity groups
        for group in mapper.get_entity_groups():
            name = group["type"]
            props = {}
            for i, addr in enumerate(group["bytes"]):
                if addr < len(ram):
                    if i == 0:
                        props["x"] = int(ram[addr])
                    elif i == 1:
                        props["y"] = int(ram[addr])
                    else:
                        props[f"byte_{i}"] = int(ram[addr])

            self.add_entity(name, props, category="player" if "player" in name else "entity")

        # Update state flags (potential items/inventory)
        for item in mapper.get_subgoal_bytes():
            addr = item["addr"]
            if addr < len(ram):
                name = item["label"]
                self.add_entity(name, {
                    "value": int(ram[addr]),
                    "collected": int(ram[addr]) > 0,
                }, category="item")

    def auto_spatial_relations(self, proximity_threshold: float = 30.0) -> None:
        """Generate spatial relations from entity positions."""
        entities = list(self._entities.values())
        # Clear old spatial relations
        self._relations = [
            r for r in self._relations
            if r.relation_type not in ("near", "far", "same_row", "same_col")
        ]

        for i, e1 in enumerate(entities):
            if e1.position is None:
                continue
            for e2 in entities[i + 1:]:
                if e2.position is None:
                    continue
                dist = e1.distance_to(e2)
                if dist < proximity_threshold:
                    self.add_relation(e1.name, "near", e2.name, learned=True)
                if e1.position[1] == e2.position[1]:
                    self.add_relation(e1.name, "same_row", e2.name, learned=True)
                if e1.position[0] == e2.position[0]:
                    self.add_relation(e1.name, "same_col", e2.name, learned=True)

    # ── Natural Language Description ─────────────────────────────────

    def describe(self) -> str:
        """
        Generate natural language description of the game state.

        Formatted for LLM consumption.
        """
        lines: List[str] = []

        # Entities by category
        for category in ["player", "item", "enemy", "obstacle", "goal", "entity", "object"]:
            entities = self.get_entities_by_category(category)
            if not entities:
                continue
            lines.append(f"## {category.title()}s")
            for e in entities:
                pos_str = f" at ({e.position[0]}, {e.position[1]})" if e.position else ""
                props_str = ", ".join(f"{k}={v}" for k, v in e.properties.items() if k not in ("x", "y", "pos_x", "pos_y"))
                line = f"- {e.name}{pos_str}"
                if props_str:
                    line += f" [{props_str}]"
                lines.append(line)

        # Relations
        if self._relations:
            lines.append("\n## Relationships")
            for r in self._relations:
                if r.relation_type in ("near", "same_row", "same_col"):
                    continue  # Skip spatial noise
                lines.append(f"- {r.source} --{r.relation_type}--> {r.target} (conf={r.confidence:.1f})")

        # Spatial
        spatial = [r for r in self._relations if r.relation_type == "near"]
        if spatial:
            lines.append("\n## Nearby")
            for r in spatial:
                lines.append(f"- {r.source} near {r.target}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage or LLM context."""
        return {
            "entities": {
                name: {
                    "category": e.category,
                    "properties": e.properties,
                    "position": e.position,
                }
                for name, e in self._entities.items()
            },
            "relations": [
                {
                    "source": r.source,
                    "type": r.relation_type,
                    "target": r.target,
                    "confidence": r.confidence,
                }
                for r in self._relations
                if r.relation_type not in ("near", "same_row", "same_col")
            ],
        }

    def report(self) -> Dict[str, Any]:
        categories = defaultdict(int)
        for e in self._entities.values():
            categories[e.category] += 1
        rel_types = defaultdict(int)
        for r in self._relations:
            rel_types[r.relation_type] += 1
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "categories": dict(categories),
            "relation_types": dict(rel_types),
        }
