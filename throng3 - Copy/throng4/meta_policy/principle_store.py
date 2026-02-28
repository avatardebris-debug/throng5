"""
throng4/meta_policy/principle_store.py
========================================
Phase 2/3: Persistent store for cross-game meta-learning principles.

Principles are authored by Tetra (via crossgame_inbox.json) or manually
inserted. They represent transferable behavioral rules ranked by confidence
and evidence count.

File: experiments/principles.json
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from throng4.meta_policy.meta_adapter import EnvClass, AdapterParams

_STORE_PATH = Path(__file__).resolve().parents[3] / "experiments" / "principles.json"


@dataclass
class Principle:
    id:          str
    text:        str
    source:      str             # "tetra" | "manual"
    env_class:   dict            # {"stochastic": bool, "sparse": bool}
    params:      dict            # AdapterParams fields
    evidence:    list[str]       # episode IDs or run labels that validated this
    confidence:  float           # 0-1
    created_at:  float = 0.0
    updated_at:  float = 0.0

    def env_class_obj(self) -> EnvClass:
        return EnvClass(**self.env_class)

    def params_obj(self, principle_id: str = "") -> AdapterParams:
        kw = {k: v for k, v in self.params.items()
              if k in AdapterParams.__dataclass_fields__}
        return AdapterParams(**kw, principle_id=principle_id or self.id)


class PrincipleStore:
    """
    Read/write interface to experiments/principles.json.
    Stateless: every call re-reads from disk (small file, infrequent access).
    """

    @staticmethod
    def load() -> list[Principle]:
        if not _STORE_PATH.exists():
            return []
        try:
            raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
            return [Principle(**p) for p in raw.get("principles", [])]
        except Exception:
            return []

    @staticmethod
    def save(principles: list[Principle]) -> None:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STORE_PATH.write_text(
            json.dumps({"principles": [asdict(p) for p in principles]}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def add(principle: Principle) -> None:
        principles = PrincipleStore.load()
        # Upsert by id
        existing = {p.id: i for i, p in enumerate(principles)}
        if principle.id in existing:
            principles[existing[principle.id]] = principle
        else:
            principles.append(principle)
        PrincipleStore.save(principles)

    @staticmethod
    def get_params(env_class: EnvClass) -> Optional[AdapterParams]:
        """
        Return AdapterParams for the highest-confidence matching principle.
        Returns None if no principles match (caller falls back to defaults).
        """
        principles = [
            p for p in PrincipleStore.load()
            if p.env_class_obj() == env_class and p.confidence >= 0.5
        ]
        if not principles:
            return None
        best = max(principles, key=lambda p: p.confidence)
        return best.params_obj()

    @staticmethod
    def all_for_class(env_class: EnvClass) -> list[Principle]:
        return [p for p in PrincipleStore.load() if p.env_class_obj() == env_class]

    @staticmethod
    def ingest_tetra_op(op: dict) -> bool:
        """
        Process a single PRINCIPLE op from crossgame_inbox.json.
        Returns True if successfully ingested.
        """
        if op.get("op") != "PRINCIPLE":
            return False
        try:
            p = Principle(
                id          = op["id"],
                text        = op["text"],
                source      = "tetra",
                env_class   = op["env_class"],
                params      = op["params"],
                evidence    = op.get("evidence", []),
                confidence  = float(op.get("confidence", 0.5)),
                created_at  = time.time(),
                updated_at  = time.time(),
            )
            PrincipleStore.add(p)
            return True
        except Exception as e:
            print(f"[PrincipleStore] Failed to ingest op: {e}")
            return False
