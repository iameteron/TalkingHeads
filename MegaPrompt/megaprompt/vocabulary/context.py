from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping

_EXO_TERMINOLOGY_PATH = (
    Path(__file__).resolve().parents[2] / "exo-planet_prompt" / "world" / "terminology.json"
)


@dataclass(frozen=True)
class Vocabulary:
    name: str
    tiles: Mapping[str, str] = field(default_factory=dict)
    items: Mapping[str, str] = field(default_factory=dict)
    mobs: Mapping[str, str] = field(default_factory=dict)
    vitals: Mapping[str, str] = field(default_factory=dict)
    labels: Mapping[str, str] = field(default_factory=dict)

    def tile(self, key: str) -> str:
        normalized = str(key or "").strip().lower()
        return self.tiles.get(normalized, _title_from_key(normalized))

    def item(self, key: str) -> str:
        normalized = str(key or "").strip().lower()
        return self.items.get(normalized, self.tile(normalized))

    def mob(self, key: str) -> str:
        raw = str(key or "").strip()
        if raw in self.mobs:
            return self.mobs[raw]
        lowered = raw.lower()
        for mob_key, display in self.mobs.items():
            if mob_key.lower() == lowered:
                return display
        return _title_from_key(raw)

    def vital(self, key: str) -> str:
        normalized = str(key or "").strip().lower()
        return self.vitals.get(normalized, normalized.replace("_", " "))

    def text(self, key: str, default: str) -> str:
        return self.labels.get(key, default)


def _title_from_key(key: str) -> str:
    if not key:
        return "Unknown"
    return key.replace("_", " ").title()


def _load_exo_from_json() -> Vocabulary:
    data = json.loads(_EXO_TERMINOLOGY_PATH.read_text(encoding="utf-8"))
    return Vocabulary(
        name="exo_planet",
        tiles=data.get("tiles", {}),
        items=data.get("items", {}),
        mobs=data.get("mobs", {}),
        vitals=data.get("vitals", {}),
        labels=data.get("labels", {}),
    )


_DEFAULT = Vocabulary(name="craftax")
_EXO = _load_exo_from_json() if _EXO_TERMINOLOGY_PATH.is_file() else Vocabulary(name="exo_planet")

_active: ContextVar[Vocabulary] = ContextVar("megaprompt_vocabulary", default=_DEFAULT)


def default_vocabulary() -> Vocabulary:
    return _DEFAULT


def exo_planet_vocabulary() -> Vocabulary:
    return _EXO


def active_vocabulary() -> Vocabulary:
    return _active.get()


@contextmanager
def use_vocabulary(vocabulary: Vocabulary) -> Iterator[None]:
    token = _active.set(vocabulary)
    try:
        yield
    finally:
        _active.reset(token)


def label_tile(key: str) -> str:
    return active_vocabulary().tile(key)


def label_item(key: str) -> str:
    return active_vocabulary().item(key)


def label_mob(key: str) -> str:
    return active_vocabulary().mob(key)


def label(key: str, *, kind: str = "tile") -> str:
    vocab = active_vocabulary()
    if kind == "item":
        return vocab.item(key)
    if kind == "mob":
        return vocab.mob(key)
    if kind == "vital":
        return vocab.vital(key)
    return vocab.tile(key)
