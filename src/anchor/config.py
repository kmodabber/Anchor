"""Load the shared config.yaml. Kamyar owns the schema; keep this tolerant."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    seed_base: int
    temperature: float
    runs_per_cell: int
    decoy_min_distance: int
    models: list[str]
    paths: dict[str, str]
    ollama: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        return (REPO_ROOT / self.paths[key]).resolve()

    def seed_for(self, market_id: str, condition: str, model: str, run: int) -> int:
        # Deterministic per-call seed. Stable across machines.
        h = abs(hash((market_id, condition, model, run))) % 1_000_000
        return self.seed_base + h


def load_config(path: Path | str | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        seed_base=int(raw.get("seed_base", 0)),
        temperature=float(raw.get("temperature", 0.7)),
        runs_per_cell=int(raw.get("runs_per_cell", 5)),
        decoy_min_distance=int(raw.get("decoy_min_distance", 20)),
        models=list(raw.get("models", [])),
        paths=dict(raw.get("paths", {})),
        ollama=dict(raw.get("ollama", {})),
        raw=raw,
    )
