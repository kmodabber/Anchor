"""Load markets.csv and apply the dev/test split.

Evan ships ``data/markets.csv`` on Nov 19. Until then the fixture in
``fixtures/dev_markets.csv`` has the same columns and can be loaded the
same way.

Hard rule: the ``outcome`` column is NEVER returned in the Market object
used for prompting. It is read only for completeness checks.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REQUIRED_COLUMNS = {
    "market_id", "source", "category", "question",
    "created", "resolved", "snapshot_date", "snapshot_price",
    "outcome", "volume",
}


@dataclass(frozen=True)
class Market:
    market_id: str
    source: str
    category: str
    question: str
    created: str
    resolved: str
    snapshot_date: str
    snapshot_price: int
    volume: float
    # outcome intentionally omitted — must not leak into prompts.


def load_markets(csv_path: Path | str) -> list[Market]:
    p = Path(csv_path)
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{p}: missing columns: {sorted(missing)}")
        out = []
        for row in reader:
            out.append(Market(
                market_id=row["market_id"],
                source=row["source"],
                category=row["category"],
                question=row["question"],
                created=row["created"],
                resolved=row["resolved"],
                snapshot_date=row["snapshot_date"],
                snapshot_price=int(float(row["snapshot_price"])),
                volume=float(row["volume"] or 0),
            ))
    return out


def load_split(path: Optional[Path | str]) -> Optional[set[str]]:
    """Return a set of market_ids from a split file, or None if the file
    is missing. Aryan + Evan own split files; we just filter on them.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    ids: set[str] = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def filter_by_split(markets: Iterable[Market], split: Optional[set[str]]) -> list[Market]:
    if split is None:
        return list(markets)
    return [m for m in markets if m.market_id in split]
