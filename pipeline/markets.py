"""Load markets.csv and apply the dev/test split.

Column spec is in data/README.md: snapshot_price is a probability from 0 to 1,
outcome is 1 (YES) or 0 (NO). Until Evan's data/markets.csv lands,
fixtures/dev_markets.csv has the same columns and format.

Hard rule: the ``outcome`` column is NEVER returned in the Market object
used for prompting, so it cannot leak into a prompt.
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
SPLITS = ("dev", "test")


@dataclass(frozen=True)
class Market:
    market_id: str
    source: str
    category: str
    question: str
    created: str
    resolved: str
    snapshot_date: str
    snapshot_price: float  # probability, 0 to 1
    volume: float
    # outcome intentionally omitted: must not leak into prompts.

    @property
    def price_points(self) -> int:
        """Price as a whole percentage, the form shown in prompts."""
        return int(round(self.snapshot_price * 100))


def load_markets(csv_path: Path | str) -> list[Market]:
    p = Path(csv_path)
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{p}: missing columns: {sorted(missing)}")
        out = []
        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            price = float(row["snapshot_price"])
            if not 0.0 <= price <= 1.0:
                # Catches a 0-100 file: int(float("72")) used to load fine and
                # int(float("0.72")) used to load as 0, both silently.
                raise ValueError(f"{p} line {line_no}: snapshot_price {price} is not "
                                 "a probability in [0, 1] (see data/README.md)")
            out.append(Market(
                market_id=row["market_id"],
                source=row["source"],
                category=row["category"],
                question=row["question"],
                created=row["created"],
                resolved=row["resolved"],
                snapshot_date=row["snapshot_date"],
                snapshot_price=price,
                volume=float(row["volume"] or 0),
            ))
    return out


def load_split(path: Optional[Path | str], which: str) -> Optional[set[str]]:
    """Market ids assigned to `which` ('dev' or 'test') in split.csv, which has
    columns market_id, split. Returns None if the file does not exist yet.
    Aryan and Evan own the split file; the pipeline only filters on it."""
    if which not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {which!r}")
    if path is None or not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        return {row["market_id"] for row in csv.DictReader(f) if row["split"] == which}


def filter_by_split(markets: Iterable[Market], split: Optional[set[str]]) -> list[Market]:
    if split is None:
        return list(markets)
    return [m for m in markets if m.market_id in split]
