"""Apply the plan's "one market per event" rule to the frozen dataset.

data/same_event_drops.csv lists markets that ask about the same underlying
event as another market (an earlier deadline, an overlapping window, a lower
threshold, or a mutually exclusive outcome). Within each group we keep the
highest-volume market, as the plan does for cross-platform duplicates. Linked
markets would make the bootstrap over markets look more certain than it is.

This script removes the listed rows from data/markets.csv and data/split.csv.
Surviving lines are kept byte-for-byte (no CSV re-serialisation), so the git
diff shows only the deleted rows. Running it again after it has been applied
changes nothing.

Run from the repo root:
    python -m scripts.drop_same_event
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from common.config import REPO_ROOT, load_config

DROPS_CSV = REPO_ROOT / "data" / "same_event_drops.csv"


def read_drops(path: Path = DROPS_CSV) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _first_field(line: str) -> str:
    return next(csv.reader([line]))[0]


def drop_rows(path: Path, drop_ids: set[str]) -> int:
    """Remove lines whose first column is in drop_ids. Returns rows removed."""
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    with open(path, encoding="utf-8", newline="") as f:
        if sum(1 for _ in csv.reader(f)) != len(lines):
            # a quoted field spans lines; line-based editing would be unsafe
            raise ValueError(f"{path}: multi-line CSV fields are not supported")
    header, body = lines[0], lines[1:]
    kept = [ln for ln in body if _first_field(ln) not in drop_ids]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header)
        f.writelines(kept)
    return len(body) - len(kept)


def check(drops: list[dict], markets: dict[str, dict]) -> list[str]:
    """Problems with the drop list against the current markets. Empty = OK."""
    problems = []
    dropped = {d["market_id"] for d in drops}
    if len(dropped) != len(drops):
        problems.append("a market_id appears twice in the drop list")
    for d in drops:
        mid, kept = d["market_id"], d["kept_market_id"]
        if kept in dropped:
            problems.append(f"{kept} is listed as kept and as dropped")
        if kept not in markets:
            problems.append(f"kept market {kept} is not in markets.csv")
        if mid in markets and kept in markets and \
                float(markets[kept]["volume"]) < float(markets[mid]["volume"]):
            problems.append(f"{mid} has more volume than the market kept for it ({kept})")
    return problems


def main() -> int:
    cfg = load_config()
    markets_path = REPO_ROOT / cfg["data"]["markets_csv"]
    split_path = REPO_ROOT / cfg["data"]["split_csv"]
    drops = read_drops()
    drop_ids = {d["market_id"] for d in drops}

    with open(markets_path, encoding="utf-8", newline="") as f:
        markets = {r["market_id"]: r for r in csv.DictReader(f)}

    present = drop_ids & markets.keys()
    if not present:
        print(f"Already applied: none of the {len(drop_ids)} listed markets are in "
              f"{markets_path.name}. Nothing to do.")
        return 0
    if present != drop_ids:
        print(f"error: only {len(present)} of {len(drop_ids)} listed markets are present; "
              "the data was partly edited by hand. Restore it from git and re-run.",
              file=sys.stderr)
        return 1

    problems = check(drops, markets)
    if problems:
        for p in problems:
            print(f"error: {p}", file=sys.stderr)
        return 1

    n_markets = drop_rows(markets_path, drop_ids)
    n_split = drop_rows(split_path, drop_ids)
    print(f"Dropped {n_markets} markets from {markets_path.name} and {n_split} rows from "
          f"{split_path.name}; {len(markets) - n_markets} markets remain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
