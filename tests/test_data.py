"""Integrity checks on the committed dataset (data/). These guard the frozen data
against hand edits, a regenerated pull, or a stale decoys.csv."""
import csv

import pytest

from common.config import REPO_ROOT, load_config
from pipeline.decoy import make_decoys
from pipeline.markets import load_markets

CFG = load_config()
DATA = CFG["data"]
MARKETS = REPO_ROOT / DATA["markets_csv"]

pytestmark = pytest.mark.skipif(not MARKETS.exists(), reason="data/markets.csv not present")


def _rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def markets():
    return _rows(MARKETS)


def test_ids_unique_and_prefixed_by_source(markets):
    ids = [r["market_id"] for r in markets]
    assert len(ids) == len(set(ids))
    prefix = {"kalshi": "ks_", "polymarket": "pm_"}
    assert all(r["market_id"].startswith(prefix[r["source"]]) for r in markets)


def test_enough_markets(markets):
    assert len(markets) >= DATA["min_markets"]


def test_values_follow_the_spec(markets):
    load_markets(MARKETS)  # pipeline loader accepts it (0-1 prices, required columns)
    lo, hi = DATA["snapshot_price_min"], DATA["snapshot_price_max"]
    assert all(lo <= float(r["snapshot_price"]) <= hi for r in markets)
    assert {r["outcome"] for r in markets} <= {"0", "1"}
    assert all(r["created"] >= DATA["created_after"] for r in markets)


def test_split_covers_exactly_the_markets(markets):
    split = _rows(REPO_ROOT / DATA["split_csv"])
    assert {r["market_id"] for r in split} == {r["market_id"] for r in markets}
    assert {r["split"] for r in split} == {"dev", "test"}
    dev = sum(r["split"] == "dev" for r in split) / len(split)
    assert abs(dev - CFG["experiment"]["dev_fraction"]) < 0.03


def test_same_event_drops_applied(markets):
    dropped = {r["market_id"] for r in _rows(REPO_ROOT / "data" / "same_event_drops.csv")}
    ids = {r["market_id"] for r in markets}
    assert not dropped & ids, "run: python -m scripts.drop_same_event"


def test_decoys_csv_is_current(markets):
    """decoys.csv must equal what the generator produces for today's markets.csv
    and config; if this fails, run: python -m scripts.make_decoys"""
    d = CFG["decoy"]
    expected = {r["market_id"]: r["decoy_price"] for r in make_decoys(
        markets, seed=CFG["experiment"]["decoy_seed"],
        low=d["low_points"], high=d["high_points"], min_gap=d["min_gap_points"])}
    actual = {r["market_id"]: float(r["decoy_price"])
              for r in _rows(REPO_ROOT / d["output_csv"])}
    assert actual == expected
