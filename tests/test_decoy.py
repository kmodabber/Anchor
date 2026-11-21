import random

import pytest

from pipeline.decoy import decoy_candidates, make_decoy, make_decoys


def test_gap_and_range_hold_for_many_prices():
    rng = random.Random(0)
    for i in range(5000):
        p = rng.uniform(0.05, 0.95)
        d = make_decoy(f"m{i}", p, seed=3346)
        assert 0.05 <= d <= 0.95
        assert abs(d - p) * 100 >= 20 - 1e-9


@pytest.mark.parametrize("p", [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
def test_edge_prices_have_candidates(p):
    assert decoy_candidates(p)


def test_deterministic_for_same_seed_and_id():
    assert make_decoy("abc", 0.4, seed=1) == make_decoy("abc", 0.4, seed=1)


def test_independent_of_market_order():
    markets = [{"market_id": f"m{i}", "snapshot_price": 0.1 + 0.008 * i} for i in range(100)]
    a = {r["market_id"]: r["decoy_price"] for r in make_decoys(markets, seed=7)}
    b = {r["market_id"]: r["decoy_price"] for r in make_decoys(list(reversed(markets)), seed=7)}
    assert a == b


def test_different_seed_changes_decoys():
    markets = [{"market_id": f"m{i}", "snapshot_price": 0.5} for i in range(50)]
    a = [r["decoy_price"] for r in make_decoys(markets, seed=1)]
    b = [r["decoy_price"] for r in make_decoys(markets, seed=2)]
    assert a != b


def test_rejects_percent_input():
    with pytest.raises(ValueError):
        make_decoy("x", 62, seed=1)  # price must be 0-1, not 0-100
