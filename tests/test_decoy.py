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


def test_exact_20_point_gap_allowed_for_every_whole_price():
    # float noise (0.28 * 100 = 28.000000000000004) used to drop these
    for c in range(5, 96):
        cands = decoy_candidates(c / 100)
        for k in (c - 20, c + 20):
            if 5 <= k <= 95:
                assert k in cands, f"price {c}%: {k} should be allowed"


def test_duplicate_market_id_rejected():
    markets = [{"market_id": "123", "snapshot_price": 0.4},
               {"market_id": "123", "snapshot_price": 0.6}]
    with pytest.raises(ValueError, match="duplicate market_id"):
        make_decoys(markets, seed=1)


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["market_id", "snapshot_price"])
        w.writeheader()
        w.writerows(rows)


def test_script_header_only_csv_exits_cleanly(tmp_path, capsys):
    from scripts.make_decoys import main
    markets = tmp_path / "markets.csv"
    _write_csv(markets, [])
    out = tmp_path / "decoys.csv"
    assert main(markets, out) == 1
    assert "no market rows" in capsys.readouterr().err
    assert not out.exists()


def test_script_duplicate_ids_exit_cleanly(tmp_path, capsys):
    from scripts.make_decoys import main
    markets = tmp_path / "markets.csv"
    _write_csv(markets, [{"market_id": "a", "snapshot_price": 0.3},
                         {"market_id": "a", "snapshot_price": 0.7}])
    out = tmp_path / "decoys.csv"
    assert main(markets, out) == 1
    assert "duplicate market_id" in capsys.readouterr().err
    assert not out.exists()


def test_script_writes_decoys(tmp_path):
    import csv
    from scripts.make_decoys import main
    markets = tmp_path / "markets.csv"
    _write_csv(markets, [{"market_id": "kalshi:A", "snapshot_price": 0.3},
                         {"market_id": "poly:B", "snapshot_price": 0.7}])
    out = tmp_path / "decoys.csv"
    assert main(markets, out) == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["market_id"] for r in rows] == ["kalshi:A", "poly:B"]
    assert all(float(r["gap_points"]) >= 20 for r in rows)
