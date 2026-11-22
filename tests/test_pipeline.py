"""End-to-end checks of the run loop with the fake model and the fixture markets."""
import copy
import json
from collections import defaultdict

import pytest

from common.config import REPO_ROOT, config_hash, load_config
from pipeline.decoy import make_decoys
from pipeline.markets import load_markets
from pipeline.model_client import FakeClient
from pipeline.runner import results_file, run_pipeline

MODEL = "qwen2.5:3b"


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def markets(cfg):
    return load_markets(REPO_ROOT / cfg["data"]["fixture_csv"])


def _run(cfg, markets, out_dir, runs=2, client=None):
    run_pipeline(cfg=cfg, client=client or FakeClient(), markets=markets,
                 models=[MODEL], runs=runs, results_dir=out_dir)
    with open(results_file(out_dir, MODEL), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_one_record_per_cell(cfg, markets, tmp_path):
    recs = _run(cfg, markets, tmp_path)
    assert len(recs) == len(markets) * 3 * 2
    assert all(r["config_hash"] == config_hash(cfg) for r in recs)


def test_seed_shared_across_conditions_not_runs(cfg, markets, tmp_path):
    seeds = defaultdict(set)
    for r in _run(cfg, markets, tmp_path):
        seeds[(r["market_id"], r["run"])].add(r["seed"])
    assert all(len(s) == 1 for s in seeds.values())  # same seed in all 3 conditions
    for m in markets:
        assert seeds[(m.market_id, 0)] != seeds[(m.market_id, 1)]


def test_decoy_fixed_per_market_and_matches_generator(cfg, markets, tmp_path):
    expected = {r["market_id"]: round(r["decoy_price"] * 100) for r in make_decoys(
        [{"market_id": m.market_id, "snapshot_price": m.snapshot_price} for m in markets],
        seed=cfg["experiment"]["decoy_seed"])}
    shown = defaultdict(set)
    for r in _run(cfg, markets, tmp_path, runs=3):
        if r["condition"] == "decoy":
            shown[r["market_id"]].add(r["shown_price"])
    assert {k: v.pop() for k, v in shown.items() if len(v) == 1} == expected


def test_shown_prices_by_condition(cfg, markets, tmp_path):
    price = {m.market_id: m.price_points for m in markets}
    for r in _run(cfg, markets, tmp_path, runs=1):
        if r["condition"] == "blind":
            assert r["shown_price"] is None
        elif r["condition"] == "informed":
            assert r["shown_price"] == price[r["market_id"]]
        else:
            assert abs(r["shown_price"] - price[r["market_id"]]) >= 20


def test_reproducible_across_runs(cfg, markets, tmp_path):
    a = _run(cfg, markets, tmp_path / "a")
    b = _run(cfg, markets, tmp_path / "b")
    key = lambda r: (r["market_id"], r["condition"], r["run"], r["seed"], r["raw_response"])
    assert [key(r) for r in a] == [key(r) for r in b]


def test_resume_skips_done_cells(cfg, markets, tmp_path):
    _run(cfg, markets, tmp_path, runs=1)
    recs = _run(cfg, markets, tmp_path, runs=2)  # only run 1 is new
    assert len(recs) == len(markets) * 3 * 2


def test_resume_refuses_a_different_config(cfg, markets, tmp_path):
    _run(cfg, markets, tmp_path, runs=1)
    other = copy.deepcopy(cfg)
    other["experiment"]["temperature"] = 0.2
    with pytest.raises(RuntimeError, match="do not mix settings"):
        _run(other, markets, tmp_path, runs=1)


def test_unknown_digest_stops_the_run(cfg, markets, tmp_path):
    class NoDigest(FakeClient):
        def digest_for(self, model):
            return "unknown"
    with pytest.raises(RuntimeError, match="no digest"):
        _run(cfg, markets, tmp_path, client=NoDigest())


def test_percent_prices_rejected(tmp_path, cfg):
    src = (REPO_ROOT / cfg["data"]["fixture_csv"]).read_text(encoding="utf-8")
    bad = tmp_path / "markets.csv"
    bad.write_text(src.replace(",0.72,", ",72,"), encoding="utf-8")
    with pytest.raises(ValueError, match="not a probability"):
        load_markets(bad)
