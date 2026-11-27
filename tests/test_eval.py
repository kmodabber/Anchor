"""Tests for the ANCHOR evaluation code. Run with:  python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from eval import config, io, metrics as M, simulate as S
from eval.evaluate import run
from eval.split import make_split


def _arrays(pb, pi, pd_, si, sd, y, sig2=None):
    a = {k: np.asarray(v, float) for k, v in dict(pb=pb, pi=pi, pd=pd_, si=si, sd=sd, y=y).items()}
    a["sig2"] = np.zeros(len(a["y"])) if sig2 is None else np.asarray(sig2, float)
    a["market_id"] = np.array([f"m{i}" for i in range(len(a["y"]))])
    return a


# --- metric definitions ------------------------------------------------------

def test_brier_known_values():
    a = _arrays(pb=[0.5, 0.5], pi=[0.9, 0.2], pd_=[0.5, 0.5], si=[1.0, 0.0], sd=[0, 1], y=[1, 0])
    s = M.compute_stats(a)
    assert s["brier_blind"][0] == pytest.approx(0.25)
    assert s["brier_informed"][0] == pytest.approx((0.01 + 0.04) / 2)
    assert s["brier_parrot"][0] == pytest.approx(0.0)
    assert s["brier_gain"][0] == pytest.approx(0.25 - 0.025)


def test_anchoring_coefficient_worked_example():
    # Blind 45%, real price 62%, informed 58% -> moved 13 of 17 points.
    # Decoy 21%, decoy answer 30% -> moved 15 of 24 points.
    a = _arrays(pb=[0.45], pi=[0.58], pd_=[0.30], si=[0.62], sd=[0.21], y=[1])
    s = M.compute_stats(a)
    assert s["ac_informed"][0] == pytest.approx(13 / 17)
    assert s["ac_decoy"][0] == pytest.approx(15 / 24)
    assert s["slope_informed"][0] == pytest.approx(13 / 17)


def test_exclusion_rule_drops_small_gaps():
    # Second market: shown price only 3 points from blind -> excluded from the coefficient.
    a = _arrays(pb=[0.40, 0.50], pi=[0.50, 0.90], pd_=[0.40, 0.50], si=[0.60, 0.53], sd=[0.9, 0.9], y=[1, 1])
    s = M.compute_stats(a)
    assert s["ac_informed"][0] == pytest.approx(0.5)
    assert M.eligible_counts(a)["n_ac_informed"] == 1


def test_corrected_slope_equals_slope_without_noise():
    rng = np.random.default_rng(0)
    pb = rng.uniform(0.2, 0.8, 50)
    si = rng.uniform(0.05, 0.95, 50)
    a = _arrays(pb=pb, pi=pb + 0.5 * (si - pb), pd_=pb, si=si, sd=si, y=np.ones(50))
    s = M.compute_stats(a)
    assert s["cslope_informed"][0] == pytest.approx(s["slope_informed"][0])
    assert s["cslope_informed"][0] == pytest.approx(0.5)


# --- input handling ----------------------------------------------------------

def test_inputs_accept_percent_and_yes_no(tmp_path):
    pd.DataFrame({"market_id": ["a", "b"], "category": ["x", "y"], "snapshot_price": [62, 30],
                  "outcome": ["YES", "no"]}).to_csv(tmp_path / "m.csv", index=False)
    m = io.load_markets(tmp_path / "m.csv")
    assert m.snapshot_price.tolist() == [0.62, 0.30]
    assert m.outcome.tolist() == [1, 0]


def test_parse_failures_and_min_valid_runs(tmp_path):
    rows = [("a", "llama3.1:8b", c, r, 50 if (c != "informed" or r <= 2) else None) for c in config.CONDITIONS
            for r in range(1, 6)]
    pd.DataFrame(rows, columns=["market_id", "model", "condition", "run", "forecast"]).to_csv(tmp_path / "r.csv", index=False)
    r = io.load_responses(tmp_path / "r.csv")
    markets = pd.DataFrame({"market_id": ["a"], "category": ["x"], "source": ["k"], "snapshot_price": [0.6], "outcome": [1]})
    mt = M.market_table(r, markets, pd.DataFrame({"market_id": ["a"], "decoy_price": [0.2]}))
    assert mt.k_informed.iloc[0] == 2          # 2 of 5 runs parsed
    assert np.isnan(mt.p_informed.iloc[0])     # below the 3-of-5 threshold
    assert not mt.complete.iloc[0]


def test_duplicate_response_rows_rejected(tmp_path):
    pd.DataFrame({"market_id": ["a", "a"], "model": ["m", "m"], "condition": ["blind", "blind"],
                  "run": [1, 1], "forecast": [40, 50]}).to_csv(tmp_path / "r.csv", index=False)
    with pytest.raises(io.DataError):
        io.load_responses(tmp_path / "r.csv")


def _pipeline_record(market_id, condition, run, parsed, shown, model="llama3.1:8b", variant="v1", phase="main"):
    """A line in the format pipeline/runner.py writes."""
    return {"model": model, "model_digest": "abc", "config_hash": "h1", "phase": phase,
            "prompt_variant": variant, "market_id": market_id, "condition": condition, "run": run,
            "seed": 1, "shown_price": shown, "prompt": "...", "raw_response": str(parsed),
            "parsed_probability": parsed, "retry_count": 0,
            "final_status": "ok" if parsed is not None else "parse_failure",
            "elapsed_s": 0.1, "timestamp": "2026-11-27T10:00:00-05:00"}


def test_reads_pipeline_jsonl_folder(tmp_path):
    import gzip
    phase = tmp_path / "main"
    phase.mkdir()
    recs = [_pipeline_record("a", "blind", 0, 40, None), _pipeline_record("a", "informed", 0, 70, 76),
            _pipeline_record("a", "decoy", 0, None, 15)]
    lines = "".join(json.dumps(r) + "\n" for r in recs)
    with gzip.open(phase / "llama3.1_8b.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(lines)
    # plain resume copy with the same content plus a half-written line: the .gz is read, not both
    (phase / "llama3.1_8b.jsonl").write_text(lines + '{"model": "lla', encoding="utf-8")
    (phase / "qwen2.5_7b.jsonl").write_text(json.dumps(_pipeline_record("a", "blind", 0, 55, None, model="qwen2.5:7b")) + "\n",
                                            encoding="utf-8")
    r = io.load_responses(phase)
    assert len(r) == 4
    llama = r[r.model == "llama3.1:8b"].set_index("condition")
    assert llama.loc["blind", "forecast"] == pytest.approx(0.40)
    assert llama.loc["informed", "shown_price"] == pytest.approx(0.76)
    assert np.isnan(llama.loc["decoy", "forecast"])
    assert set(r.prompt_variant) == {config.ORIGINAL_VARIANT}


def test_jsonl_and_csv_sources_combine_but_dev_main_collide(tmp_path):
    (tmp_path / "main.jsonl").write_text(json.dumps(_pipeline_record("a", "blind", 0, 40, None)) + "\n", encoding="utf-8")
    (tmp_path / "dev.jsonl").write_text(json.dumps(_pipeline_record("a", "blind", 0, 45, None, phase="dev")) + "\n",
                                        encoding="utf-8")
    (tmp_path / "para.jsonl").write_text(json.dumps(_pipeline_record("a", "blind", 0, 50, None, variant="paraphrase")) + "\n",
                                         encoding="utf-8")
    r = io.load_responses([tmp_path / "main.jsonl", tmp_path / "para.jsonl"])
    assert sorted(r.prompt_variant) == ["original", "paraphrase"]
    with pytest.raises(io.DataError, match="phases"):
        io.load_responses([tmp_path / "main.jsonl", tmp_path / "dev.jsonl"])


# --- split -------------------------------------------------------------------

def test_split_is_deterministic_stratified_and_order_free():
    markets, _, _, _ = S.simulate(300, seed=1, test_ids=[])
    a = make_split(markets)
    b = make_split(markets.sample(frac=1, random_state=5))
    assert a.equals(b)
    assert set(a.market_id) == set(markets.market_id)
    m = a.merge(markets[["market_id", "category"]])
    for _, g in m.groupby("category"):
        assert abs((g.split == "dev").mean() - 0.30) <= 0.5 / len(g) + 1e-9


# --- recovery of known truth on simulated data --------------------------------

@pytest.fixture(scope="module")
def sim():
    markets, decoys, resp, truth = S.simulate(300, seed=11, test_ids=[])
    markets = markets.assign(snapshot_price=markets.snapshot_price / 100, outcome=(markets.outcome == "YES").astype(int))
    decoys = decoys.assign(decoy_price=decoys.decoy_price / 100)
    resp = resp.assign(forecast=resp.forecast / 100, shown_price=resp.shown_price / 100)
    return markets, decoys, resp, truth


def test_corrected_slope_recovers_true_weights(sim):
    markets, decoys, resp, truth = sim
    mt = M.market_table(resp, markets, decoys)
    for model, t in truth.items():
        s = M.summarise(M.model_arrays(mt[mt.model == model]), n_boot=500)
        assert s["cslope_informed"] == pytest.approx(t["w_inf"], abs=0.06), model
        assert s["cslope_decoy"] == pytest.approx(t["w_dec"], abs=0.06), model
        assert s["cslope_gap_lo"] <= t["gap"] + 0.02 and t["gap"] - 0.02 <= s["cslope_gap_hi"], model


def test_parrot_model_not_called_reasoning(sim):
    markets, decoys, resp, _ = sim
    mt = M.market_table(resp, markets, decoys)
    s = M.summarise(M.model_arrays(mt[mt.model == "llama3.2:3b"]), n_boot=500)
    assert M.verdict(s).startswith("Parroting")


def test_end_to_end_writes_outputs(tmp_path):
    markets, decoys, _, _ = S.simulate(120, seed=3, test_ids=[])
    split = make_split(markets.assign(snapshot_price=markets.snapshot_price / 100))
    markets, decoys, resp, _ = S.simulate(120, seed=3, test_ids=split.loc[split.split == "test", "market_id"])
    for name, df in (("m", markets), ("d", decoys), ("r", resp), ("s", split)):
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    summary, contrasts, robust = run(io.load_markets(tmp_path / "m.csv"), io.load_decoys(tmp_path / "d.csv"),
                                     io.load_responses(tmp_path / "r.csv"), io.load_split(tmp_path / "s.csv"),
                                     "test", tmp_path / "out", n_boot=200, figures=True)
    for f in ("summary.md", "model_summary.csv", "model_comparisons.csv", "robustness.csv",
              "breakdown_category.csv", "parse_rates.csv", "run_info.json", "figures/anchoring.png"):
        assert (tmp_path / "out" / f).exists(), f
    assert len(contrasts) == 4 and len(robust) == 4
    assert json.loads((tmp_path / "out" / "run_info.json").read_text())["subset"] == "test"
