"""Run loop: iterate (model, market, condition, run), call the client, parse,
retry on parse failure, log everything as JSONL, resume on restart.

All settings come from config/config.yaml via common.config:
- seeds: common.config.call_seed(cfg, market_id, run), shared by the three
  conditions so they differ only in the price line
- decoys: pipeline.decoy.make_decoys with experiment.decoy_seed, the same
  function and seed that write data/decoys.csv, so one fixed decoy per market
- every record carries the config hash; a results file is never resumed under
  a different config
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from common.config import call_seed, config_hash
from .decoy import make_decoys
from .markets import Market
from .model_client import ModelClient
from .parser import parse_probability
from .prompts import CONDITIONS, render


@dataclass
class RunStats:
    calls: int = 0
    retries: int = 0
    parse_failures: int = 0
    seconds: float = 0.0
    failed_examples: list[str] = field(default_factory=list)

    def failure_rate(self) -> float:
        return self.parse_failures / self.calls if self.calls else 0.0


def results_file(results_dir: Path, model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return results_dir / f"{safe}.jsonl"


def _load_completed(path: Path, current_hash: str) -> set[tuple[str, str, int]]:
    """Keys already logged in `path`. Raises if any record came from a different
    config, so one file never mixes two experimental settings."""
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # half-written last line from an interrupted run
            if rec.get("config_hash") != current_hash:
                raise RuntimeError(
                    f"{path} has records from config {rec.get('config_hash')}, but the "
                    f"current config is {current_hash}. Move the file aside or use a "
                    "new results directory; do not mix settings in one file.")
            if rec.get("final_status") in ("ok", "parse_failure"):
                done.add((rec["market_id"], rec["condition"], int(rec["run"])))
    return done


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def decoy_points(cfg: dict, markets: list[Market]) -> dict[str, int]:
    """market_id -> decoy as a whole percentage. Also rejects duplicate ids."""
    d = cfg["decoy"]
    rows = make_decoys(
        [{"market_id": m.market_id, "snapshot_price": m.snapshot_price} for m in markets],
        seed=cfg["experiment"]["decoy_seed"],
        low=d["low_points"], high=d["high_points"], min_gap=d["min_gap_points"])
    return {r["market_id"]: int(round(r["decoy_price"] * 100)) for r in rows}


def run_pipeline(
    *,
    cfg: dict,
    client: ModelClient,
    markets: Iterable[Market],
    models: list[str],
    runs: int,
    results_dir: Path,
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, RunStats]:
    """models: Ollama tags, e.g. 'llama3.1:8b'."""
    exp = cfg["experiment"]
    attempts = 1 + cfg["parser"]["retries"]
    chash = config_hash(cfg)
    markets = list(markets)
    decoys = decoy_points(cfg, markets)
    results_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, RunStats] = {m: RunStats() for m in models}

    for model in models:
        out_path = results_file(results_dir, model)
        completed = _load_completed(out_path, chash)
        digest = client.digest_for(model)
        if digest == "unknown":
            raise RuntimeError(f"no digest for {model}: is Ollama running and the model "
                               "pulled? (python -m scripts.check_ollama)")
        s = stats[model]

        for market in markets:
            for condition in conditions:
                for run in range(runs):
                    if (market.market_id, condition, run) in completed:
                        continue

                    seed = call_seed(cfg, market.market_id, run)
                    prompt = render(
                        condition=condition,
                        question=market.question,
                        snapshot_date=market.snapshot_date,
                        real_price=market.price_points,
                        decoy_price=decoys[market.market_id],
                    )

                    t0 = time.time()
                    retries = 0
                    parsed: float | None = None
                    raw_text = ""
                    for attempt in range(attempts):
                        resp = client.generate(
                            model=model,
                            prompt=prompt.text,
                            # a retry needs new randomness; keep it in 31 bits
                            seed=(seed + attempt) % 2**31,
                            temperature=exp["temperature"],
                            max_tokens=exp["max_output_tokens"],
                        )
                        raw_text = resp.raw_text
                        parsed = parse_probability(raw_text)
                        if parsed is not None:
                            break
                        retries += 1

                    elapsed = time.time() - t0
                    s.calls += 1
                    s.seconds += elapsed
                    s.retries += retries
                    final_status = "ok" if parsed is not None else "parse_failure"
                    if parsed is None:
                        s.parse_failures += 1
                        if len(s.failed_examples) < 5:
                            s.failed_examples.append(raw_text[:200])

                    _append(out_path, {
                        "model": model,
                        "model_digest": digest,
                        "config_hash": chash,
                        "market_id": market.market_id,
                        "condition": condition,
                        "run": run,
                        "seed": seed,               # retry k used (seed + k) % 2**31
                        "shown_price": prompt.shown_price,  # whole percent, None if blind
                        "prompt": prompt.text,
                        "raw_response": raw_text,
                        "parsed_probability": parsed,       # 0-100, None on failure
                        "retry_count": retries,
                        "final_status": final_status,
                        "elapsed_s": round(elapsed, 4),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })

    return stats
