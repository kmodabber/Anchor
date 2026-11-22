"""Pipeline: iterate (market, condition, model, run), call the client,
parse, retry once on parse failure, log everything as JSONL, resume on
restart.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Config
from .decoy import decoy_for
from .markets import Market
from .model_client import ModelClient
from .parser import parse_probability
from .prompts import CONDITIONS, render


@dataclass
class CallKey:
    model: str
    market_id: str
    condition: str
    run: int

    def as_tuple(self) -> tuple[str, str, str, int]:
        return (self.model, self.market_id, self.condition, self.run)


@dataclass
class RunStats:
    calls: int = 0
    retries: int = 0
    parse_failures: int = 0
    seconds: float = 0.0
    failed_examples: list[str] = field(default_factory=list)

    def failure_rate(self) -> float:
        return self.parse_failures / self.calls if self.calls else 0.0


def _results_file(results_dir: Path, model: str) -> Path:
    safe = model.replace(":", "_").replace("/", "_")
    return results_dir / f"{safe}.jsonl"


def _load_completed(path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("final_status") == "ok" or rec.get("final_status") == "parse_failure":
                done.add((rec["market_id"], rec["condition"], int(rec["run"])))
    return done


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline(
    *,
    cfg: Config,
    client: ModelClient,
    markets: Iterable[Market],
    models: list[str],
    conditions: tuple[str, ...] = CONDITIONS,
    runs: int | None = None,
    results_dir: Path | None = None,
    precomputed_decoys: dict[str, int] | None = None,
) -> dict[str, RunStats]:
    runs = runs if runs is not None else cfg.runs_per_cell
    results_dir = results_dir or cfg.path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    precomputed_decoys = precomputed_decoys or {}
    stats: dict[str, RunStats] = {m: RunStats() for m in models}
    markets = list(markets)

    for model in models:
        out_path = _results_file(results_dir, model)
        completed = _load_completed(out_path)
        digest = client.digest_for(model)
        s = stats[model]

        for market in markets:
            for condition in conditions:
                for run in range(runs):
                    key = (market.market_id, condition, run)
                    if key in completed:
                        continue

                    seed = cfg.seed_for(market.market_id, condition, model, run)
                    decoy_price = precomputed_decoys.get(market.market_id)
                    if condition == "decoy" and decoy_price is None:
                        decoy_price = decoy_for(
                            market.market_id,
                            market.snapshot_price,
                            seed=seed,
                            min_distance=cfg.decoy_min_distance,
                        )

                    prompt = render(
                        condition=condition,
                        question=market.question,
                        snapshot_date=market.snapshot_date,
                        real_price=market.snapshot_price,
                        decoy_price=decoy_price,
                    )

                    t0 = time.time()
                    retries = 0
                    parsed: float | None = None
                    raw_text = ""
                    for attempt in range(2):  # one retry on parse failure
                        resp = client.generate(
                            model=model,
                            prompt=prompt.text,
                            seed=seed + attempt,  # second try: bumped seed
                            temperature=cfg.temperature,
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

                    record = {
                        "model": model,
                        "model_digest": digest,
                        "market_id": market.market_id,
                        "condition": condition,
                        "run": run,
                        "seed": seed,
                        "shown_price": prompt.shown_price,
                        "prompt": prompt.text,
                        "raw_response": raw_text,
                        "parsed_probability": parsed,
                        "retry_count": retries,
                        "final_status": final_status,
                        "elapsed_s": round(elapsed, 4),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    }
                    _append(out_path, record)

    return stats
