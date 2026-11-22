# ANCHOR — Method notes (for the Method section of the report)

These notes freeze the exact prompts, models, and call settings used by
the pipeline. Prompts are frozen Monday Nov 23; any later change should
go in as a second named variant in `pipeline/prompts.py` so old runs
remain reproducible.

## Models

Served locally through Ollama (http://localhost:11434). Digests are
logged per call so the exact model build is recoverable.

| Tag             | Family  | Params |
|-----------------|---------|--------|
| llama3.2:3b     | Llama   | 3B     |
| llama3.1:8b     | Llama   | 8B     |
| qwen2.5:3b      | Qwen    | 3B     |
| qwen2.5:7b      | Qwen    | 7B     |

## Call settings

All values below are read from `config/config.yaml`; this list mirrors it.

- Temperature: 0.7 (`experiment.temperature`)
- Max output tokens: 16 (`experiment.max_output_tokens`, sent as Ollama `num_predict`)
- Runs per cell (market × condition × model): 5 main, 2 dev, 3 paraphrase
  (`experiment.runs`)
- Every call is a fresh request with no chat history.
- Seed: sha256 of `(experiment.call_seed, market_id, run)`, see
  `common.config.call_seed`. The condition is left out on purpose, so
  blind, informed and decoy for the same market and run share a seed and
  differ only in the price line (common random numbers). The seed is the
  same on every machine and every Python process.
- A parse failure triggers one retry (`parser.retries`) with seed + 1. A
  second failure is logged and counted against the model's failure rate.
- Models with over 10% parse failures on the dev set
  (`parser.max_failure_rate`) are replaced or dropped per group rule.

## Conditions

Three conditions, identical in every way except the price line:

1. **Blind** — no price shown.
2. **Informed** — the real snapshot price shown.
3. **Decoy** — a random whole-percent price in [5, 95], at least 20
   points from the real one (`decoy` section of the config). Drawn once
   per market by `pipeline/decoy.py` with `experiment.decoy_seed`, so the
   same decoy is shown in every run; `data/decoys.csv` records them.

Prices are stored as probabilities from 0 to 1 in `data/markets.csv` and
shown in prompts as whole percentages (0.72 → 72).

The informed and decoy prompts share **identical wording** for the
price line. The model is never told whether the price is real or
random.

## Prompt template (frozen set, English, variant: `v1`)

```
You are forecasting a binary event.

Question: {question}
Snapshot date: {snapshot_date}
Current market price (implied probability, 0-100): {price}
Based on the information above, estimate the probability that the event
resolves YES. Reply with a single integer from 0 to 100 representing the
percent probability. Do not include any other text.

Probability (0-100):
```

Blind condition: the `Current market price` line is omitted entirely;
no other text changes.

The snapshot date is identical across all three conditions for a given
market. The `outcome` column is never included in any prompt.

## Logged fields per call (JSONL)

`model`, `model_digest`, `config_hash`, `market_id`, `condition`, `run`,
`seed`, `shown_price` (whole percent, null for blind), `prompt`,
`raw_response`, `parsed_probability` (0-100, null on failure),
`retry_count`, `final_status`, `elapsed_s`, `timestamp`.

## Reproducibility

- Ollama version pinned in `config/config.yaml` (`ollama.version`) and
  checked, with model digests, by `python -m scripts.check_ollama`.
- Deterministic seeds from the shared config.
- Resumable runs: already-logged `(market_id, condition, run)` tuples
  are skipped on restart. A results file is never resumed under a
  different config: the run stops if the config hash differs.
- A run stops if a model's digest cannot be read, so every logged
  response is tied to a known model build.
