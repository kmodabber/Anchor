# ANCHOR — Method notes (for the Method section of the report)

These notes freeze the exact prompts, models, and call settings used by
the pipeline. Prompts are frozen Monday Nov 23; any later change should
go in as a second named variant in `src/anchor/prompts.py` so old runs
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

- Temperature: 0.7
- Runs per cell (market × condition × model): 5
- Every call is a fresh request with no chat history.
- Seed: deterministic per `(market_id, condition, model, run)`, derived
  from `seed_base` in `config/config.yaml`. See
  `anchor.config.Config.seed_for`.
- A parse failure triggers one retry with a bumped seed. A second
  failure is logged and counted against the model's failure rate.
- Models with over 10% parse failures on the dev set are replaced or
  dropped per group rule.

## Conditions

Three conditions, identical in every way except the price line:

1. **Blind** — no price shown.
2. **Informed** — the real snapshot price shown.
3. **Decoy** — a random price shown, at least 20 points from the real
   one (per `decoy_min_distance`).

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

`model`, `model_digest`, `market_id`, `condition`, `run`, `seed`,
`shown_price`, `prompt`, `raw_response`, `parsed_probability`,
`retry_count`, `final_status`, `elapsed_s`, `timestamp`.

## Reproducibility

- Pinned Ollama version and model digests recorded in `SETUP.md` and
  printed by `scripts/check_ollama.py`.
- Deterministic seeds from the shared config.
- Resumable runs: already-logged `(market_id, condition, run)` tuples
  are skipped on restart.
