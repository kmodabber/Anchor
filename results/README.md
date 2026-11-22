# results/

- `raw/`: one JSONL file per model, one line per call, committed compressed
  (`.jsonl.gz`). Each line holds the model tag and digest, config hash,
  market_id, condition, run number, seed, shown price, prompt, raw response
  text and parsed probability. Field list: `docs/method_notes.md`.
  - `raw/dry_run/`, `raw/parser_test/`: scratch output of `pipeline.cli parser-test`.
- `tables/`, `figures/`: produced by the scripts in `eval/`.
