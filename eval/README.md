# ANCHOR evaluation code (eval/)

Evaluation scripts for the CS3346A Group 16 project. Owner: Aryan Bhasin. Definitions are in `PROTOCOL.md` at the repo root.

Run every command from the repo root. Definitions: `PROTOCOL.md` (root); tests: `tests/test_eval.py`.

## Setup

```
pip install -r requirements.txt
python -m pytest -q tests/test_eval.py   # 13 tests
```

## Use

1. `data/split.csv` is already committed. Never regenerate it (`python -m eval.split` exists only to document how a split is made).
2. Dev-set numbers while prompts are being tuned:
   ```
   python -m eval.evaluate --markets data/markets.csv --decoys data/decoys.csv \
       --responses results/raw/main --split data/split.csv --subset dev --out results/eval_dev
   ```
3. Final results, once, after prompts are frozen: the same command with `--subset test --out results/eval_test`.
   Add the robustness run with `--responses results/raw/main results/raw/paraphrase`.

`--responses` takes one or more pipeline phase folders (every `<model>.jsonl.gz` in them, so Llama and Qwen files are read together), single `.jsonl`/`.jsonl.gz` files, or a CSV. Don't pass `results/raw/dev` and `results/raw/main` together: both use the same prompt, so their rows collide and loading stops with an error.

## Outputs (in --out)

| File | Contents |
|---|---|
| `summary.md` | All tables in readable form, with verdicts |
| `model_summary.csv` | Every statistic with 95% interval, per model and prompt variant |
| `model_comparisons.csv` | The four planned size and family contrasts |
| `robustness.csv` | Original vs paraphrased prompt, per model |
| `breakdown_category.csv`, `breakdown_source.csv` | Secondary breakdowns |
| `parse_rates.csv` | Parse failures per model and condition |
| `market_level.csv` | Per-market means, for checking and for other figures |
| `run_info.json` | Settings, subset, time and data-check warnings |
| `figures/` | `anchoring.png`, `brier.png`, `shift_vs_gap.png`, `model_comparisons.png` |

## Input formats (for Nima, Evan and Kamyar)

- Pipeline JSONL (preferred): read as written by `python -m pipeline.cli run`; no conversion needed.
- `responses.csv` (alternative): one row per call with `market_id, model, condition, run, forecast`. `condition` is `blind`, `informed` or `decoy`; `model` is the Ollama tag; `forecast` is the parsed probability (0–100 or 0–1), empty when parsing failed. Optional: `prompt_variant` (`original` or `paraphrase`), `shown_price`, `parse_ok`.
- `markets.csv`: the plan's columns; needs at least `market_id, category, snapshot_price, outcome` (`YES`/`NO` or 1/0).
- `decoys.csv`: `market_id, decoy_price`.

## Trying it without real data

```
python -m eval.simulate --out sim_data
python -m eval.evaluate --markets sim_data/markets.csv --decoys sim_data/decoys.csv \
    --responses sim_data/responses.csv --split sim_data/split.csv --subset test --out sim_data/eval
```
Simulated files are fake. Never put them in the results folder.
