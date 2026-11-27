# ANCHOR evaluation protocol

CS3346A Group 16. Owner: Aryan Bhasin. Status: **draft for team sign-off**.

This document fixes how every reported number is computed. It must be agreed and frozen before any test-set results are looked at. After freezing, any change is logged in section 11 with the date and reason, and results computed before the change are kept.

The code in `eval/` implements this document exactly. The settings in `eval/config.py` match the values below.

## 1. Question and design

Each of four models (Llama 3.2 3B, Llama 3.1 8B, Qwen 2.5 3B, Qwen 2.5 7B) forecasts every market under three conditions, with prompts identical except for the price line:

- **Blind:** the question only.
- **Informed:** the question plus the real market price at the snapshot (7 days before resolution).
- **Decoy:** the question plus a random price at least 20 points from the real price, drawn once per market with a fixed seed.

A model that reasons about the price should move toward the real price more than toward the decoy, and should become more accurate when given the real price. A model that parrots moves toward any number it is shown.

## 2. Data and split

- Markets follow the eligibility rules in the project plan (created on or after 2025-01-01, resolved, snapshot price between 5% and 95%, one market per event, cross-platform duplicates removed).
- **Split:** 30% development, 70% test, stratified by category. Within each category, markets are sorted by `market_id`, shuffled with a seed derived from `SPLIT_SEED = 3346` and the category name, and the first round(0.3 × n) go to dev. Produced by `python -m eval.split` and saved as `split.csv`; the file is committed and never regenerated after the test run.
- **Dev** is used freely for prompt and parser tuning. **Test** is evaluated once, after prompts are frozen. Only test-set numbers are reported as main results.

## 3. Input files

| File | Owner | Required columns |
|---|---|---|
| `markets.csv` | Evan | `market_id, category, snapshot_price, outcome` (plus `source, created, …` from the plan) |
| `decoys.csv` | Kamyar | `market_id, decoy_price` |
| `split.csv` | Aryan | `market_id, split` (`dev` or `test`) |
| `responses.csv` | Nima | `market_id, model, condition, run, forecast`; optional `prompt_variant` (default `original`), `shown_price`, `parse_ok` |

- Responses can also be read straight from the pipeline's raw JSONL (`results/raw/<phase>/<model>.jsonl.gz`): `parsed_probability` becomes `forecast`, `final_status = ok` becomes `parse_ok`, and the pipeline's prompt variant `v1` is called `original` here. No definition changes.
- One row per model call. `condition` is `blind`, `informed` or `decoy`. `model` is the Ollama tag (`llama3.1:8b`).
- Prices and forecasts may be 0–1 or 0–100; a column whose maximum exceeds 1 is read as percent.
- A parse failure is a row with an empty `forecast` (or `parse_ok` = 0). Failed calls stay in the file so failure rates can be reported.
- If `shown_price` is present, it is used as the price the model saw, and any disagreement with `markets.csv` or `decoys.csv` is reported as a data warning.

## 4. Metrics

### 4.1 Market-level forecast

For each model, market and condition, the forecast is the **mean of the valid runs**. It is kept only if at least half of the attempted runs parsed (3 of 5 in the main run). A market enters a model's analysis only if all three conditions are kept ("complete"). All comparisons within a model use the same complete markets.

### 4.2 Accuracy

- **Brier score:** mean of (forecast − outcome)² over markets, outcome 1 for yes and 0 for no. Range 0 to 1, lower is better.
- **Parrot baseline:** a forecaster that outputs the real price. Its Brier score is the market's own accuracy.
- **Brier gain** = Brier(blind) − Brier(informed). Positive means the real price made the model more accurate.

### 4.3 Anchoring

For market *i* and condition *c* (informed or decoy), let *s* be the shown price, *p_b* the blind forecast and *p_c* the forecast with the price. Define the gap *x = s − p_b* and the move *y = p_c − p_b*. All three measures read 0 when the model ignores the price and 1 when it copies it.

1. **Mean anchoring coefficient** (proposal definition, secondary): mean of *y / x* over markets with |*x*| ≥ 0.05. The median is also reported.
2. **Slope** (proposal's check, secondary): Σ*xy* / Σ*x*² over all complete markets.
3. **Corrected slope** (primary): (Σ*xy* − Σσ²) / (Σ*x*² − Σσ²), where σ²ᵢ = (sample variance of market *i*'s blind runs) / (number of valid blind runs). Markets with one valid blind run use the median σ² of the other markets.

**Why the corrected slope is primary.** The blind forecast is an average of a few noisy runs, so it contains run-to-run noise *e*. That same *e* appears with a minus sign in both *x* and *y*, so it adds Σσ² to both the numerator and the denominator of the slope, which pulls every slope toward 1. The pull is stronger when the shown prices are close to the blind forecast, which is more often true of the real price than of the decoy (decoys are at least 20 points from the real price). So the uncorrected measures overstate informed anchoring more than decoy anchoring, and can report a difference between real and random prices that does not exist. Subtracting Σσ², which is estimated from the blind runs themselves, removes this bias. Section 10 shows the size of the effect on simulated data.

## 5. Main test and verdicts

For each model, on the test set, with 95% bootstrap intervals (section 8):

- **Anchoring gap** = corrected slope (informed) − corrected slope (decoy).
- **Brier gain** as in 4.2.

| Verdict | Rule |
|---|---|
| Reasoning | gap interval above 0 **and** Brier gain interval above 0 |
| Discriminates | gap interval above 0, Brier gain interval contains 0 |
| Parroting | gap interval contains 0 and decoy slope interval above 0 |
| Ignores the price | informed and decoy slope intervals both contain 0 or are below it |
| Inverted | gap interval below 0 |

"Informed − parrot" Brier is reported for every model as context. It answers whether a model adds anything beyond the market price; it is not part of the verdict.

## 6. Model comparisons

Paired over markets that are complete for both models, using the same bootstrap resamples for both. Four planned contrasts, each reported as model A − model B:

- Size within Llama: Llama 3.1 8B − Llama 3.2 3B
- Size within Qwen: Qwen 2.5 7B − Qwen 2.5 3B
- Family at small size: Qwen 2.5 3B − Llama 3.2 3B
- Family at medium size: Qwen 2.5 7B − Llama 3.1 8B

For each: difference in anchoring gap, in Brier gain and in informed Brier. A difference is called clear only when its interval excludes 0. No other model contrasts are tested.

## 7. Robustness and secondary analyses

- **Paraphrase robustness:** the test set is rerun with Talha's paraphrased prompt (`prompt_variant` = `paraphrase`). For each model the result **holds** if the anchoring-gap interval has the same sign under both wordings (both above 0, both below 0, or both containing 0). The paired change in gap is also reported.
- **By category and by platform:** anchoring gap and Brier gain per group. Groups under 20 markets are reported but marked and not interpreted. These are descriptive; no verdicts are drawn from them.
- **Parse failures:** calls, failures and rate per model, condition and prompt variant. A rate above 10% is flagged (the plan's replacement rule).
- **Secondary anchoring measures:** the mean coefficient, its median and the uncorrected slope are reported for every model next to the primary measure, so readers can compare with the proposal's definition.

## 8. Uncertainty

- Nonparametric bootstrap over markets: 2,000 resamples, seed 3346, 95% percentile intervals.
- Every statistic for a model (and both models in a contrast) is recomputed on the same resampled market set, so paired differences account for market-to-market variation.
- An interval is reported as undefined if more than 10% of resamples give an undefined value.

## 9. Reporting rules

- Main results are test-set numbers from a single run of `python -m eval.evaluate --subset test`, after prompts are frozen.
- Dev-set numbers may appear in the report only labelled as development results.
- Every reported number comes from the files in the results folder; none are typed by hand.
- `run_info.json` records the settings, the subset, the run time and any data-check warnings, and is committed with the results.

## 10. Validation on simulated data

`python -m eval.simulate` creates fake markets and model answers where each model's true anchoring weights are known. On 40 simulated datasets of 300 markets, for a model that moves 60% of the way toward any number (true gap 0):

| Measure | Average gap reported | Wrongly calls it discriminating (5 runs) | (3 runs) |
|---|---|---|---|
| Mean anchoring coefficient | +0.034 | 15% | 15% |
| Slope | +0.020 | 22% | 20% |
| Corrected slope | −0.001 | 0% | 0% |

A correct 95% interval should do this about 2.5% of the time. The uncorrected measures fail; the corrected slope does not. On the same simulated data, the corrected slope recovers each model's true informed and decoy weights to within about 0.03 (`tests/test_eval.py` checks this).

## 11. Changes from the proposal and change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | Primary anchoring measure changed from the mean coefficient to the corrected slope; both proposal measures kept as secondary | Section 4.3 and section 10: the proposal measures report false differences between real and decoy prices |
| 2026-11-22 | Markets need at least half of their runs parsed in each condition | Avoids forecasts resting on one run |
| 2026-11-27 | Evaluation reads the pipeline's JSONL files directly (input format only; metrics unchanged) | The pipeline writes JSONL, not `responses.csv`; reading it directly avoids a hand conversion step |
| | *Protocol frozen (date, team sign-off)* | |
