# ANCHOR: Do Language Models Reason?

CS3346A Group 16 final project. Do open-weight language models reason about a
prediction-market price or simply copy it? Each model forecasts resolved
Polymarket and Kalshi questions under three conditions: no price (blind), the
real price (informed) and a random price (decoy).

Team: Talha Chaudhry, Nima Abbasi, Evan Thomas, Aryan Bhasin, Kamyar Modabber.

## Layout

| Folder | Contents | Owner |
|---|---|---|
| `config/` | `config.yaml`: every seed, model tag, temperature, run count and path | Kamyar |
| `common/` | shared helpers (config loader) | Kamyar |
| `data/` | market dataset, decoys, split, data card | Evan |
| `pipeline/` | prompts, Ollama calls, output parser, decoy generator | Nima, Kamyar |
| `eval/` | Brier, anchoring coefficient, parrot baseline, bootstrap | Aryan |
| `results/` | raw responses, tables, figures | Aryan, Kamyar |
| `report/` | IEEE LaTeX source | Talha |
| `scripts/` | command-line entry points | all |
| `docs/` | contributions log, meeting notes, method notes | Kamyar, Nima |
| `fixtures/` | 10 made-up markets for testing the pipeline before the real data | Nima |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                        # should pass before any run
```

Install [Ollama](https://ollama.com) and pull the models for your machine
(full steps in `SETUP.md`):

```bash
ollama pull llama3.2:3b && ollama pull llama3.1:8b     # Aryan's RTX 4060
ollama pull qwen2.5:3b  && ollama pull qwen2.5:7b      # Talha's RTX 5070
python -m scripts.check_ollama                         # version + digests; copy them below
```

## Running

All commands run from the repo root.

```bash
python -m scripts.make_decoys                                   # needs data/markets.csv
python -m pipeline.cli parser-test --n 10 --fixture --dry-run   # fake model, no Ollama
python -m pipeline.cli parser-test --n 10                       # real models, dev markets only
python -m pipeline.cli parser-test --models qwen2.5:3b qwen2.5:7b   # one machine's models
# eval commands: to be added by Aryan
```

## Reproducibility record

Fill in before the main run. Both machines must match on everything except hardware.

| Item | Value |
|---|---|
| Pipeline commit | |
| Ollama version | |
| Python version | |
| Data pull date | |
| Model digests | llama3.2:3b `…`, llama3.1:8b `…`, qwen2.5:3b `…`, qwen2.5:7b `…` |
| Hardware | Aryan: RTX 4060 8 GB; Talha: RTX 5070 12 GB |
| Seeds | see `config/config.yaml` |

## Rules

- Change settings only in `config/config.yaml`, by commit. Never edit it locally on a run machine.
- The test set is run once, after prompts are frozen. Log any rerun and the reason in `docs/MEETINGS.md`.
- Commit your own work from your own account. Contribution history is part of the grade.
