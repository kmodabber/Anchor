# results/

- `raw/<phase>/`: one file per model, one JSON line per call, written by
  `python -m pipeline.cli run`. Commit the `.jsonl.gz` files; the plain
  `.jsonl` next to each is the working/resume copy and is git-ignored. Each
  line holds the model tag and digest, config hash, phase, prompt variant,
  market_id, condition, run number, seed, shown price, prompt, raw response
  text and parsed probability. Field list: `docs/method_notes.md`.
  - `raw/scratch/`: rehearsal output (`--fixture` / `--dry-run`), git-ignored.
  - `raw/dry_run/`, `raw/parser_test/`: output of `pipeline.cli parser-test`.
- `manifests/<phase>_<machine>.jsonl`: one line per real run: commit, config
  hash, Ollama version, model digests, Python version, OS and GPU. Source for
  the reproducibility appendix.
- `tables/`, `figures/`: produced by the scripts in `eval/`.
