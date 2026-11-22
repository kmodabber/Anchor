# ANCHOR — Ollama + Python setup

Two teammates need identical Ollama versions and model tags. Follow this
exactly. Do not substitute tags; the digests must match between machines.

## Pinned versions

- Ollama: **one exact version on both machines**, set in `config/config.yaml`
  (`ollama.version`) once we confirm a version that supports Talha's RTX 5070.
  `python -m scripts.check_ollama` fails if a machine does not match.
- Model tags:
  - `llama3.2:3b`
  - `llama3.1:8b`
  - `qwen2.5:3b`
  - `qwen2.5:7b`

## 1. Install Ollama

- **Windows**: download the installer from https://ollama.com/download/windows
  and run it. After install, open a new terminal.
- **macOS**: `brew install ollama` (or download the .dmg).
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

Verify:

```
ollama --version
```

Install the exact version in `config/config.yaml` (`ollama.version`), not just the latest.

## 2. Start the Ollama server

- **Windows/macOS**: the installer runs it as a background service. Confirm
  with `curl http://localhost:11434/api/tags` — you should get JSON.
- **Linux**: `ollama serve &` (or systemd).

## 3. Pull the four models

Each pull is several GB. Expect 15–40 min total depending on bandwidth.

```
ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

## 4. Verify — run the project's check script

From the repo root, with the Python environment from step 5 set up and active
(the script reads `config/config.yaml`):

```
python -m scripts.check_ollama
```

It prints the Ollama version and the digest for each required model. Paste
the output in the group chat so we can confirm both machines match.

## 5. Python environment

```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Troubleshooting

- `ollama list` empty after a pull → the service isn't running. Restart it.
- Digest mismatch between machines → one of you pulled a tag that was
  re-published. Re-pull on the older machine; tags are mutable, digests are
  not.
- Out of VRAM on the 4060 (8 GB) with the 7B/8B models → set
  `OLLAMA_NUM_GPU=999` to force full GPU offload and close other GPU apps;
  if it still OOMs, run those two models CPU-only with `OLLAMA_NUM_GPU=0`
  (slow but works).
