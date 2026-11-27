"""Frozen evaluation settings. Change only by team decision, and log the change."""

CONDITIONS = ("blind", "informed", "decoy")
ORIGINAL_VARIANT = "original"
# The pipeline's prompt variant names (pipeline/prompts.py TEMPLATES) -> names used here.
VARIANT_ALIASES = {"v1": ORIGINAL_VARIANT}

# Primary anchoring measure for the main test and the verdicts:
#   "cslope" - noise-corrected slope (primary; see PROTOCOL.md section 4.3)
#   "ac"     - mean anchoring coefficient (the proposal's measure; secondary)
#   "slope"  - uncorrected slope (secondary)
PRIMARY = "cslope"

# Anchoring coefficient exclusion: drop a market when |shown - p_blind| < this.
MIN_GAP = 0.05

# A market's forecast for one model/condition is kept only if at least this
# fraction of its runs parsed (e.g. 3 of 5).
MIN_VALID_RUN_FRAC = 0.5

# Bootstrap over markets.
N_BOOT = 2000
BOOT_SEED = 3346
CI_LEVEL = 0.95

# Groups smaller than this are reported but flagged as too small to interpret.
MIN_GROUP_N = 20

# Eligibility rules from the plan, used only for data checks.
MIN_CREATED_DATE = "2025-01-01"
SNAPSHOT_RANGE = (0.05, 0.95)
MIN_DECOY_DISTANCE = 0.20

# Dev/test split.
DEV_FRACTION = 0.30
SPLIT_SEED = 3346

# Display names and the 2x2 design (family x size).
MODELS = {
    "llama3.2:3b": {"label": "Llama 3.2 3B", "family": "Llama", "size": "small"},
    "llama3.1:8b": {"label": "Llama 3.1 8B", "family": "Llama", "size": "medium"},
    "qwen2.5:3b": {"label": "Qwen 2.5 3B", "family": "Qwen", "size": "small"},
    "qwen2.5:7b": {"label": "Qwen 2.5 7B", "family": "Qwen", "size": "medium"},
}

# Planned model contrasts: (name, model_a, model_b) -> reports a minus b.
CONTRASTS = [
    ("Size within Llama (8B - 3B)", "llama3.1:8b", "llama3.2:3b"),
    ("Size within Qwen (7B - 3B)", "qwen2.5:7b", "qwen2.5:3b"),
    ("Family at small size (Qwen - Llama)", "qwen2.5:3b", "llama3.2:3b"),
    ("Family at medium size (Qwen - Llama)", "qwen2.5:7b", "llama3.1:8b"),
]
