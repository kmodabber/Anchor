import copy

import pytest

from common.config import call_seed, config_hash, load_config, models_for_machine, validate


def test_repo_config_is_valid():
    cfg = load_config()
    assert len(cfg["models"]) == 4
    assert {m["family"] for m in cfg["models"]} == {"llama", "qwen"}


def test_each_machine_has_two_models():
    cfg = load_config()
    assert len(models_for_machine(cfg, "aryan-4060")) == 2
    assert len(models_for_machine(cfg, "talha-5070")) == 2


def test_hash_changes_when_config_changes():
    cfg = load_config()
    other = copy.deepcopy(cfg)
    other["experiment"]["temperature"] = 0.0
    assert config_hash(cfg) != config_hash(other)


def test_bad_conditions_rejected():
    cfg = load_config()
    cfg["conditions"] = ["blind", "informed"]
    with pytest.raises(ValueError):
        validate(cfg)


def test_call_seed_is_deterministic_and_in_range():
    cfg = load_config()
    s = call_seed(cfg, "kalshi:ABC", 3)
    assert s == call_seed(cfg, "kalshi:ABC", 3)
    assert 0 <= s < 2**31


def test_call_seed_varies_by_market_run_and_base_seed():
    cfg = load_config()
    seeds = {call_seed(cfg, f"m{i}", r) for i in range(50) for r in range(1, 6)}
    assert len(seeds) == 250  # no collisions across 50 markets x 5 runs
    other = copy.deepcopy(cfg)
    other["experiment"]["call_seed"] += 1
    assert call_seed(cfg, "m1", 1) != call_seed(other, "m1", 1)


@pytest.mark.parametrize("section, key, bad", [
    ("experiment", "call_seed", "3346"),   # string, not int
    ("experiment", "bootstrap_seed", True),
    ("parser", "retries", -1),
    ("parser", "max_failure_rate", 10),    # percent instead of fraction
    ("evaluation", "bootstrap_resamples", 0),
    ("evaluation", "ci_level", 95),        # percent instead of fraction
])
def test_bad_new_settings_rejected(section, key, bad):
    cfg = load_config()
    cfg[section][key] = bad
    with pytest.raises(ValueError):
        validate(cfg)


def test_missing_evaluation_section_rejected():
    cfg = load_config()
    del cfg["evaluation"]
    with pytest.raises(ValueError):
        validate(cfg)
