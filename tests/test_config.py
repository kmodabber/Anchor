import copy

import pytest

from common.config import config_hash, load_config, models_for_machine, validate


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
