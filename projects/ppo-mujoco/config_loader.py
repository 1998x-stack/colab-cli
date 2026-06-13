"""Two-tier config: _defaults.json base + per-env JSON override."""
import json
import os
from typing import Dict, Any


def load_config(env_name: str, config_dir: str = "configs") -> Dict[str, Any]:
    defaults_path = os.path.join(config_dir, "_defaults.json")
    with open(defaults_path) as f:
        config = json.load(f)
    env_path = os.path.join(config_dir, f"{env_name}.json")
    if os.path.exists(env_path):
        with open(env_path) as f:
            overrides = json.load(f)
        config.update(overrides)
    config.setdefault("env_id", env_name)
    return config


def list_configs(config_dir: str = "configs"):
    files = sorted(os.listdir(config_dir))
    return [f.replace(".json", "") for f in files
            if f.endswith(".json") and not f.startswith("_")]
