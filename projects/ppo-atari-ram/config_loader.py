"""Two-tier config: _defaults.json base + per-env JSON override."""
import json
import os
from typing import Dict, Any


def load_config(env_id: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Load merged config for an environment.

    Merges _defaults.json (base) with <env_id>.json (overrides).
    The env_id key is always set from the override file.
    """
    defaults_path = os.path.join(config_dir, "_defaults.json")
    with open(defaults_path) as f:
        config = json.load(f)

    env_path = os.path.join(config_dir, f"{env_id}.json")
    if os.path.exists(env_path):
        with open(env_path) as f:
            overrides = json.load(f)
        config.update(overrides)

    config.setdefault("env_id", env_id)
    return config


def list_configs(config_dir: str = "configs"):
    """List all env config files (excluding _defaults.json)."""
    files = sorted(os.listdir(config_dir))
    return [f.replace(".json", "") for f in files
            if f.endswith(".json") and not f.startswith("_")]
