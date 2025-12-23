"""
Configuration utilities for loading YAML configs
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent
