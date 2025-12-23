"""Unit tests for package imports"""

import pytest


def test_pangu_import():
    """Test that pangu package can be imported"""
    import pangu

    assert pangu.__version__ == "0.1.0"


def test_modules_import():
    """Test that all modules can be imported"""
    from pangu import data, models, inference, training, utils

    # Just verify they exist
    assert data is not None
    assert models is not None
    assert inference is not None
    assert training is not None
    assert utils is not None


def test_config_utils():
    """Test config utilities"""
    from pangu.utils.config import get_project_root
    import os

    root = get_project_root()
    assert root.exists()
    assert (root / "pangu").exists() or (root / "src" / "pangu").exists()
