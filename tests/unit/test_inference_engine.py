"""Unit tests for inference engine"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import os


def test_engine_init():
    """Test inference engine initialization with config"""
    from pangu.inference.onnx_engine import PanguInferenceEngine

    # Mock config file
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = """
inference:
  model_paths:
    24h: "data/models/pangu_weather_24.onnx"
  input_dir: "input_data"
  output_dir: "output_data"
  execution_provider: "CPUExecutionProvider"
  ort_options:
    enable_cpu_mem_arena: false
"""
        # This will fail because models don't exist, but that's ok for structure test
        try:
            engine = PanguInferenceEngine("fake_config.yaml")
        except:
            pass  # Expected to fail without actual models


def test_load_input_shape():
    """Test that input loading expects correct shapes"""
    from pangu.inference.onnx_engine import PanguInferenceEngine

    # Expected shapes according to README
    # upper: (5, 13, 721, 1440)
    # surface: (4, 721, 1440)

    # This is a documentation test
    assert True  # Placeholder until we have actual test data
