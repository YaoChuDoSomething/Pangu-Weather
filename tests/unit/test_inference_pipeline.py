"""Inference pipeline 單元測試。

測試 preprocessing.py 與 postprocessing.py 的核心功能。
"""

import pytest
import torch
import numpy as np


class TestWeatherPreprocessor:
    """WeatherPreprocessor 測試。"""

    def test_init_with_stats(self):
        """測試使用統計值初始化。"""
        from pangu.inference.preprocessing import WeatherPreprocessor

        mean = {"upper": np.zeros((5, 13, 1, 1)), "surface": np.zeros((4, 1, 1))}
        std = {"upper": np.ones((5, 13, 1, 1)), "surface": np.ones((4, 1, 1))}

        preprocessor = WeatherPreprocessor(mean=mean, std=std)

        assert preprocessor.mean is not None
        assert preprocessor.std is not None

    def test_normalize_with_stats(self):
        """測試有統計值時的正規化。"""
        from pangu.inference.preprocessing import WeatherPreprocessor

        mean = {"upper": np.ones((5, 13, 1, 1)) * 10, "surface": np.ones((4, 1, 1)) * 5}
        std = {"upper": np.ones((5, 13, 1, 1)) * 2, "surface": np.ones((4, 1, 1)) * 2}

        preprocessor = WeatherPreprocessor(mean=mean, std=std)

        upper = torch.ones(1, 5, 13, 10, 10) * 10
        surface = torch.ones(1, 4, 10, 10) * 5

        upper_norm, surface_norm = preprocessor.normalize(upper, surface)

        # (10 - 10) / 2 = 0
        assert torch.allclose(upper_norm, torch.zeros_like(upper_norm))
        assert torch.allclose(surface_norm, torch.zeros_like(surface_norm))

    def test_normalize_without_stats_skips(self):
        """測試無統計值時跳過正規化。"""
        from pangu.inference.preprocessing import WeatherPreprocessor

        preprocessor = WeatherPreprocessor()

        upper = torch.randn(1, 5, 13, 10, 10)
        surface = torch.randn(1, 4, 10, 10)

        upper_out, surface_out = preprocessor.normalize(upper, surface)

        # 應該返回原始資料
        torch.testing.assert_close(upper_out, upper)
        torch.testing.assert_close(surface_out, surface)

    def test_to_tensor(self):
        """測試 numpy 轉 tensor。"""
        from pangu.inference.preprocessing import WeatherPreprocessor

        preprocessor = WeatherPreprocessor()

        upper = np.random.randn(5, 13, 10, 10).astype(np.float32)
        surface = np.random.randn(4, 10, 10).astype(np.float32)

        upper_t, surface_t = preprocessor.to_tensor(upper, surface)

        assert upper_t.shape == (1, 5, 13, 10, 10)
        assert surface_t.shape == (1, 4, 10, 10)
        assert upper_t.dtype == torch.float32

    def test_prepare_input(self):
        """測試完整輸入準備流程。"""
        from pangu.inference.preprocessing import WeatherPreprocessor

        preprocessor = WeatherPreprocessor()

        upper = np.random.randn(5, 13, 10, 10).astype(np.float32)
        surface = np.random.randn(4, 10, 10).astype(np.float32)

        upper_t, surface_t = preprocessor.prepare_input(upper, surface, normalize=False)

        assert isinstance(upper_t, torch.Tensor)
        assert isinstance(surface_t, torch.Tensor)


class TestWeatherPostprocessor:
    """WeatherPostprocessor 測試。"""

    def test_init_with_stats(self):
        """測試使用統計值初始化。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        mean = {"upper": np.zeros((5, 13, 1, 1)), "surface": np.zeros((4, 1, 1))}
        std = {"upper": np.ones((5, 13, 1, 1)), "surface": np.ones((4, 1, 1))}

        postprocessor = WeatherPostprocessor(mean=mean, std=std)

        assert postprocessor.mean is not None
        assert postprocessor.std is not None

    def test_denormalize_with_stats(self):
        """測試有統計值時的反正規化。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        mean = {"upper": np.ones((5, 13, 1, 1)) * 10, "surface": np.ones((4, 1, 1)) * 5}
        std = {"upper": np.ones((5, 13, 1, 1)) * 2, "surface": np.ones((4, 1, 1)) * 2}

        postprocessor = WeatherPostprocessor(mean=mean, std=std)

        upper = torch.zeros(1, 5, 13, 10, 10)  # normalized = 0
        surface = torch.zeros(1, 4, 10, 10)

        upper_denorm, surface_denorm = postprocessor.denormalize(upper, surface)

        # 0 * 2 + 10 = 10
        assert torch.allclose(upper_denorm, torch.ones_like(upper_denorm) * 10)
        assert torch.allclose(surface_denorm, torch.ones_like(surface_denorm) * 5)

    def test_denormalize_without_stats_skips(self):
        """測試無統計值時跳過反正規化。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        postprocessor = WeatherPostprocessor()

        upper = torch.randn(1, 5, 13, 10, 10)
        surface = torch.randn(1, 4, 10, 10)

        upper_out, surface_out = postprocessor.denormalize(upper, surface)

        torch.testing.assert_close(upper_out, upper)
        torch.testing.assert_close(surface_out, surface)

    def test_to_numpy(self):
        """測試 tensor 轉 numpy。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        postprocessor = WeatherPostprocessor()

        upper = torch.randn(1, 5, 13, 10, 10)
        surface = torch.randn(1, 4, 10, 10)

        upper_np, surface_np = postprocessor.to_numpy(upper, surface)

        assert upper_np.shape == (5, 13, 10, 10)  # squeezed
        assert surface_np.shape == (4, 10, 10)
        assert isinstance(upper_np, np.ndarray)

    def test_process_output(self):
        """測試完整輸出處理流程。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        postprocessor = WeatherPostprocessor()

        upper = torch.randn(1, 5, 13, 10, 10)
        surface = torch.randn(1, 4, 10, 10)

        upper_np, surface_np = postprocessor.process_output(
            upper, surface, denormalize=False
        )

        assert isinstance(upper_np, np.ndarray)
        assert isinstance(surface_np, np.ndarray)

    def test_save_netcdf_not_implemented(self, tmp_path):
        """測試 save_netcdf 拋出 NotImplementedError。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        postprocessor = WeatherPostprocessor()

        upper = np.random.randn(5, 13, 10, 10)
        surface = np.random.randn(4, 10, 10)

        with pytest.raises(NotImplementedError):
            postprocessor.save_netcdf(upper, surface, str(tmp_path / "out.nc"))


class TestPreprocessorPostprocessorRoundtrip:
    """前處理/後處理往返測試。"""

    def test_normalize_denormalize_roundtrip(self):
        """測試正規化與反正規化互為逆操作。"""
        from pangu.inference.preprocessing import WeatherPreprocessor
        from pangu.inference.postprocessing import WeatherPostprocessor

        mean = {
            "upper": np.random.randn(5, 13, 1, 1).astype(np.float32),
            "surface": np.random.randn(4, 1, 1).astype(np.float32),
        }
        std = {
            "upper": (np.abs(np.random.randn(5, 13, 1, 1)) + 0.1).astype(np.float32),
            "surface": (np.abs(np.random.randn(4, 1, 1)) + 0.1).astype(np.float32),
        }

        preprocessor = WeatherPreprocessor(mean=mean, std=std)
        postprocessor = WeatherPostprocessor(mean=mean, std=std)

        original_upper = torch.randn(1, 5, 13, 10, 10)
        original_surface = torch.randn(1, 4, 10, 10)

        # Forward: normalize
        normalized_upper, normalized_surface = preprocessor.normalize(
            original_upper, original_surface
        )

        # Backward: denormalize
        recovered_upper, recovered_surface = postprocessor.denormalize(
            normalized_upper, normalized_surface
        )

        torch.testing.assert_close(
            recovered_upper, original_upper, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            recovered_surface, original_surface, rtol=1e-5, atol=1e-5
        )
