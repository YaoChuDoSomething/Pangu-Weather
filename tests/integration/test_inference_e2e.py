"""端到端推論整合測試。

測試完整的推論流程：input → preprocess → model → postprocess → output。
"""

import pytest
import torch
import numpy as np


class TestInferenceE2E:
    """端到端推論測試。"""

    @pytest.fixture
    def model(self):
        """建立測試用模型。"""
        from pangu.models import create_pangu_model

        model = create_pangu_model()
        model.eval()
        return model

    @pytest.fixture
    def preprocessor(self):
        """建立前處理器。"""
        from pangu.inference.preprocessing import WeatherPreprocessor

        return WeatherPreprocessor()

    @pytest.fixture
    def postprocessor(self):
        """建立後處理器。"""
        from pangu.inference.postprocessing import WeatherPostprocessor

        return WeatherPostprocessor()

    def test_full_pipeline_shapes(self, model, preprocessor, postprocessor):
        """測試完整 pipeline 的輸入輸出形狀一致。"""
        # 準備輸入
        upper_np = np.random.randn(5, 13, 721, 1440).astype(np.float32)
        surface_np = np.random.randn(4, 721, 1440).astype(np.float32)

        # 前處理
        upper_t, surface_t = preprocessor.prepare_input(
            upper_np, surface_np, normalize=False
        )

        assert upper_t.shape == (1, 5, 13, 721, 1440)
        assert surface_t.shape == (1, 4, 721, 1440)

        # 模型推論
        with torch.no_grad():
            output_upper, output_surface = model(upper_t, surface_t)

        assert output_upper.shape == (1, 5, 13, 721, 1440)
        assert output_surface.shape == (1, 4, 721, 1440)

        # 後處理
        upper_out, surface_out = postprocessor.process_output(
            output_upper, output_surface, denormalize=False
        )

        assert upper_out.shape == (5, 13, 721, 1440)
        assert surface_out.shape == (4, 721, 1440)

    def test_pipeline_output_is_finite(self, model, preprocessor, postprocessor):
        """測試輸出不含 NaN 或 Inf。"""
        upper_np = np.random.randn(5, 13, 721, 1440).astype(np.float32)
        surface_np = np.random.randn(4, 721, 1440).astype(np.float32)

        upper_t, surface_t = preprocessor.prepare_input(
            upper_np, surface_np, normalize=False
        )

        with torch.no_grad():
            output_upper, output_surface = model(upper_t, surface_t)

        upper_out, surface_out = postprocessor.process_output(
            output_upper, output_surface, denormalize=False
        )

        assert np.isfinite(upper_out).all(), "Upper output contains NaN/Inf"
        assert np.isfinite(surface_out).all(), "Surface output contains NaN/Inf"


class TestInferenceFailureCases:
    """推論失敗案例測試。"""

    def test_wrong_input_shape_raises(self):
        """測試錯誤輸入形狀應拋出錯誤。"""
        from pangu.models import create_pangu_model

        model = create_pangu_model()
        model.eval()

        # 錯誤的形狀
        wrong_upper = torch.randn(1, 3, 13, 721, 1440)  # 應該是 5 變數
        correct_surface = torch.randn(1, 4, 721, 1440)

        with pytest.raises(Exception):  # 可能是 RuntimeError 或 ValueError
            with torch.no_grad():
                model(wrong_upper, correct_surface)

    def test_batch_size_one_works(self):
        """測試 batch size = 1 正常運作。"""
        from pangu.models import create_pangu_model

        model = create_pangu_model()
        model.eval()

        upper = torch.randn(1, 5, 13, 721, 1440)
        surface = torch.randn(1, 4, 721, 1440)

        with torch.no_grad():
            output_upper, output_surface = model(upper, surface)

        assert output_upper.shape[0] == 1
        assert output_surface.shape[0] == 1
