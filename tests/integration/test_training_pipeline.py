"""訓練 pipeline 整合測試。

測試完整的訓練流程：dataset → dataloader → model → loss → optimizer。
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch


class TestTrainingPipeline:
    """訓練 pipeline 測試。"""

    @pytest.fixture
    def model(self):
        """建立測試用模型。"""
        from pangu.models import create_pangu_model

        model = create_pangu_model()
        model.train()
        return model

    @pytest.fixture
    def loss_fn(self):
        """建立損失函數。"""
        from pangu.training import PanguLoss

        return PanguLoss(surface_weight=0.25)

    @pytest.fixture
    def metrics(self):
        """建立指標計算器。"""
        from pangu.training import WeatherMetrics

        return WeatherMetrics()

    def test_single_training_step(self, model, loss_fn):
        """測試單次訓練步驟。"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 準備資料
        input_upper = torch.randn(1, 5, 13, 721, 1440)
        input_surface = torch.randn(1, 4, 721, 1440)
        target_upper = torch.randn(1, 5, 13, 721, 1440)
        target_surface = torch.randn(1, 4, 721, 1440)

        # Forward pass
        output_upper, output_surface = model(input_upper, input_surface)

        # Compute loss
        loss = loss_fn(output_upper, target_upper, output_surface, target_surface)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
        )
        assert has_grad, "No gradients computed"

        # Optimizer step (just verify no error)
        optimizer.step()

        assert loss.item() >= 0

    def test_metrics_accumulation(self, model, loss_fn, metrics):
        """測試指標累積。"""
        input_upper = torch.randn(1, 5, 13, 721, 1440)
        input_surface = torch.randn(1, 4, 721, 1440)
        target_upper = torch.randn(1, 5, 13, 721, 1440)
        target_surface = torch.randn(1, 4, 721, 1440)

        with torch.no_grad():
            output_upper, output_surface = model(input_upper, input_surface)

        # Update metrics
        batch_metrics = metrics.update(
            output_upper, target_upper, output_surface, target_surface
        )

        assert "rmse_upper" in batch_metrics
        assert "mae_surface" in batch_metrics

        # Compute average
        avg_metrics = metrics.compute()

        assert "rmse_upper" in avg_metrics
        assert avg_metrics["rmse_upper"] >= 0


class TestTrainingConfig:
    """訓練配置測試。"""

    def test_config_defaults(self):
        """測試配置預設值。"""
        from pangu.training import TrainingConfig

        config = TrainingConfig()

        assert config.epochs > 0
        assert config.batch_size > 0
        assert config.learning_rate > 0

    def test_config_custom(self):
        """測試自訂配置。"""
        from pangu.training import TrainingConfig

        config = TrainingConfig(epochs=50, batch_size=4, learning_rate=5e-5)

        assert config.epochs == 50
        assert config.batch_size == 4
        assert config.learning_rate == 5e-5


class TestDataloader:
    """Dataloader 測試。"""

    def test_dataloader_creation(self):
        """測試 dataloader 建立。"""
        from pangu.training import create_dataloader, TrainingConfig

        config = TrainingConfig(batch_size=2)

        # 使用 mock dataset
        with patch("pangu.training.dataloader.WeatherDataset") as MockDataset:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)
            MockDataset.return_value = mock_dataset

            dataloader = create_dataloader(config, data_dir="data/train")

            # Verify dataloader was created
            assert dataloader is not None


class TestLossComponents:
    """損失函數元件測試。"""

    def test_upper_loss(self):
        """測試上層損失計算。"""
        from pangu.training import PanguLoss

        loss_fn = PanguLoss(surface_weight=0.0)  # 只計算 upper loss

        output_upper = torch.randn(1, 5, 13, 721, 1440)
        target_upper = torch.randn(1, 5, 13, 721, 1440)
        output_surface = torch.zeros(1, 4, 721, 1440)
        target_surface = torch.zeros(1, 4, 721, 1440)

        loss = loss_fn(output_upper, target_upper, output_surface, target_surface)

        assert loss.item() > 0

    def test_surface_loss(self):
        """測試地表損失計算。"""
        from pangu.training import PanguLoss

        loss_fn = PanguLoss(surface_weight=1.0)

        output_upper = torch.zeros(1, 5, 13, 721, 1440)
        target_upper = torch.zeros(1, 5, 13, 721, 1440)
        output_surface = torch.randn(1, 4, 721, 1440)
        target_surface = torch.randn(1, 4, 721, 1440)

        loss = loss_fn(output_upper, target_upper, output_surface, target_surface)

        assert loss.item() > 0

    def test_combined_loss(self):
        """測試組合損失。"""
        from pangu.training import PanguLoss

        loss_fn = PanguLoss(surface_weight=0.25)

        output_upper = torch.randn(1, 5, 13, 721, 1440)
        target_upper = torch.randn(1, 5, 13, 721, 1440)
        output_surface = torch.randn(1, 4, 721, 1440)
        target_surface = torch.randn(1, 4, 721, 1440)

        loss = loss_fn(output_upper, target_upper, output_surface, target_surface)

        assert loss.item() > 0
