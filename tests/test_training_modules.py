"""測試訓練模組的基本功能。"""

import torch
from pangu.training import (
    WeatherDataset,
    PanguLoss,
    WeatherMetrics,
    TrainingConfig,
)


def test_weather_dataset():
    """測試資料集建立。"""
    dataset = WeatherDataset(data_dir="data/train", start_year=2010, end_year=2011)
    assert dataset is not None
    print(f"✓ WeatherDataset 建立成功，長度: {len(dataset)}")


def test_pangu_loss():
    """測試 Pangu 損失函數。"""
    criterion = PanguLoss(surface_weight=0.25)

    # 建立測試資料
    B = 2
    output_upper = torch.randn(B, 5, 13, 721, 1440)
    target_upper = torch.randn(B, 5, 13, 721, 1440)
    output_surface = torch.randn(B, 4, 721, 1440)
    target_surface = torch.randn(B, 4, 721, 1440)

    # 計算損失
    loss = criterion(output_upper, target_upper, output_surface, target_surface)

    assert loss.item() >= 0
    print(f"✓ PanguLoss 計算成功，損失值: {loss.item():.6f}")


def test_weather_metrics():
    """測試氣象指標。"""
    metrics = WeatherMetrics()

    # 建立測試資料
    B = 2
    pred_upper = torch.randn(B, 5, 13, 721, 1440)
    target_upper = torch.randn(B, 5, 13, 721, 1440)
    pred_surface = torch.randn(B, 4, 721, 1440)
    target_surface = torch.randn(B, 4, 721, 1440)

    # 更新指標
    batch_metrics = metrics.update(
        pred_upper, target_upper, pred_surface, target_surface
    )

    assert "rmse_upper" in batch_metrics
    assert "mae_surface" in batch_metrics

    # 計算平均指標
    avg_metrics = metrics.compute()

    print("✓ WeatherMetrics 計算成功:")
    print(f"  RMSE (upper): {avg_metrics['rmse_upper']:.4f}")
    print(f"  MAE (surface): {avg_metrics['mae_surface']:.4f}")
    print(f"  ACC (upper): {avg_metrics['acc_upper']:.4f}")


def test_training_config():
    """測試訓練配置。"""
    config = TrainingConfig(epochs=10, batch_size=2, learning_rate=1e-4)

    assert config.epochs == 10
    assert config.batch_size == 2
    assert config.learning_rate == 1e-4

    print("✓ TrainingConfig 建立成功")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")


if __name__ == "__main__":
    print("=" * 60)
    print("開始測試訓練模組")
    print("=" * 60)
    print()

    try:
        test_pangu_loss()
        print()

        test_weather_metrics()
        print()

        test_training_config()
        print()

        test_weather_dataset()
        print()

        print("=" * 60)
        print("✓ 所有測試通過！")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ 測試失敗: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        raise
