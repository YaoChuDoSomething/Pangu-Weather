"""氣象評估指標。"""

import torch
import numpy as np
from typing import Dict, Optional


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    計算 RMSE (Root Mean Square Error)。

    Args:
        pred: 預測值
        target: 目標值

    Returns:
        RMSE 值
    """
    mse = torch.mean((pred - target) ** 2)
    return torch.sqrt(mse).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    計算 MAE (Mean Absolute Error)。

    Args:
        pred: 預測值
        target: 目標值

    Returns:
        MAE 值
    """
    return torch.mean(torch.abs(pred - target)).item()


def bias(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    計算偏差 (Bias)。

    Args:
        pred: 預測值
        target: 目標值

    Returns:
        平均偏差
    """
    return torch.mean(pred - target).item()


def acc(
    pred: torch.Tensor, target: torch.Tensor, climatology: Optional[torch.Tensor] = None
) -> float:
    """
    計算異常相關係數 (Anomaly Correlation Coefficient)。

    ACC 是氣象預報中的重要指標，衡量預測的空間模式與實際的相似度。

    Args:
        pred: 預測值
        target: 目標值
        climatology: 氣候平均值（若為 None 則使用 target 的平均）

    Returns:
        ACC 值 (範圍 -1 到 1)
    """
    if climatology is None:
        climatology = torch.mean(target)

    pred_anomaly = pred - climatology
    target_anomaly = target - climatology

    numerator = torch.sum(pred_anomaly * target_anomaly)
    denominator = torch.sqrt(torch.sum(pred_anomaly**2) * torch.sum(target_anomaly**2))

    if denominator < 1e-8:
        return 0.0

    return (numerator / denominator).item()


class WeatherMetrics:
    """氣象評估指標計算器。"""

    def __init__(self):
        """初始化指標計算器。"""
        self.reset()

    def reset(self) -> None:
        """重置累積的指標。"""
        self.metrics = {
            "rmse_upper": [],
            "rmse_surface": [],
            "mae_upper": [],
            "mae_surface": [],
            "acc_upper": [],
            "acc_surface": [],
            "bias_upper": [],
            "bias_surface": [],
        }

    def update(
        self,
        pred_upper: torch.Tensor,
        target_upper: torch.Tensor,
        pred_surface: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> Dict[str, float]:
        """
        更新指標並返回當前批次的指標值。

        Args:
            pred_upper: 預測上層變數
            target_upper: 目標上層變數
            pred_surface: 預測地表變數
            target_surface: 目標地表變數

        Returns:
            當前批次的指標字典
        """
        batch_metrics = {}

        # 上層指標
        batch_metrics["rmse_upper"] = rmse(pred_upper, target_upper)
        batch_metrics["mae_upper"] = mae(pred_upper, target_upper)
        batch_metrics["acc_upper"] = acc(pred_upper, target_upper)
        batch_metrics["bias_upper"] = bias(pred_upper, target_upper)

        # 地表指標
        batch_metrics["rmse_surface"] = rmse(pred_surface, target_surface)
        batch_metrics["mae_surface"] = mae(pred_surface, target_surface)
        batch_metrics["acc_surface"] = acc(pred_surface, target_surface)
        batch_metrics["bias_surface"] = bias(pred_surface, target_surface)

        # 累積
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)

        return batch_metrics

    def compute(self) -> Dict[str, float]:
        """
        計算累積的平均指標。

        Returns:
            平均指標字典
        """
        avg_metrics = {}

        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0

        return avg_metrics

    def get_summary(self) -> str:
        """
        取得指標摘要字串。

        Returns:
            格式化的指標摘要
        """
        avg_metrics = self.compute()

        summary = "Weather Metrics Summary:\n"
        summary += f"  Upper Air:\n"
        summary += f"    RMSE: {avg_metrics['rmse_upper']:.4f}\n"
        summary += f"    MAE:  {avg_metrics['mae_upper']:.4f}\n"
        summary += f"    ACC:  {avg_metrics['acc_upper']:.4f}\n"
        summary += f"    Bias: {avg_metrics['bias_upper']:.4f}\n"
        summary += f"  Surface:\n"
        summary += f"    RMSE: {avg_metrics['rmse_surface']:.4f}\n"
        summary += f"    MAE:  {avg_metrics['mae_surface']:.4f}\n"
        summary += f"    ACC:  {avg_metrics['acc_surface']:.4f}\n"
        summary += f"    Bias: {avg_metrics['bias_surface']:.4f}\n"

        return summary
