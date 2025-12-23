"""資料轉換工具。"""

import torch
import numpy as np
from typing import Tuple, Optional


def normalize(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    正規化資料到 N(0, 1)。

    Args:
        data: 輸入資料
        mean: 均值（若為 None 則計算）
        std: 標準差（若為 None 則計算）

    Returns:
        正規化後的資料
    """
    if mean is None:
        mean = np.mean(data, axis=tuple(range(data.ndim - 2, data.ndim)), keepdims=True)
    if std is None:
        std = np.std(data, axis=tuple(range(data.ndim - 2, data.ndim)), keepdims=True)

    return (data - mean) / (std + 1e-8)


def denormalize(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    反正規化資料。

    Args:
        data: 正規化的資料
        mean: 均值
        std: 標準差

    Returns:
        原始尺度的資料
    """
    return data * std + mean


def compute_statistics(
    data: np.ndarray, axis: Optional[Tuple[int, ...]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算資料的均值與標準差。

    Args:
        data: 輸入資料
        axis: 計算軸

    Returns:
        (mean, std)
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)

    return mean, std


def regrid(
    data: np.ndarray, target_shape: Tuple[int, int], method: str = "bilinear"
) -> np.ndarray:
    """
    網格重新採樣（簡化版）。

    Args:
        data: 輸入資料 (..., H, W)
        target_shape: 目標形狀 (H', W')
        method: 插值方法

    Returns:
        重新採樣的資料
    """
    # 簡化實作：使用 torch 的插值
    data_tensor = torch.from_numpy(data).float()

    # 確保有 batch 和 channel 維度
    original_shape = data_tensor.shape
    if len(original_shape) == 2:
        data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 3:
        data_tensor = data_tensor.unsqueeze(0)

    # 插值
    mode = "bilinear" if method == "bilinear" else "nearest"
    resampled = torch.nn.functional.interpolate(
        data_tensor,
        size=target_shape,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # 恢復原始維度
    if len(original_shape) == 2:
        resampled = resampled.squeeze(0).squeeze(0)
    elif len(original_shape) == 3:
        resampled = resampled.squeeze(0)

    return resampled.numpy()


class StandardScaler:
    """標準化縮放器。"""

    def __init__(self):
        """初始化縮放器。"""
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        """
        擬合縮放器。

        Args:
            data: 訓練資料

        Returns:
            self
        """
        self.mean, self.std = compute_statistics(data)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        轉換資料。

        Args:
            data: 輸入資料

        Returns:
            正規化的資料
        """
        if self.mean is None or self.std is None:
            raise ValueError("須先呼叫 fit()")

        return normalize(data, self.mean, self.std)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        反轉換資料。

        Args:
            data: 正規化的資料

        Returns:
            原始尺度的資料
        """
        if self.mean is None or self.std is None:
            raise ValueError("須先呼叫 fit()")

        return denormalize(data, self.mean, self.std)
