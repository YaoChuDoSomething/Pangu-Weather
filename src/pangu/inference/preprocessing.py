"""推論前處理模組。"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WeatherPreprocessor:
    """氣象資料前處理器。"""

    def __init__(
        self,
        mean: Optional[Dict[str, np.ndarray]] = None,
        std: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        初始化前處理器。

        Args:
            mean: 均值字典 {'upper': array, 'surface': array}
            std: 標準差字典 {'upper': array, 'surface': array}
        """
        self.mean = mean
        self.std = std

    def normalize(
        self, upper: torch.Tensor, surface: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        正規化輸入資料。

        Args:
            upper: (B, 5, 13, H, W) 上層變數
            surface: (B, 4, H, W) 地表變數

        Returns:
            正規化後的 (upper, surface)
        """
        if self.mean is None or self.std is None:
            logger.warning("統計資訊未設定，跳過正規化")
            return upper, surface

        # 正規化上層
        upper_mean = torch.from_numpy(self.mean["upper"]).to(upper.device)
        upper_std = torch.from_numpy(self.std["upper"]).to(upper.device)
        upper_normalized = (upper - upper_mean) / upper_std

        # 正規化地表
        surface_mean = torch.from_numpy(self.mean["surface"]).to(surface.device)
        surface_std = torch.from_numpy(self.std["surface"]).to(surface.device)
        surface_normalized = (surface - surface_mean) / surface_std

        return upper_normalized, surface_normalized

    def to_tensor(
        self, upper: np.ndarray, surface: np.ndarray, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        轉換 numpy array 為 tensor。

        Args:
            upper: (5, 13, H, W) numpy array
            surface: (4, H, W) numpy array
            device: 目標設備

        Returns:
            (upper_tensor, surface_tensor)
        """
        upper_tensor = torch.from_numpy(upper).float().unsqueeze(0).to(device)
        surface_tensor = torch.from_numpy(surface).float().unsqueeze(0).to(device)

        return upper_tensor, surface_tensor

    def prepare_input(
        self,
        upper: np.ndarray,
        surface: np.ndarray,
        device: str = "cpu",
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的輸入準備流程。

        Args:
            upper: numpy array
            surface: numpy array
            device: 設備
            normalize: 是否正規化

        Returns:
            準備好的 tensors
        """
        # 轉換為 tensor
        upper_t, surface_t = self.to_tensor(upper, surface, device)

        # 正規化
        if normalize:
            upper_t, surface_t = self.normalize(upper_t, surface_t)

        return upper_t, surface_t

    def load_stats(self, stats_path: str) -> None:
        """
        載入統計資訊。

        Args:
            stats_path: NPZ 檔案路徑
        """
        stats = np.load(stats_path)
        self.mean = {"upper": stats["upper_mean"], "surface": stats["surface_mean"]}
        self.std = {"upper": stats["upper_std"], "surface": stats["surface_std"]}
        logger.info(f"載入統計資訊: {stats_path}")
