"""推論後處理模組。"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WeatherPostprocessor:
    """氣象資料後處理器。"""

    def __init__(
        self,
        mean: Optional[Dict[str, np.ndarray]] = None,
        std: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        初始化後處理器。

        Args:
            mean: 均值字典
            std: 標準差字典
        """
        self.mean = mean
        self.std = std

    def denormalize(
        self, upper: torch.Tensor, surface: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反正規化輸出資料。

        Args:
            upper: (B, 5, 13, H, W) 正規化的上層變數
            surface: (B, 4, H, W) 正規化的地表變數

        Returns:
            反正規化後的 (upper, surface)
        """
        if self.mean is None or self.std is None:
            logger.warning("統計資訊未設定，跳過反正規化")
            return upper, surface

        # 反正規化上層
        upper_mean = torch.from_numpy(self.mean["upper"]).to(upper.device)
        upper_std = torch.from_numpy(self.std["upper"]).to(upper.device)
        upper_denorm = upper * upper_std + upper_mean

        # 反正規化地表
        surface_mean = torch.from_numpy(self.mean["surface"]).to(surface.device)
        surface_std = torch.from_numpy(self.std["surface"]).to(surface.device)
        surface_denorm = surface * surface_std + surface_mean

        return upper_denorm, surface_denorm

    def to_numpy(
        self, upper: torch.Tensor, surface: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        轉換 tensor 為 numpy array。

        Args:
            upper: (B, 5, 13, H, W) tensor
            surface: (B, 4, H, W) tensor

        Returns:
            (upper_np, surface_np)
        """
        upper_np = upper.squeeze(0).cpu().numpy()
        surface_np = surface.squeeze(0).cpu().numpy()

        return upper_np, surface_np

    def process_output(
        self, upper: torch.Tensor, surface: torch.Tensor, denormalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整的輸出處理流程。

        Args:
            upper: 模型輸出 tensor
            surface: 模型輸出 tensor
            denormalize: 是否反正規化

        Returns:
            處理後的 numpy arrays
        """
        # 反正規化
        if denormalize:
            upper, surface = self.denormalize(upper, surface)

        # 轉換為 numpy
        upper_np, surface_np = self.to_numpy(upper, surface)

        return upper_np, surface_np

    def save_netcdf(
        self,
        upper: np.ndarray,
        surface: np.ndarray,
        output_path: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        儲存為 NetCDF 格式（待實作）。

        Args:
            upper: (5, 13, H, W) array
            surface: (4, H, W) array
            output_path: 輸出路徑
            metadata: 元資料
        """
        # TODO: 使用 xarray 實作
        logger.info(f"NetCDF 輸出待實作: {output_path}")
        raise NotImplementedError("NetCDF 輸出需要 xarray")

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
