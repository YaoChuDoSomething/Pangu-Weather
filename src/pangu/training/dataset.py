"""氣象資料集類別，用於訓練盤古模型。"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WeatherDataset(Dataset):
    """
    氣象資料集基類。

    根據 pseudocode.py，需要載入時間 t 的資料作為輸入，
    時間 t+lead_time 的資料作為目標。
    """

    def __init__(
        self,
        data_dir: str,
        start_year: int = 1979,
        end_year: int = 2017,
        lead_time_hours: int = 6,
        file_pattern: str = "era5_*.nc",
    ):
        """
        初始化氣象資料集。

        Args:
            data_dir: 資料目錄
            start_year: 起始年份
            end_year: 結束年份
            lead_time_hours: 預報時距（小時）
            file_pattern: 檔案匹配模式
        """
        self.data_dir = Path(data_dir)
        self.start_year = start_year
        self.end_year = end_year
        self.lead_time_hours = lead_time_hours
        self.file_pattern = file_pattern

        # 載入資料檔案列表
        self.data_files = self._discover_files()

        if len(self.data_files) == 0:
            logger.warning(f"在 {data_dir} 中未找到符合 {file_pattern} 的檔案")

        # 統計資訊（mean/std）用於正規化
        self.stats = None

    def _discover_files(self) -> List[Path]:
        """發現資料檔案。"""
        files = sorted(self.data_dir.glob(self.file_pattern))
        logger.info(f"發現 {len(files)} 個資料檔案")
        return files

    def load_statistics(self, stats_path: str) -> None:
        """
        載入統計資訊（mean, std）。

        Args:
            stats_path: 統計檔案路徑（NPZ 格式）
        """
        self.stats = np.load(stats_path)
        logger.info(f"載入統計資訊: {stats_path}")

    def __len__(self) -> int:
        """資料集大小。"""
        # 每個檔案可能包含多個時間步
        # 這裡簡化為檔案數量（實際應根據時間步計算）
        return max(0, len(self.data_files) - 1)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        取得訓練樣本。

        Args:
            idx: 索引

        Returns:
            (input, input_surface, target, target_surface)
            - input: (5, 13, H, W) 上層變數
            - input_surface: (4, H, W) 地表變數
            - target: (5, 13, H, W) 目標上層變數
            - target_surface: (4, H, W) 目標地表變數
        """
        # TODO: 實作實際的資料讀取邏輯
        # 這裡提供佔位實作，實際需要從 NetCDF/GRIB 讀取

        # Placeholder dimensions (根據 Pangu-Weather 規格)
        # 上層: 5 變數 x 13 氣壓層 x 721 lat x 1440 lon
        # 地表: 4 變數 x 721 lat x 1440 lon
        input_upper = torch.randn(5, 13, 721, 1440)
        input_surface = torch.randn(4, 721, 1440)

        target_upper = torch.randn(5, 13, 721, 1440)
        target_surface = torch.randn(4, 721, 1440)

        # 正規化（如果有統計資訊）
        if self.stats is not None:
            input_upper = self._normalize(input_upper, "upper")
            input_surface = self._normalize(input_surface, "surface")
            target_upper = self._normalize(target_upper, "upper")
            target_surface = self._normalize(target_surface, "surface")

        return input_upper, input_surface, target_upper, target_surface

    def _normalize(self, data: torch.Tensor, data_type: str) -> torch.Tensor:
        """
        正規化資料到 N(0, 1)。

        Args:
            data: 輸入資料
            data_type: 'upper' 或 'surface'

        Returns:
            正規化後的資料
        """
        if self.stats is None:
            return data

        mean_key = f"{data_type}_mean"
        std_key = f"{data_type}_std"

        if mean_key in self.stats and std_key in self.stats:
            mean = torch.from_numpy(self.stats[mean_key]).float()
            std = torch.from_numpy(self.stats[std_key]).float()
            return (data - mean) / std

        return data


class PanguDataset(WeatherDataset):
    """
    盤古專用資料集。

    特定於 Pangu-Weather 的變數配置：
    - 上層: Z, Q, T, U, V (13個氣壓層: 1000-50 hPa)
    - 地表: MSLP, U10, V10, T2M
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 變數名稱（供參考）
        self.upper_vars = ["Z", "Q", "T", "U", "V"]
        self.pressure_levels = [
            1000,
            925,
            850,
            700,
            600,
            500,
            400,
            300,
            250,
            200,
            150,
            100,
            50,
        ]
        self.surface_vars = ["MSLP", "U10", "V10", "T2M"]

    def _load_netcdf_sample(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        從 NetCDF 載入單一樣本（待實作）。

        Args:
            file_path: NetCDF 檔案路徑

        Returns:
            (upper_data, surface_data)
        """
        # TODO: 使用 xarray 或 netCDF4 讀取
        # import xarray as xr
        # ds = xr.open_dataset(file_path)
        # upper = ds[self.upper_vars].values
        # surface = ds[self.surface_vars].values
        # return upper, surface

        raise NotImplementedError("NetCDF 讀取邏輯待實作")
