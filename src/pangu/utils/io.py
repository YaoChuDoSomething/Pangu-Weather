"""I/O 工具模組。"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class NetCDFReader:
    """NetCDF 檔案讀取器（需要 xarray）。"""

    def __init__(self):
        """初始化讀取器。"""
        try:
            import xarray as xr

            self.xr = xr
        except ImportError:
            logger.error("需要安裝 xarray: uv add xarray")
            raise

    def read(
        self, file_path: str, variables: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        讀取 NetCDF 檔案。

        Args:
            file_path: 檔案路徑
            variables: 要讀取的變數列表

        Returns:
            變數字典
        """
        ds = self.xr.open_dataset(file_path)

        if variables is None:
            variables = list(ds.data_vars)

        data = {var: ds[var].values for var in variables if var in ds}

        logger.info(f"讀取 NetCDF: {file_path}, 變數: {list(data.keys())}")
        return data

    def close(self):
        """關閉資源。"""
        pass


class NetCDFWriter:
    """NetCDF 檔案寫入器。"""

    def __init__(self):
        """初始化寫入器。"""
        try:
            import xarray as xr

            self.xr = xr
        except ImportError:
            logger.error("需要安裝 xarray: uv add xarray")
            raise

    def write(
        self,
        data: Dict[str, np.ndarray],
        output_path: str,
        coords: Optional[Dict] = None,
        attrs: Optional[Dict] = None,
    ) -> None:
        """
        寫入 NetCDF 檔案。

        Args:
            data: 資料字典
            output_path: 輸出路徑
            coords: 座標字典
            attrs: 屬性字典
        """
        # 建立 Dataset
        ds = self.xr.Dataset(
            {name: (["var", "level", "lat", "lon"], arr) for name, arr in data.items()},
            coords=coords,
            attrs=attrs or {},
        )

        # 寫入檔案
        ds.to_netcdf(output_path)
        logger.info(f"寫入 NetCDF: {output_path}")


class NPYReader:
    """NPY/NPZ 檔案讀取器。"""

    @staticmethod
    def read(file_path: str) -> Any:
        """
        讀取 npy/npz 檔案。

        Args:
            file_path: 檔案路徑

        Returns:
            numpy array 或字典
        """
        path = Path(file_path)

        if path.suffix == ".npy":
            data = np.load(file_path)
        elif path.suffix == ".npz":
            data = dict(np.load(file_path))
        else:
            raise ValueError(f"不支援的檔案格式: {path.suffix}")

        logger.info(f"讀取 {path.suffix}: {file_path}")
        return data


class NPYWriter:
    """NPY/NPZ 檔案寫入器。"""

    @staticmethod
    def write(data: Any, output_path: str, compressed: bool = True) -> None:
        """
        寫入 npy/npz 檔案。

        Args:
            data: numpy array 或字典
            output_path: 輸出路徑
            compressed: 是否壓縮（僅對 npz 有效）
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, dict):
            if compressed:
                np.savez_compressed(output_path, **data)
            else:
                np.savez(output_path, **data)
        else:
            np.save(output_path, data)

        logger.info(f"寫入 {path.suffix}: {output_path}")
