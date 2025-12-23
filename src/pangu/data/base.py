"""抽象基類定義資料下載器的統一介面。"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from pathlib import Path


class DataDownloaderProtocol(ABC):
    """資料下載器的抽象基類，定義統一介面。"""

    @abstractmethod
    def download(self, config: Dict[str, Any], output_path: str, **kwargs) -> None:
        """
        下載資料的主要方法。

        Args:
            config: 下載配置參數
            output_path: 輸出檔案路徑
            **kwargs: 額外參數

        Raises:
            ValueError: 配置參數無效
            RuntimeError: 下載失敗
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證配置參數是否有效。

        Args:
            config: 配置參數字典

        Returns:
            True if valid, False otherwise
        """
        pass

    def ensure_output_dir(self, output_path: str) -> Path:
        """
        確保輸出目錄存在。

        Args:
            output_path: 輸出檔案路徑

        Returns:
            Path object of the parent directory
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent


class DownloadError(Exception):
    """資料下載過程中發生的錯誤。"""

    pass


class ConfigValidationError(Exception):
    """配置驗證失敗的錯誤。"""

    pass
