"""配置管理模組，處理 YAML 配置的載入、驗證與解析。"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)


class DataConfig:
    """資料收集配置管理類別。"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器。

        Args:
            config_path: YAML 配置檔案路徑，若為 None 則使用空配置
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        """
        載入 YAML 配置檔案。

        Args:
            config_path: YAML 配置檔案路徑

        Raises:
            FileNotFoundError: 配置檔案不存在
            yaml.YAMLError: YAML 解析錯誤
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        logger.info(f"成功載入配置: {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        取得配置值，支援點號分隔的巢狀鍵。

        Args:
            key: 配置鍵，可使用 "." 分隔巢狀層級，如 "datasets.era5_surface"
            default: 預設值

        Returns:
            配置值或預設值
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        取得特定資料集的配置。

        Args:
            dataset_name: 資料集名稱

        Returns:
            資料集配置字典，若不存在則返回 None
        """
        datasets = self.config.get("datasets", {})
        return datasets.get(dataset_name)

    def validate_dataset_config(
        self, dataset_name: str, required_keys: List[str]
    ) -> bool:
        """
        驗證資料集配置是否包含必要的鍵。

        Args:
            dataset_name: 資料集名稱
            required_keys: 必要的配置鍵列表

        Returns:
            True if valid, False otherwise
        """
        dataset_config = self.get_dataset(dataset_name)

        if not dataset_config:
            logger.error(f"資料集配置不存在: {dataset_name}")
            return False

        missing_keys = [key for key in required_keys if key not in dataset_config]

        if missing_keys:
            logger.error(f"資料集 {dataset_name} 缺少必要配置: {missing_keys}")
            return False

        return True

    def get_common(self) -> Dict[str, Any]:
        """
        取得通用配置。

        Returns:
            通用配置字典
        """
        return self.config.get("common", {})

    def merge_with_common(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        將資料集配置與通用配置合併。

        Args:
            dataset_config: 資料集特定配置

        Returns:
            合併後的配置
        """
        common = self.get_common()
        merged = {**common, **dataset_config}
        return merged
