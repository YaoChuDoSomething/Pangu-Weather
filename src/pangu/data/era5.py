"""ERA5 資料下載器，透過 CDS API 下載再分析資料。"""

import logging
import time
from typing import Any, Dict, Optional, Callable

from pangu.data.base import DataDownloaderProtocol, DownloadError, ConfigValidationError

logger = logging.getLogger(__name__)

try:
    import cdsapi
except ImportError:
    logger.error("無法匯入 cdsapi。請確認已在環境中安裝：pip install cdsapi")
    raise


class ERA5Downloader(DataDownloaderProtocol):
    """ERA5 再分析資料下載器。"""

    def __init__(
        self, api_key: Optional[str] = None, max_retries: int = 3, retry_delay: int = 5
    ):
        """
        初始化 CDS API 客戶端。

        Args:
            api_key: 可選的 API 金鑰。若為 None，則從 .cdsapirc 或環境變數讀取
            max_retries: 最大重試次數
            retry_delay: 重試間隔（秒）
        """
        self.client = cdsapi.Client(key=api_key)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def download(
        self,
        config: Dict[str, Any],
        output_path: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        """
        下載 ERA5 資料。

        Args:
            config: 下載配置，需包含 dataset_name 與 request 參數
            output_path: 輸出檔案路徑
            progress_callback: 進度回調函數
            **kwargs: 傳遞給 CDS API 的額外參數

        Raises:
            ConfigValidationError: 配置無效
            DownloadError: 下載失敗
        """
        if not self.validate_config(config):
            raise ConfigValidationError(f"ERA5 配置無效: {config}")

        self.ensure_output_dir(output_path)

        dataset_name = config["dataset_name"]
        request_params = self._build_request(config)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"下載 {dataset_name} 到 {output_path} (嘗試 {attempt}/{self.max_retries})"
                )

                if progress_callback:
                    progress_callback(f"開始下載 {dataset_name}")

                self.client.retrieve(dataset_name, request_params, output_path)

                logger.info(f"下載完成: {output_path}")
                if progress_callback:
                    progress_callback(f"完成: {output_path}")

                return

            except Exception as e:
                logger.warning(f"下載失敗 (嘗試 {attempt}/{self.max_retries}): {e}")

                if attempt < self.max_retries:
                    logger.info(f"等待 {self.retry_delay} 秒後重試...")
                    time.sleep(self.retry_delay)
                else:
                    raise DownloadError(
                        f"下載 {dataset_name} 失敗，已嘗試 {self.max_retries} 次: {e}"
                    )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證 ERA5 配置。

        Args:
            config: 配置字典

        Returns:
            True if valid
        """
        required_keys = ["dataset_name"]

        for key in required_keys:
            if key not in config:
                logger.error(f"ERA5 配置缺少必要鍵: {key}")
                return False

        return True

    def _build_request(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        建立 CDS API 請求參數。

        Args:
            config: 原始配置

        Returns:
            CDS API 請求參數字典
        """
        # 排除非 CDS API 的鍵
        excluded_keys = {"type", "dataset_name"}
        request = {k: v for k, v in config.items() if k not in excluded_keys}

        return request
