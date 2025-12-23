"""GFS 資料下載器，透過 GDEX API 下載全球預報系統資料。"""

import logging
import time
from typing import Any, Dict, Optional, Protocol

from pangu.data.base import DataDownloaderProtocol, DownloadError, ConfigValidationError

logger = logging.getLogger(__name__)


class GdexClientProtocol(Protocol):
    def submit(self, file_path: str) -> Dict[str, Any]: ...
    def submit_json(self, data: Dict[str, Any]) -> Dict[str, Any]: ...
    def get_status(self, request_id: str) -> Dict[str, Any]: ...
    def download(self, request_id: str, out_dir: str) -> None: ...


class GFSGDEXDownloader(DataDownloaderProtocol):
    """GFS 資料下載器，使用 GDEX API。"""

    def __init__(
        self, client: Optional[GdexClientProtocol] = None, poll_interval: int = 10
    ):
        """
        初始化下載器。

        Args:
            client: 可選的 GDEX client 實體（供測試使用）
            poll_interval: 狀態輪詢間隔（秒）
        """
        self.client = client or self._load_default_client()
        self.poll_interval = poll_interval

    def _load_default_client(self) -> GdexClientProtocol:
        try:
            from gdex_api_client import gdex_client

            return gdex_client
        except ImportError:
            logger.error("gdex_api_client not found.")
            raise

    def download(self, config: Dict[str, Any], output_path: str, **kwargs) -> None:
        """
        執行完整下載流程: Submit → Wait → Download。

        Args:
            config: 下載配置（GDEX 請求 payload）
            output_path: 輸出目錄
            **kwargs: 額外參數

        Raises:
            ConfigValidationError: 配置無效
            DownloadError: 下載失敗
        """
        if not self.validate_config(config):
            raise ConfigValidationError(f"GDEX 配置無效: {config}")

        self.ensure_output_dir(output_path)

        request_id = self.submit(config)
        if self._wait_for_completion(request_id):
            self._download_files(request_id, output_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證 GDEX 配置。

        Args:
            config: 配置字典

        Returns:
            True if valid
        """
        # GDEX 接受 dict 或 file path
        if isinstance(config, dict):
            return True
        if isinstance(config, str):
            return True

        logger.error(f"GDEX 配置必須是 dict 或 str，收到: {type(config)}")
        return False

    def submit(self, payload: Any) -> str:
        """提交請求並返回 Request ID。"""
        logger.info("提交 GDEX 請求...")

        # 根據 payload 類型分派
        if isinstance(payload, dict):
            res = self.client.submit_json(payload)
        else:
            res = self.client.submit(str(payload))

        if not self._is_submission_successful(res):
            raise DownloadError(f"提交失敗: {res}")

        req_id = res["data"]["request_index"]
        logger.info(f"請求已提交，ID: {req_id}")
        return req_id

    def _wait_for_completion(self, request_id: str) -> bool:
        """輪詢狀態直到完成或錯誤。"""
        while True:
            status = self._check_status(request_id)
            logger.info(f"狀態: {status}")

            if status == "Completed":
                return True
            if status == "Error":
                logger.error("GDEX 請求失敗")
                return False

            time.sleep(self.poll_interval)

    def _check_status(self, request_id: str) -> str:
        res = self.client.get_status(request_id)
        if "data" in res and res["data"]:
            return res["data"].get("status", "Unknown")
        return "Unknown"

    def _download_files(self, request_id: str, output_dir: str) -> None:
        logger.info("下載檔案...")
        self.client.download(request_id, out_dir=output_dir)
        logger.info("下載完成")

    def _is_submission_successful(self, res: Dict[str, Any]) -> bool:
        return "data" in res and "request_index" in res["data"]
