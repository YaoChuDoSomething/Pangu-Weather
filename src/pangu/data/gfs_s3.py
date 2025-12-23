"""GFS 即時資料下載器，支援 AWS S3 存取（AWS CLI 或 Boto3）。"""

import os
import logging
import subprocess
from typing import Any, Dict

from pangu.data.base import DataDownloaderProtocol, DownloadError, ConfigValidationError

logger = logging.getLogger(__name__)


class GFSRealtimeDownloader(DataDownloaderProtocol):
    """GFS 即時資料下載器，從 AWS S3 取得資料。"""

    def __init__(self, bucket: str = "noaa-gfs-bdp-pds", use_boto3: bool = False):
        """
        初始化下載器。

        Args:
            bucket: S3 bucket 名稱
            use_boto3: 是否使用 Boto3（False 則使用 AWS CLI）
        """
        self.bucket = bucket
        self.use_boto3 = use_boto3
        self.s3_client = None

        if use_boto3:
            try:
                import boto3

                self.s3_client = boto3.client("s3")
                logger.info("使用 Boto3 客戶端")
            except ImportError:
                logger.warning("無法匯入 boto3，回退到 AWS CLI")
                self.use_boto3 = False

    def download(self, config: Dict[str, Any], output_path: str, **kwargs) -> None:
        """
        下載 GFS 預報檔案序列。

        Args:
            config: 配置，需包含 date_str, cycle, forecast_hours
            output_path: 輸出目錄
            **kwargs: 額外參數

        Raises:
            ConfigValidationError: 配置無效
            DownloadError: 下載失敗
        """
        if not self.validate_config(config):
            raise ConfigValidationError(f"GFS S3 配置無效: {config}")

        self.ensure_output_dir(output_path)

        date_str = config["date_str"]
        cycle = config["cycle"]
        forecast_hours = config.get("forecast_hours", [0])
        res = config.get("resolution", "0p25")

        for fh in forecast_hours:
            s3_key = self.get_gfs_path(date_str, cycle, fh, res)
            filename = os.path.basename(s3_key)
            local_path = os.path.join(output_path, filename)

            self._download_file(s3_key, local_path)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        驗證 GFS S3 配置。

        Args:
            config: 配置字典

        Returns:
            True if valid
        """
        required_keys = ["date_str", "cycle"]

        for key in required_keys:
            if key not in config:
                logger.error(f"GFS S3 配置缺少必要鍵: {key}")
                return False

        return True

    def _download_file(self, s3_key: str, local_path: str) -> None:
        """
        下載單一檔案。

        Args:
            s3_key: S3 物件鍵
            local_path: 本地檔案路徑
        """
        s3_uri = f"s3://{self.bucket}/{s3_key}"
        logger.info(f"下載 {s3_uri} 到 {local_path}...")

        try:
            if self.use_boto3 and self.s3_client:
                self.s3_client.download_file(self.bucket, s3_key, local_path)
            else:
                self._download_with_cli(s3_uri, local_path)

            logger.info("下載成功")

        except Exception as e:
            raise DownloadError(f"下載失敗: {e}")

    def _download_with_cli(self, s3_uri: str, local_path: str) -> None:
        """使用 AWS CLI 下載。"""
        cmd = ["aws", "s3", "cp", s3_uri, local_path, "--no-sign-request"]

        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            raise DownloadError(f"AWS CLI 下載失敗: {e}")

    def get_gfs_path(
        self, date_str: str, cycle: str, forecast_hour: int, res: str = "0p25"
    ) -> str:
        """
        建構 GFS S3 鍵。

        Args:
            date_str: 日期字串 YYYYMMDD
            cycle: 週期 HH（'00', '06', '12', '18'）
            forecast_hour: 預報時數
            res: 解析度字串（預設 '0p25'）

        Returns:
            S3 物件鍵
        """
        cycle = f"{int(cycle):02d}"
        f_hour = f"{int(forecast_hour):03d}"

        # 格式: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
        key = f"gfs.{date_str}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.{res}.f{f_hour}"
        return key
