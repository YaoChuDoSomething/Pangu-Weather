"""Data downloaders 單元測試。

測試 ERA5Downloader 與 GFSGDEXDownloader 的核心功能。
使用 mock 避免實際的 API 呼叫。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pangu.data.base import DownloadError, ConfigValidationError


class TestERA5Downloader:
    """ERA5Downloader 測試。"""

    def test_init_with_default_client(self):
        """測試預設初始化。"""
        with patch("pangu.data.era5.cdsapi") as mock_cdsapi:
            mock_cdsapi.Client.return_value = Mock()
            from pangu.data.era5 import ERA5Downloader

            downloader = ERA5Downloader()
            assert downloader.max_retries == 3
            assert downloader.retry_delay == 5
            mock_cdsapi.Client.assert_called_once()

    def test_validate_config_success(self):
        """測試有效配置驗證。"""
        with patch("pangu.data.era5.cdsapi"):
            from pangu.data.era5 import ERA5Downloader

            downloader = ERA5Downloader()
            config = {"dataset_name": "reanalysis-era5-single-levels"}
            assert downloader.validate_config(config) is True

    def test_validate_config_missing_key(self):
        """測試缺少必要鍵的配置。"""
        with patch("pangu.data.era5.cdsapi"):
            from pangu.data.era5 import ERA5Downloader

            downloader = ERA5Downloader()
            config = {"some_key": "value"}  # 缺少 dataset_name
            assert downloader.validate_config(config) is False

    def test_build_request_excludes_internal_keys(self):
        """測試 _build_request 排除內部鍵。"""
        with patch("pangu.data.era5.cdsapi"):
            from pangu.data.era5 import ERA5Downloader

            downloader = ERA5Downloader()
            config = {
                "type": "era5",
                "dataset_name": "test-dataset",
                "product_type": "reanalysis",
                "variable": ["temperature"],
            }
            request = downloader._build_request(config)
            assert "type" not in request
            assert "dataset_name" not in request
            assert "product_type" in request
            assert "variable" in request

    def test_download_success(self, tmp_path):
        """測試成功下載。"""
        with patch("pangu.data.era5.cdsapi") as mock_cdsapi:
            mock_client = Mock()
            mock_cdsapi.Client.return_value = mock_client

            from pangu.data.era5 import ERA5Downloader

            downloader = ERA5Downloader()
            config = {"dataset_name": "test-dataset", "product_type": "reanalysis"}
            output_path = str(tmp_path / "output.nc")

            downloader.download(config, output_path)

            mock_client.retrieve.assert_called_once()

    def test_download_retry_on_failure(self, tmp_path):
        """測試下載失敗時的重試機制。"""
        with patch("pangu.data.era5.cdsapi") as mock_cdsapi:
            with patch("pangu.data.era5.time.sleep"):  # 避免實際等待
                mock_client = Mock()
                # 前兩次失敗，第三次成功
                mock_client.retrieve.side_effect = [
                    Exception("Network error"),
                    Exception("Timeout"),
                    None,
                ]
                mock_cdsapi.Client.return_value = mock_client

                from pangu.data.era5 import ERA5Downloader

                downloader = ERA5Downloader(max_retries=3)
                config = {"dataset_name": "test-dataset"}
                output_path = str(tmp_path / "output.nc")

                downloader.download(config, output_path)

                assert mock_client.retrieve.call_count == 3

    def test_download_max_retries_exceeded(self, tmp_path):
        """測試超過最大重試次數後拋出錯誤。"""
        with patch("pangu.data.era5.cdsapi") as mock_cdsapi:
            with patch("pangu.data.era5.time.sleep"):
                mock_client = Mock()
                mock_client.retrieve.side_effect = Exception("Always fails")
                mock_cdsapi.Client.return_value = mock_client

                from pangu.data.era5 import ERA5Downloader

                downloader = ERA5Downloader(max_retries=2)
                config = {"dataset_name": "test-dataset"}
                output_path = str(tmp_path / "output.nc")

                with pytest.raises(DownloadError):
                    downloader.download(config, output_path)


class TestGFSGDEXDownloader:
    """GFSGDEXDownloader 測試。"""

    def test_init_with_injected_client(self):
        """測試使用注入的 client 初始化。"""
        mock_client = Mock()
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)
        assert downloader.client == mock_client
        assert downloader.poll_interval == 10

    def test_validate_config_dict(self):
        """測試 dict 配置驗證。"""
        mock_client = Mock()
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)
        assert downloader.validate_config({"key": "value"}) is True

    def test_validate_config_string(self):
        """測試 string 配置驗證。"""
        mock_client = Mock()
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)
        assert downloader.validate_config("/path/to/config.json") is True

    def test_submit_dict_payload(self):
        """測試提交 dict payload。"""
        mock_client = Mock()
        mock_client.submit_json.return_value = {"data": {"request_index": "REQ-12345"}}
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)
        request_id = downloader.submit({"dataset": "gfs"})

        assert request_id == "REQ-12345"
        mock_client.submit_json.assert_called_once()

    def test_submit_failure(self):
        """測試提交失敗。"""
        mock_client = Mock()
        mock_client.submit_json.return_value = {"error": "Invalid request"}
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)

        with pytest.raises(DownloadError):
            downloader.submit({"dataset": "gfs"})

    def test_check_status(self):
        """測試狀態檢查。"""
        mock_client = Mock()
        mock_client.get_status.return_value = {"data": {"status": "Completed"}}
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)
        status = downloader._check_status("REQ-12345")

        assert status == "Completed"

    def test_is_submission_successful(self):
        """測試提交成功判斷。"""
        mock_client = Mock()
        from pangu.data.gfs_gdex import GFSGDEXDownloader

        downloader = GFSGDEXDownloader(client=mock_client)

        assert (
            downloader._is_submission_successful({"data": {"request_index": "123"}})
            is True
        )
        assert downloader._is_submission_successful({"error": "failed"}) is False
        assert downloader._is_submission_successful({"data": {}}) is False
