"""Utils 模組單元測試。

測試 transforms.py 與 io.py 的核心功能。
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch


class TestTransforms:
    """transforms.py 測試。"""

    def test_normalize_with_provided_stats(self):
        """測試使用提供的統計值進行正規化。"""
        from pangu.utils.transforms import normalize

        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        mean = np.array([[25.0]])
        std = np.array([[10.0]])

        result = normalize(data, mean, std)

        expected = (data - 25.0) / 10.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_computes_stats(self):
        """測試自動計算統計值。"""
        from pangu.utils.transforms import normalize

        data = np.random.randn(10, 100, 100)
        result = normalize(data)

        # 正規化後的資料應接近 N(0, 1)
        assert abs(result.mean()) < 0.5
        assert abs(result.std() - 1.0) < 0.5

    def test_denormalize(self):
        """測試反正規化。"""
        from pangu.utils.transforms import denormalize

        data = np.array([[0.0, 1.0], [-1.0, 2.0]])
        mean = np.array([[10.0]])
        std = np.array([[5.0]])

        result = denormalize(data, mean, std)

        expected = data * 5.0 + 10.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_denormalize_inverse(self):
        """測試正規化與反正規化互為逆操作。"""
        from pangu.utils.transforms import normalize, denormalize, compute_statistics

        original = np.random.randn(5, 10, 10) * 100 + 50
        mean, std = compute_statistics(original)

        normalized = normalize(original, mean, std)
        recovered = denormalize(normalized, mean, std)

        np.testing.assert_array_almost_equal(original, recovered)

    def test_compute_statistics(self):
        """測試統計值計算。"""
        from pangu.utils.transforms import compute_statistics

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mean, std = compute_statistics(data)

        assert mean.shape == (1, 1)  # keepdims=True
        assert std.shape == (1, 1)
        np.testing.assert_almost_equal(mean[0, 0], 3.5)

    def test_regrid_bilinear(self):
        """測試雙線性插值 regrid。"""
        from pangu.utils.transforms import regrid

        data = np.random.randn(100, 100)
        result = regrid(data, (50, 50), method="bilinear")

        assert result.shape == (50, 50)

    def test_regrid_nearest(self):
        """測試最近鄰插值 regrid。"""
        from pangu.utils.transforms import regrid

        data = np.random.randn(100, 100)
        result = regrid(data, (200, 200), method="nearest")

        assert result.shape == (200, 200)

    def test_regrid_3d(self):
        """測試 3D 資料 regrid。"""
        from pangu.utils.transforms import regrid

        data = np.random.randn(5, 100, 100)
        result = regrid(data, (50, 50))

        assert result.shape == (5, 50, 50)


class TestStandardScaler:
    """StandardScaler 測試。"""

    def test_fit(self):
        """測試 fit 方法。"""
        from pangu.utils.transforms import StandardScaler

        scaler = StandardScaler()
        data = np.random.randn(100, 10, 10)

        result = scaler.fit(data)

        assert result is scaler  # 返回 self
        assert scaler.mean is not None
        assert scaler.std is not None

    def test_transform_before_fit_raises(self):
        """測試未 fit 就 transform 應拋出錯誤。"""
        from pangu.utils.transforms import StandardScaler

        scaler = StandardScaler()
        data = np.random.randn(10, 10)

        with pytest.raises(ValueError, match="須先呼叫 fit"):
            scaler.transform(data)

    def test_fit_transform_inverse(self):
        """測試完整流程：fit → transform → inverse_transform。"""
        from pangu.utils.transforms import StandardScaler

        scaler = StandardScaler()
        original = np.random.randn(50, 20, 20) * 100 + 50

        scaler.fit(original)
        transformed = scaler.transform(original)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(original, recovered)


class TestNPYIO:
    """NPY/NPZ I/O 測試。"""

    def test_npy_write_read(self, tmp_path):
        """測試 NPY 寫入與讀取。"""
        from pangu.utils.io import NPYReader, NPYWriter

        data = np.random.randn(10, 20)
        file_path = str(tmp_path / "test.npy")

        NPYWriter.write(data, file_path)
        loaded = NPYReader.read(file_path)

        np.testing.assert_array_equal(data, loaded)

    def test_npz_write_read(self, tmp_path):
        """測試 NPZ 寫入與讀取。"""
        from pangu.utils.io import NPYReader, NPYWriter

        data = {"arr1": np.random.randn(5, 5), "arr2": np.random.randn(3, 3)}
        file_path = str(tmp_path / "test.npz")

        NPYWriter.write(data, file_path, compressed=True)
        loaded = NPYReader.read(file_path)

        assert set(loaded.keys()) == {"arr1", "arr2"}
        np.testing.assert_array_equal(data["arr1"], loaded["arr1"])

    def test_unsupported_format_raises(self, tmp_path):
        """測試不支援的格式應拋出錯誤。"""
        from pangu.utils.io import NPYReader

        file_path = str(tmp_path / "test.csv")
        Path(file_path).touch()

        with pytest.raises(ValueError, match="不支援的檔案格式"):
            NPYReader.read(file_path)

    def test_write_creates_parent_dirs(self, tmp_path):
        """測試寫入時自動建立父目錄。"""
        from pangu.utils.io import NPYWriter

        data = np.array([1, 2, 3])
        file_path = str(tmp_path / "nested" / "deep" / "test.npy")

        NPYWriter.write(data, file_path)

        assert Path(file_path).exists()


class TestNetCDFIO:
    """NetCDF I/O 測試（需要 xarray）。"""

    @pytest.fixture
    def skip_if_no_xarray(self):
        """跳過若無 xarray。"""
        pytest.importorskip("xarray")

    def test_netcdf_reader_init(self, skip_if_no_xarray):
        """測試 NetCDFReader 初始化。"""
        from pangu.utils.io import NetCDFReader

        reader = NetCDFReader()
        assert hasattr(reader, "xr")

    def test_netcdf_writer_init(self, skip_if_no_xarray):
        """測試 NetCDFWriter 初始化。"""
        from pangu.utils.io import NetCDFWriter

        writer = NetCDFWriter()
        assert hasattr(writer, "xr")
