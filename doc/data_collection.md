# 資料收集指南

## 概覽

本專案支援從以下來源收集氣象資料：
- **ERA5**: ECMWF 再分析資料 (CDS API)
- **GFS**: 美國 NCEP 全球預報系統 (GDEX/S3)

## 設定

### ERA5 API 設定

1. 註冊 [CDS 帳號](https://cds.climate.copernicus.eu/)
2. 建立 `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api/v2
key: <UID>:<API-KEY>
```

### GFS S3 設定

AWS CLI 不需要認證（公開資料）：
```bash
aws s3 ls s3://noaa-gfs-bdp-pds/ --no-sign-request
```

## 使用方式

### 基本下載

```python
from pangu.data import ERA5Downloader, GFSRealtimeDownloader

# ERA5
era5 = ERA5Downloader()
era5.download(
    config={
        'dataset_name': 'reanalysis-era5-pressure-levels',
        'variable': ['geopotential', 'temperature'],
        'pressure_level': ['500', '850'],
        'year': '2020',
        'month': '01',
        'day': '01',
        'time': '00:00'
    },
    output_path='data/era5_2020010100.nc'
)

# GFS S3
gfs = GFSRealtimeDownloader()
gfs.download(
    config={
        'date_str': '20231201',
        'cycle': '00',
        'forecast_hours': [0, 6, 12]
    },
    output_path='data/gfs/'
)
```

### 配置檔案

編輯 `configs/data_prepare.yaml`:

```yaml
datasets:
  era5_surface:
    type: cds
    dataset_name: reanalysis-era5-single-levels
    variable:
      - mean_sea_level_pressure
      - 10m_u_component_of_wind
      - 10m_v_component_of_wind
      - 2m_temperature
```

### 執行腳本

```bash
python scripts/data_prepare.py --config configs/data_prepare.yaml
```

## Pangu 專用資料

盤古模型需要特定的變數組合：

**Upper Air (5 變數 × 13 層)**:
- 變數: Z, Q, T, U, V
- 氣壓層: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

**Surface (4 變數)**:
- MSLP, U10, V10, T2M

合併表面與高層資料請參考 `pangu_era5.py`。
