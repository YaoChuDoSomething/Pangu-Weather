"""DataLoader 配置與工廠函數。"""

import logging
from typing import Optional
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    建立 DataLoader 的工廠函數。

    Args:
        dataset: PyTorch Dataset
        batch_size: 批次大小
        shuffle: 是否打亂資料
        num_workers: 多程序載入的 worker 數量
        pin_memory: 是否使用 pin memory（GPU 訓練建議開啟）
        drop_last: 是否丟棄最後不完整的批次

    Returns:
        配置好的 DataLoader
    """
    logger.info(
        f"建立 DataLoader: batch_size={batch_size}, shuffle={shuffle}, "
        f"num_workers={num_workers}, pin_memory={pin_memory}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    return dataloader


def build_train_val_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple:
    """
    建立訓練與驗證 DataLoader。

    Args:
        train_dataset: 訓練資料集
        val_dataset: 驗證資料集（可選）
        batch_size: 批次大小
        num_workers: Worker 數量
        pin_memory: Pin memory

    Returns:
        (train_loader, val_loader) 或 (train_loader, None)
    """
    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = build_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return train_loader, val_loader
