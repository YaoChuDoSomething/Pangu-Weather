"""日誌系統。"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pangu",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    設定日誌系統。

    Args:
        name: Logger 名稱
        log_file: 日誌檔案路徑
        level: 日誌等級
        format_string: 格式字串

    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 移除現有 handlers
    logger.handlers.clear()

    # 預設格式
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "pangu") -> logging.Logger:
    """
    取得 logger。

    Args:
        name: Logger 名稱

    Returns:
        Logger 實例
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """TensorBoard 日誌記錄器（需要 tensorboard）。"""

    def __init__(self, log_dir: str = "runs"):
        """
        初始化 TensorBoard logger。

        Args:
            log_dir: 日誌目錄
        """
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
            self.log_dir = log_dir
        except ImportError:
            logging.warning("需要安裝 tensorboard: uv add tensorboard")
            self.writer = None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """記錄標量值。"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: dict, step: int) -> None:
        """記錄多個標量值。"""
        if self.writer:
            self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """記錄直方圖。"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def close(self) -> None:
        """關閉 writer。"""
        if self.writer:
            self.writer.close()
