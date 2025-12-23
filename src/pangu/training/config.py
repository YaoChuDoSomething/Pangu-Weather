"""訓練配置類別。"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """訓練配置類別。"""

    # 訓練參數
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 3e-6

    # 學習率排程
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # 梯度裁剪
    clip_grad: Optional[float] = 1.0

    # Checkpoint
    save_every: int = 10
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # 日誌
    log_every: int = 100
    log_dir: str = "logs"

    # 資料
    data_dir: str = "data/train"
    val_data_dir: Optional[str] = None
    num_workers: int = 4

    # 分散式訓練
    distributed: bool = False
    world_size: int = 1
    rank: int = 0

    # 混合精度
    use_amp: bool = False

    # 其他
    seed: int = 42
    device: str = "cuda"

    # 擴展參數
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """
        從 YAML 檔案載入配置。

        Args:
            yaml_path: YAML 檔案路徑

        Returns:
            TrainingConfig 實例
        """
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 提取 training 部分
        training_dict = config_dict.get("training", {})

        logger.info(f"從 {yaml_path} 載入訓練配置")

        return cls(**training_dict)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典。"""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler,
            "warmup_epochs": self.warmup_epochs,
            "clip_grad": self.clip_grad,
            "save_every": self.save_every,
            "checkpoint_dir": self.checkpoint_dir,
            "log_every": self.log_every,
            "distributed": self.distributed,
            "use_amp": self.use_amp,
            "seed": self.seed,
            "device": self.device,
        }

    def save_yaml(self, yaml_path: str) -> None:
        """
        儲存配置到 YAML 檔案。

        Args:
            yaml_path: 輸出路徑
        """
        config_dict = {"training": self.to_dict()}

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"配置已儲存到 {yaml_path}")
