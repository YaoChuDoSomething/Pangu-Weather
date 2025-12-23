"""訓練器類別，負責訓練迴圈、驗證與 Checkpoint 管理。

遵循 SRP 原則，將訓練邏輯拆分為多個方法。
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pangu.training.metrics import WeatherMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    盤古模型訓練器。

    職責：
    - 訓練迴圈執行
    - 驗證邏輯
    - Checkpoint 管理
    - 日誌記錄

    透過依賴注入實現低耦合。
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        log_every: int = 100,
        save_every: int = 10,
        clip_grad: Optional[float] = 1.0,
        use_amp: bool = False,
    ):
        """
        初始化訓練器。

        Args:
            model: 模型
            criterion: 損失函數
            optimizer: 優化器
            scheduler: 學習率排程器（可選）
            device: 設備
            checkpoint_dir: Checkpoint 目錄
            log_every: 日誌記錄間隔（steps）
            save_every: 儲存間隔（epochs）
            clip_grad: 梯度裁剪閾值
            use_amp: 使用混合精度訓練
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every = log_every
        self.save_every = save_every
        self.clip_grad = clip_grad
        self.use_amp = use_amp

        # 建立 checkpoint 目錄
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 混合精度 scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # 訓練狀態
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # 指標計算器
        self.metrics = WeatherMetrics()

        logger.info(f"Trainer 初始化完成，設備: {device}")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        訓練單一 epoch。

        Args:
            train_loader: 訓練資料載入器
            epoch: 當前 epoch 編號

        Returns:
            訓練指標字典
        """
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (
            input_upper,
            input_surface,
            target_upper,
            target_surface,
        ) in enumerate(train_loader):
            # 移動資料到設備
            input_upper = input_upper.to(self.device)
            input_surface = input_surface.to(self.device)
            target_upper = target_upper.to(self.device)
            target_surface = target_surface.to(self.device)

            # 前向傳播
            loss = self._forward_step(
                input_upper, input_surface, target_upper, target_surface
            )

            # 反向傳播
            self._backward_step(loss)

            # 累積損失
            total_loss += loss.item()
            self.global_step += 1

            # 日誌記錄
            if batch_idx % self.log_every == 0:
                self._log_training_step(epoch, batch_idx, num_batches, loss.item())

        # 學習率排程
        if self.scheduler is not None and not isinstance(
            self.scheduler, ReduceLROnPlateau
        ):
            self.scheduler.step()

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    def _forward_step(
        self,
        input_upper: torch.Tensor,
        input_surface: torch.Tensor,
        target_upper: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> torch.Tensor:
        """
        執行前向傳播。

        Args:
            input_upper: 輸入上層變數
            input_surface: 輸入地表變數
            target_upper: 目標上層變數
            target_surface: 目標地表變數

        Returns:
            損失值
        """
        self.optimizer.zero_grad()

        if self.use_amp:
            with torch.cuda.amp.autocast():
                output_upper, output_surface = self.model(input_upper, input_surface)
                loss = self.criterion(
                    output_upper, target_upper, output_surface, target_surface
                )
        else:
            output_upper, output_surface = self.model(input_upper, input_surface)
            loss = self.criterion(
                output_upper, target_upper, output_surface, target_surface
            )

        return loss

    def _backward_step(self, loss: torch.Tensor) -> None:
        """
        執行反向傳播與參數更新。

        Args:
            loss: 損失值
        """
        if self.use_amp:
            self.scaler.scale(loss).backward()

            if self.clip_grad is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

            if self.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            self.optimizer.step()

    def _log_training_step(
        self, epoch: int, batch_idx: int, num_batches: int, loss: float
    ) -> None:
        """
        記錄訓練步驟。

        Args:
            epoch: Epoch 編號
            batch_idx: 批次索引
            num_batches: 總批次數
            loss: 損失值
        """
        lr = self.optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch} [{batch_idx}/{num_batches}] "
            f"Loss: {loss:.6f}, LR: {lr:.2e}, Step: {self.global_step}"
        )

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        驗證模型。

        Args:
            val_loader: 驗證資料載入器

        Returns:
            驗證指標字典
        """
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = len(val_loader)

        for input_upper, input_surface, target_upper, target_surface in val_loader:
            input_upper = input_upper.to(self.device)
            input_surface = input_surface.to(self.device)
            target_upper = target_upper.to(self.device)
            target_surface = target_surface.to(self.device)

            output_upper, output_surface = self.model(input_upper, input_surface)
            loss = self.criterion(
                output_upper, target_upper, output_surface, target_surface
            )

            total_loss += loss.item()

            # 更新指標
            self.metrics.update(
                output_upper, target_upper, output_surface, target_surface
            )

        avg_loss = total_loss / num_batches
        val_metrics = self.metrics.compute()
        val_metrics["val_loss"] = avg_loss

        logger.info(f"驗證結果: Loss={avg_loss:.6f}")
        logger.info(self.metrics.get_summary())

        # 更新學習率（如果使用 ReduceLROnPlateau）
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return val_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        儲存 checkpoint。

        Args:
            epoch: Epoch 編號
            is_best: 是否為最佳模型
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # 儲存最新 checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint 已儲存: {checkpoint_path}")

        # 儲存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"最佳模型已儲存: {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        載入 checkpoint。

        Args:
            checkpoint_path: Checkpoint 路徑

        Returns:
            恢復的 epoch 編號
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"從 {checkpoint_path} 恢復訓練，epoch={self.current_epoch}")

        return self.current_epoch

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from: Optional[str] = None,
    ) -> None:
        """
        執行完整訓練流程。

        Args:
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            epochs: 訓練 epoch 數
            resume_from: 恢復訓練的 checkpoint 路徑
        """
        start_epoch = 0

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1

        logger.info(f"開始訓練，epoch {start_epoch} 到 {epochs}")

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch

            # 訓練
            train_metrics = self.train_epoch(train_loader, epoch)
            logger.info(
                f"Epoch {epoch} 完成，train_loss: {train_metrics['train_loss']:.6f}"
            )

            # 驗證
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

                # 追蹤最佳模型
                val_loss = val_metrics["val_loss"]
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False

            # 儲存 checkpoint
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        logger.info("訓練完成！")
