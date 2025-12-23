"""氣象專用損失函數。"""

import torch
import torch.nn as nn
from typing import Tuple


class PanguLoss(nn.Module):
    """
    盤古模型損失函數。

    根據 pseudocode.py:
    loss = TensorAbs(output-target) + TensorAbs(output_surface-target_surface) * 0.25

    使用 MAE (L1 Loss) 並對地表損失加權 0.25。
    """

    def __init__(self, surface_weight: float = 0.25):
        """
        初始化損失函數。

        Args:
            surface_weight: 地表損失權重（預設 0.25）
        """
        super().__init__()
        self.surface_weight = surface_weight
        self.mae = nn.L1Loss()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        output_surface: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> torch.Tensor:
        """
        計算損失。

        Args:
            output: 預測上層變數 (B, 5, 13, H, W)
            target: 目標上層變數 (B, 5, 13, H, W)
            output_surface: 預測地表變數 (B, 4, H, W)
            target_surface: 目標地表變數 (B, 4, H, W)

        Returns:
            總損失值
        """
        loss_upper = self.mae(output, target)
        loss_surface = self.mae(output_surface, target_surface)

        total_loss = loss_upper + loss_surface * self.surface_weight

        return total_loss


class WeightedMSELoss(nn.Module):
    """加權 MSE 損失，可用於不同變數的差異化權重。"""

    def __init__(self, upper_weight: float = 1.0, surface_weight: float = 0.25):
        """
        初始化加權 MSE 損失。

        Args:
            upper_weight: 上層變數權重
            surface_weight: 地表變數權重
        """
        super().__init__()
        self.upper_weight = upper_weight
        self.surface_weight = surface_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        output_surface: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> torch.Tensor:
        """計算加權 MSE 損失。"""
        loss_upper = self.mse(output, target) * self.upper_weight
        loss_surface = self.mse(output_surface, target_surface) * self.surface_weight

        return loss_upper + loss_surface


class PhysicsConstraintLoss(nn.Module):
    """
    物理約束損失（實驗性）。

    可用於強化質量守恆、能量守恆等物理定律。
    """

    def __init__(self, weight: float = 0.01):
        """
        初始化物理約束損失。

        Args:
            weight: 物理約束項權重
        """
        super().__init__()
        self.weight = weight

    def forward(
        self, output: torch.Tensor, output_surface: torch.Tensor
    ) -> torch.Tensor:
        """
        計算物理約束損失。

        Args:
            output: 預測上層變數
            output_surface: 預測地表變數

        Returns:
            物理約束損失值
        """
        # Placeholder: 實際需要根據物理方程實作
        # 例如：質量守恆、能量守恆等

        # 簡單的範數約束作為示例
        constraint = torch.mean(torch.abs(output)) + torch.mean(
            torch.abs(output_surface)
        )

        return constraint * self.weight


class CombinedLoss(nn.Module):
    """組合多個損失函數。"""

    def __init__(
        self,
        use_mae: bool = True,
        use_mse: bool = False,
        use_physics: bool = False,
        surface_weight: float = 0.25,
        physics_weight: float = 0.01,
    ):
        """
        初始化組合損失。

        Args:
            use_mae: 使用 MAE 損失
            use_mse: 使用 MSE 損失
            use_physics: 使用物理約束損失
            surface_weight: 地表損失權重
            physics_weight: 物理約束權重
        """
        super().__init__()

        self.losses = []

        if use_mae:
            self.losses.append(("mae", PanguLoss(surface_weight)))

        if use_mse:
            self.losses.append(("mse", WeightedMSELoss(surface_weight=surface_weight)))

        if use_physics:
            self.losses.append(("physics", PhysicsConstraintLoss(physics_weight)))

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        output_surface: torch.Tensor,
        target_surface: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        計算組合損失。

        Returns:
            (total_loss, loss_dict)
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss_fn in self.losses:
            if name == "physics":
                loss_val = loss_fn(output, output_surface)
            else:
                loss_val = loss_fn(output, target, output_surface, target_surface)

            total_loss += loss_val
            loss_dict[name] = loss_val.item()

        return total_loss, loss_dict
