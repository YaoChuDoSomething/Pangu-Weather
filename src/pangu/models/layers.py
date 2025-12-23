"""Patch Embedding 與 Recovery 層。

基於 pseudocode.py 實作 3D 與 2D 的 patch embedding/recovery。
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbedding(nn.Module):
    """
    Patch Embedding 層。

    將 3D 上層資料 (5, 13, 721, 1440) 和 2D 地表資料 (4, 721, 1440)
    轉換為 patch tokens。

    根據 pseudocode.py:
    - 使用 Conv3d 處理上層資料 (patch_size = (2, 4, 4))
    - 使用 Conv2d 處理地表資料 (patch_size = (4, 4))
    - 加入 3 個常數遮罩：land_mask, soil_type, topography
    """

    def __init__(
        self, patch_size: Tuple[int, int, int] = (2, 4, 4), embed_dim: int = 192
    ):
        """
        初始化 Patch Embedding。

        Args:
            patch_size: Patch 大小 (Z, H, W)
            embed_dim: 嵌入維度
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 3D Conv for upper air (5 vars -> embed_dim)
        self.conv3d = nn.Conv3d(
            in_channels=5,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # 2D Conv for surface (4 vars + 3 constants = 7 -> embed_dim)
        self.conv2d = nn.Conv2d(
            in_channels=7,
            out_channels=embed_dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )

        # 常數遮罩（需要載入或初始化）
        # 這裡使用可學習參數作為 placeholder
        # 實際應從檔案載入
        self.land_mask = nn.Parameter(torch.zeros(1, 1, 721, 1440), requires_grad=False)
        self.soil_type = nn.Parameter(torch.zeros(1, 1, 721, 1440), requires_grad=False)
        self.topography = nn.Parameter(
            torch.zeros(1, 1, 721, 1440), requires_grad=False
        )

    def forward(
        self, input_upper: torch.Tensor, input_surface: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播。

        Args:
            input_upper: (B, 5, 13, 721, 1440) 上層變數
            input_surface: (B, 4, 721, 1440) 地表變數

        Returns:
            (B, Z*H*W, C) embedded tokens
        """
        B = input_upper.shape[0]

        # Zero-pad 上層資料以符合 patch_size
        # 原始: (B, 5, 13, 721, 1440)
        # Pad to: (B, 5, 14, 724, 1440) -> (8, 181, 360) after patch
        input_upper = self._pad_3d(input_upper)

        # 應用 3D convolution
        upper_embedded = self.conv3d(input_upper)  # (B, C, 7, 181, 360)

        # 加入常數遮罩（原始尺寸 721x1440）
        land_mask = self.land_mask.expand(B, -1, -1, -1)
        soil_type = self.soil_type.expand(B, -1, -1, -1)
        topography = self.topography.expand(B, -1, -1, -1)

        # Concatenate surface vars with constants (都是原始尺寸)
        input_surface = torch.cat(
            [input_surface, land_mask, soil_type, topography], dim=1
        )  # (B, 7, 721, 1440)

        # 現在統一 pad
        input_surface = self._pad_2d(input_surface)  # (B, 7, 724, 1440)

        # 應用 2D convolution
        surface_embedded = self.conv2d(input_surface)  # (B, C, 181, 360)

        # 在 Z 維度加一層以便 concat
        surface_embedded = surface_embedded.unsqueeze(2)  # (B, C, 1, 181, 360)

        # Concat in Z dimension
        x = torch.cat([surface_embedded, upper_embedded], dim=2)  # (B, C, 8, 181, 360)

        # Reshape to (B, Z*H*W, C)
        x = x.permute(0, 2, 3, 4, 1)  # (B, 8, 181, 360, C)
        x = x.reshape(B, -1, self.embed_dim)  # (B, 8*181*360, C)

        return x

    def _pad_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad 3D tensor。"""
        # Pad Z: 13 -> 14 (pad 1 on top)
        # Pad H: 721 -> 724 (pad 3 on top, 0 on bottom for asymmetry)
        # Pad W: 1440 -> 1440 (no padding needed, divisible by 4)
        pad = (0, 0, 3, 0, 1, 0)  # (W_left, W_right, H_left, H_right, Z_left, Z_right)
        return nn.functional.pad(x, pad, mode="constant", value=0)

    def _pad_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-pad 2D tensor。"""
        # Pad H: 721 -> 724
        # Pad W: 1440 -> 1440
        pad = (0, 0, 3, 0)  # (W_left, W_right, H_left, H_right)
        return nn.functional.pad(x, pad, mode="constant", value=0)

    def load_constants(
        self, land_mask: torch.Tensor, soil_type: torch.Tensor, topography: torch.Tensor
    ):
        """
        載入常數遮罩。

        Args:
            land_mask: (1, 1, 721, 1440)
            soil_type: (1, 1, 721, 1440)
            topography: (1, 1, 721, 1440)
        """
        self.land_mask.data = land_mask
        self.soil_type.data = soil_type
        self.topography.data = topography


class PatchRecovery(nn.Module):
    """
    Patch Recovery 層。

    將 patch tokens 恢復為原始的上層與地表資料。
    使用 transposed convolution (ConvTranspose3d/2d)。
    """

    def __init__(self, dim: int = 384, patch_size: Tuple[int, int, int] = (2, 4, 4)):
        """
        初始化 Patch Recovery。

        Args:
            dim: 輸入維度（skip connection 後為 384）
            patch_size: Patch 大小
        """
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size

        # Transposed Conv3d for upper air
        self.conv_transpose_3d = nn.ConvTranspose3d(
            in_channels=dim, out_channels=5, kernel_size=patch_size, stride=patch_size
        )

        # Transposed Conv2d for surface
        self.conv_transpose_2d = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=4,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播。

        Args:
            x: (B, Z*H*W, C) embedded tokens

        Returns:
            (output_upper, output_surface)
            - output_upper: (B, 5, 13, 721, 1440)
            - output_surface: (B, 4, 721, 1440)
        """
        B = x.shape[0]
        Z, H, W = 8, 181, 360

        # Reshape to (B, Z, H, W, C)
        x = x.reshape(B, Z, H, W, self.dim)
        x = x.permute(0, 4, 1, 2, 3)  # (B, C, Z, H, W)

        # 分離 surface (第一層) 和 upper (其餘層)
        surface_tokens = x[:, :, 0, :, :]  # (B, C, H, W)
        upper_tokens = x[:, :, 1:, :, :]  # (B, C, 7, H, W)

        # Apply transposed convolutions
        output_upper = self.conv_transpose_3d(upper_tokens)  # (B, 5, 14, 724, 1440)
        output_surface = self.conv_transpose_2d(surface_tokens)  # (B, 4, 724, 1440)

        # Crop to remove padding
        output_upper = self._crop_3d(output_upper)  # (B, 5, 13, 721, 1440)
        output_surface = self._crop_2d(output_surface)  # (B, 4, 721, 1440)

        return output_upper, output_surface

    def _crop_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Crop 3D tensor to original size。"""
        # Remove padding: (B, 5, 14, 724, 1440) -> (B, 5, 13, 721, 1440)
        return x[:, :, 1:14, 3:724, :]

    def _crop_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Crop 2D tensor to original size。"""
        # Remove padding: (B, 4, 724, 1440) -> (B, 4, 721, 1440)
        return x[:, :, 3:724, :]
