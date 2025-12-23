"""Pangu-Weather 主模型架構。

基於 pseudocode.py 的完整實作，使用 3D Swin Transformer。
"""

import torch
import torch.nn as nn
import numpy as np

from pangu.models.layers import PatchEmbedding, PatchRecovery
from pangu.models.upsample_downsample import UpSample, DownSample
from pangu.models.blocks import EarthSpecificLayer


class PanguModel(nn.Module):
    """
    Pangu-Weather 模型。

    架構（基於 pseudocode.py）：
    - Patch Embedding
    - Encoder
      - Layer1 (2 blocks, dim=192, heads=6) @ (8,360,181)
      - DownSample → (8,180,91)
    - Bottleneck
      - Layer2 (6 blocks, dim=384, heads=12) @ (8,180,91)
    - Decoder
      - Layer3 (6 blocks, dim=384, heads=12) @ (8,180,91)
      - UpSample → (8,360,181)
      - Layer4 (2 blocks, dim=192, heads=6) @ (8,360,181)
      - Skip Connection (concat)
    - Patch Recovery
    """

    def __init__(
        self,
        patch_size: tuple = (2, 4, 4),
        embed_dim: int = 192,
        depths: list = None,
        num_heads: list = None,
        window_size: tuple = (2, 6, 12),
        drop_path_rate: float = 0.2,
        dropout: float = 0.0,
    ):
        """
        初始化 PanguModel。

        Args:
            patch_size: Patch size (Z, H, W)
            embed_dim: Base embedding dimension
            depths: Number of blocks in each layer [2, 6, 6, 2]
            num_heads: Number of attention heads [6, 12, 12, 6]
            window_size: Window size for attention
            drop_path_rate: DropPath rate (linearly increased)
            dropout: Dropout rate
        """
        super().__init__()

        if depths is None:
            depths = [2, 6, 6, 2]
        if num_heads is None:
            num_heads = [6, 12, 12, 6]

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads

        # DropPath rates (linearly spaced from 0 to drop_path_rate)
        total_depth = sum(depths)
        dpr = np.linspace(0, drop_path_rate, total_depth).tolist()

        # Patch Embedding
        self.input_layer = PatchEmbedding(patch_size, embed_dim)

        # Encoder Layer 1
        self.layer1 = EarthSpecificLayer(
            depth=depths[0],
            dim=embed_dim,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path_list=dpr[: depths[0]],
            dropout=dropout,
        )

        # Downsample
        self.downsample = DownSample(embed_dim)

        # Bottleneck Layer 2
        self.layer2 = EarthSpecificLayer(
            depth=depths[1],
            dim=embed_dim * 2,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path_list=dpr[depths[0] : depths[0] + depths[1]],
            dropout=dropout,
        )

        # Decoder Layer 3
        self.layer3 = EarthSpecificLayer(
            depth=depths[2],
            dim=embed_dim * 2,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path_list=dpr[
                depths[0] + depths[1] : depths[0] + depths[1] + depths[2]
            ],
            dropout=dropout,
        )

        # Upsample
        self.upsample = UpSample(embed_dim * 2, embed_dim)

        # Decoder Layer 4
        self.layer4 = EarthSpecificLayer(
            depth=depths[3],
            dim=embed_dim,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path_list=dpr[depths[0] + depths[1] + depths[2] :],
            dropout=dropout,
        )

        # Patch Recovery
        self.output_layer = PatchRecovery(embed_dim * 2, patch_size)

    def forward(self, input_upper: torch.Tensor, input_surface: torch.Tensor) -> tuple:
        """
        前向傳播。

        Args:
            input_upper: (B, 5, 13, 721, 1440) 上層變數
            input_surface: (B, 4, 721, 1440) 地表變數

        Returns:
            (output_upper, output_surface)
            - output_upper: (B, 5, 13, 721, 1440)
            - output_surface: (B, 4, 721, 1440)
        """
        # Patch Embedding
        x = self.input_layer(input_upper, input_surface)  # (B, 8*360*181, C)

        # Encoder
        # Layer 1 @ (8, 360, 181)
        x = self.layer1(x, 8, 360, 181)

        # Store for skip connection
        skip = x

        # Downsample to (8, 180, 91)
        x = self.downsample(x, 8, 360, 181)

        # Bottleneck
        # Layer 2 @ (8, 180, 91)
        x = self.layer2(x, 8, 180, 91)

        # Decoder
        # Layer 3 @ (8, 180, 91)
        x = self.layer3(x, 8, 180, 91)

        # Upsample to (8, 360, 181)
        x = self.upsample(x)

        # Layer 4 @ (8, 360, 181)
        x = self.layer4(x, 8, 360, 181)

        # Skip connection (concatenate)
        x = torch.cat([skip, x], dim=-1)  # (B, 8*360*181, 2C)

        # Patch Recovery
        output_upper, output_surface = self.output_layer(x)

        return output_upper, output_surface

    def load_constants(
        self, land_mask: torch.Tensor, soil_type: torch.Tensor, topography: torch.Tensor
    ):
        """
        載入常數遮罩到 PatchEmbedding。

        Args:
            land_mask: (1, 1, 721, 1440)
            soil_type: (1, 1, 721, 1440)
            topography: (1, 1, 721, 1440)
        """
        self.input_layer.load_constants(land_mask, soil_type, topography)


def create_pangu_model(config: dict = None) -> PanguModel:
    """
    工廠函數，建立 PanguModel。

    Args:
        config: 模型配置字典

    Returns:
        PanguModel 實例
    """
    if config is None:
        config = {}

    model = PanguModel(
        patch_size=config.get("patch_size", (2, 4, 4)),
        embed_dim=config.get("embed_dim", 192),
        depths=config.get("depths", [2, 6, 6, 2]),
        num_heads=config.get("num_heads", [6, 12, 12, 6]),
        window_size=config.get("window_size", (2, 6, 12)),
        drop_path_rate=config.get("drop_path_rate", 0.2),
        dropout=config.get("dropout", 0.0),
    )

    return model
