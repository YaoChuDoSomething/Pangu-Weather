"""Earth-Specific 3D Window Attention。

這是 Pangu-Weather 的核心創新,使用可學習的 Earth-Specific Bias
來補償球面網格的不均勻性。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class EarthAttention3D(nn.Module):
    """
    Earth-Specific 3D Window Attention。

    相較於標準的 relative position bias，使用 Earth-Specific bias
    來處理經緯度網格的不均勻性（極地較密集）。

    基於 pseudocode.py 的實作。
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (2, 6, 12),
        dropout_rate: float = 0.0,
    ):
        """
        初始化 EarthAttention3D。

        Args:
            dim: 輸入維度
            num_heads: Attention head 數量
            window_size: 3D window size (Z, H, W)
            dropout_rate: Dropout 比率
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)

        # Earth-Specific Bias
        # 為每種型(type_of_windows) 建立不同的偏置參數
        # type_of_windows = (input_shape[0]//window_size[0]) * (input_shape[1]//window_size[1])
        # 假設 input_shape = (8, 360, 181)
        self.type_of_windows = (8 // window_size[0]) * (360 // window_size[1])

        # Bias shape: ((2*W-1) * H*H * Z*Z, type_of_windows, num_heads)
        bias_size = (
            (2 * window_size[2] - 1) * (window_size[1] ** 2) * (window_size[0] ** 2)
        )

        self.earth_specific_bias = nn.Parameter(
            torch.zeros(bias_size, self.type_of_windows, num_heads)
        )
        nn.init.trunc_normal_(self.earth_specific_bias, std=0.02)

        # Position index for bias reuse
        self.register_buffer("position_index", self._construct_index())

    def _construct_index(self) -> torch.Tensor:
        """
        建構位置索引以重用對稱參數。

        Returns:
            position_index: (window_volume, window_volume) 索引
        """
        wz, wh, ww = self.window_size

        # Index in pressure level (Z)
        coords_z = torch.arange(wz)
        # Index in latitude (H)
        coords_h = torch.arange(wh)
        # Index in longitude (W)
        coords_w = torch.arange(ww)

        # Create meshgrid
        coords = torch.stack(
            torch.meshgrid([coords_z, coords_h, coords_w], indexing="ij")
        )  # (3, Z, H, W)
        coords_flatten = coords.reshape(3, -1)  # (3, Z*H*W)

        # Relative coordinates
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # (3, N, N)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (N, N, 3)

        # Shift to start from 0
        relative_coords[:, :, 2] += ww - 1  # W dimension
        relative_coords[:, :, 1] *= 2 * ww - 1  # H dimension
        relative_coords[:, :, 0] *= (2 * ww - 1) * wh * wh  # Z dimension

        # Sum to get final index
        position_index = relative_coords.sum(-1)  # (N, N)

        return position_index

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x: (B*num_windows, window_volume, C) windowed tokens
            mask: (num_windows, window_volume, window_volume) attention mask

        Returns:
            (B*num_windows, window_volume, C) attended tokens
        """
        B_win, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B_win, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_win, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale query
        q = q * self.scale

        # Attention: Q @ K^T
        attn = q @ k.transpose(-2, -1)  # (B_win, num_heads, N, N)

        # Add Earth-Specific Bias
        earth_bias = self.earth_specific_bias[
            self.position_index.view(-1)
        ]  # (N*N, type_of_windows, num_heads)
        earth_bias = earth_bias.view(N, N, self.type_of_windows, self.num_heads)
        earth_bias = earth_bias.permute(
            2, 3, 0, 1
        ).contiguous()  # (type_of_windows, num_heads, N, N)

        # 擴展以匹配 batch
        earth_bias = earth_bias.unsqueeze(0)  # (1, type_of_windows, num_heads, N, N)

        # 根據 window type 選擇對應的 bias
        # 簡化：這裡假設所有 window 使用相同 bias（可根據位置選擇）
        earth_bias = earth_bias.mean(dim=1, keepdim=True)  # (1, 1, num_heads, N, N)

        attn = attn + earth_bias.squeeze(1)  # (B_win, num_heads, N, N)

        # Apply mask if provided
        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(B_win // num_win, num_win, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)  # Add mask
            attn = attn.view(-1, self.num_heads, N, N)

        # Softmax
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Attention @ V
        x = (attn @ v).transpose(1, 2).reshape(B_win, N, C)

        # Output projection
        x = self.proj(x)
        x = self.dropout(x)

        return x
