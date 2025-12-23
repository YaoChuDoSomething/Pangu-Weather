"""UpSample 與 DownSample 層。

基於 pseudocode.py 的實作。
"""

import torch
import torch.nn as nn


class DownSample(nn.Module):
    """
    Down-sampling 層。

    將解析度從 (8, 360, 181) 降至 (8, 180, 91)，
    同時將通道數從 C 增加到 2C。
    """

    def __init__(self, dim: int = 192):
        """
        初始化 DownSample。

        Args:
            dim: 輸入維度
        """
        super().__init__()
        self.dim = dim

        # LayerNorm 與 Linear
        self.norm = nn.LayerNorm(4 * dim)
        self.linear = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor, Z: int, H: int, W: int) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x: (B, Z*H*W, C) tokens
            Z, H, W: 空間維度

        Returns:
            (B, Z*H'*W', 2C) downsampled tokens
        """
        B, _, C = x.shape

        # Reshape to  (B, Z, H, W, C)
        x = x.reshape(B, Z, H, W, C)

        # Pad for downsampling (需要能被2整除)
        # H: 360 已經可被2整除，不需pad
        # W: 181 -> 182 (pad 1)
        # Z: 8 已經可被2整除
        pad_w = (182 - W) if W % 2 == 1 else 0
        if pad_w > 0:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, 0, 0, 0))  # (B, Z, H, W+1, C)

        _, Z_new, H_new, W_new, _ = x.shape

        # Reorganize for downsampling (2x2 patches in H and W)
        # Reshape: (B, Z, H//2, 2, W//2, 2, C)
        x = x.reshape(B, Z_new, H_new // 2, 2, W_new // 2, 2, C)

        # Transpose: (B, Z, H//2, W//2, 2, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)

        # Reshape to merge 2x2 patches: (B, Z*(H//2)*(W//2), 4*C)
        x = x.reshape(B, Z_new * (H_new // 2) * (W_new // 2), 4 * C)

        # Apply normalization and linear
        x = self.norm(x)
        x = self.linear(x)

        return x


class UpSample(nn.Module):
    """
    Up-sampling 層。

    將解析度從 (8, 180, 91) 升至 (8, 360, 181)，
    同時將通道數從 2C 降至 C。
    """

    def __init__(self, input_dim: int = 384, output_dim: int = 192):
        """
        初始化 UpSample。

        Args:
            input_dim: 輸入維度（2C）
            output_dim: 輸出維度（C）
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Linear layers
        self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。

        Args:
            x: (B, Z*H*W, 2C) tokens

        Returns:
            (B, Z*H'*W', C) upsampled tokens
        """
        B = x.shape[0]

        # Increase channels: (B, N, 2C) -> (B, N, 4C)
        x = self.linear1(x)

        # Reshape for upsampling
        # Assuming input is (B, 8*180*91, 4C)
        Z, H, W = 8, 180, 91
        C = self.output_dim

        # Reshape: (B, Z, H, W, 2, 2, C)
        x = x.reshape(B, Z, H, W, 2, 2, C)

        # Transpose: (B, Z, H, 2, W, 2, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)

        # Reshape: (B, Z, H*2, W*2, C)
        x = x.reshape(B, Z, H * 2, W * 2, C)

        # Crop to target size: (8, 360, 181)
        x = x[:, :, :360, :181, :]

        # Reshape back to tokens
        x = x.reshape(B, Z * 360 * 181, C)

        # Normalization and mix
        x = self.norm(x)
        x = self.linear2(x)

        return x
