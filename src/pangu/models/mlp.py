"""MLP (Feed-Forward Network) 模組。"""

import torch.nn as nn


class Mlp(nn.Module):
    """
    MLP 層，用於 Transformer block。

    結構: Linear -> GeLU -> Dropout -> Linear -> Dropout
    Expansion ratio: 4x
    """

    def __init__(self, dim: int, dropout_rate: float = 0.0, expansion_ratio: int = 4):
        """
        初始化 MLP。

        Args:
            dim: 輸入/輸出維度
            dropout_rate: Dropout 比率
            expansion_ratio: 隱藏層擴展比率（預設 4）
        """
        super().__init__()

        hidden_dim = dim * expansion_ratio

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向傳播。

        Args:
            x: (B, N, C) tokens

        Returns:
            (B, N,C) transformed tokens
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
