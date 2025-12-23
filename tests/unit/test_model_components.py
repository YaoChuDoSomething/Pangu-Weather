"""Model components 單元測試。

測試 attention.py, blocks.py, layers.py, upsample_downsample.py 的獨立元件。
"""

import pytest
import torch
import torch.nn as nn


class TestEarthAttention3D:
    """EarthAttention3D 測試。"""

    def test_forward_shape(self):
        """測試前向傳播輸出形狀。"""
        from pangu.models.attention import EarthAttention3D

        dim = 192
        num_heads = 6
        window_size = (2, 6, 12)
        window_volume = window_size[0] * window_size[1] * window_size[2]

        attn = EarthAttention3D(dim, num_heads, window_size)

        # Input: (B*num_windows, window_volume, C)
        B_win = 16
        x = torch.randn(B_win, window_volume, dim)

        with torch.no_grad():
            output = attn(x)

        assert output.shape == (B_win, window_volume, dim)

    def test_num_heads_divides_dim(self):
        """測試 dim 必須能被 num_heads 整除。"""
        from pangu.models.attention import EarthAttention3D

        dim = 192
        num_heads = 6  # 192 / 6 = 32 ✓

        attn = EarthAttention3D(dim, num_heads)
        assert attn.scale == (dim // num_heads) ** -0.5

    def test_position_index_shape(self):
        """測試 position_index buffer 形狀。"""
        from pangu.models.attention import EarthAttention3D

        window_size = (2, 6, 12)
        window_volume = 2 * 6 * 12

        attn = EarthAttention3D(192, 6, window_size)

        assert attn.position_index.shape == (window_volume, window_volume)


class TestEarthSpecificBlock:
    """EarthSpecificBlock 測試。"""

    def test_forward_shape(self):
        """測試前向傳播輸出形狀。"""
        from pangu.models.blocks import EarthSpecificBlock

        dim = 192
        num_heads = 6
        Z, H, W = 8, 181, 360

        block = EarthSpecificBlock(dim, num_heads)

        B = 2
        x = torch.randn(B, Z * H * W, dim)

        with torch.no_grad():
            output = block(x, Z, H, W, roll=False)

        assert output.shape == (B, Z * H * W, dim)

    def test_forward_with_roll(self):
        """測試帶 cyclic shift 的前向傳播。"""
        from pangu.models.blocks import EarthSpecificBlock

        dim = 192
        block = EarthSpecificBlock(dim, 6)

        B, Z, H, W = 1, 8, 181, 360
        x = torch.randn(B, Z * H * W, dim)

        with torch.no_grad():
            output = block(x, Z, H, W, roll=True)

        assert output.shape == x.shape

    def test_residual_connection(self):
        """測試殘差連接存在。"""
        from pangu.models.blocks import EarthSpecificBlock

        block = EarthSpecificBlock(192, 6)

        # 若 drop_path=0，殘差應該直接加上
        assert hasattr(block, "drop_path")


class TestEarthSpecificLayer:
    """EarthSpecificLayer 測試。"""

    def test_forward_shape(self):
        """測試 Layer 前向傳播。"""
        from pangu.models.blocks import EarthSpecificLayer

        depth = 2
        dim = 192
        layer = EarthSpecificLayer(depth, dim, num_heads=6)

        B, Z, H, W = 1, 8, 181, 360
        x = torch.randn(B, Z * H * W, dim)

        with torch.no_grad():
            output = layer(x, Z, H, W)

        assert output.shape == x.shape

    def test_alternating_roll(self):
        """測試交替 roll 機制。"""
        from pangu.models.blocks import EarthSpecificLayer

        layer = EarthSpecificLayer(depth=4, dim=192, num_heads=6)

        # 4 blocks: roll=[False, True, False, True]
        assert len(layer.blocks) == 4


class TestPatchEmbedding:
    """PatchEmbedding 測試。"""

    def test_forward_shape(self):
        """測試 Patch Embedding 輸出形狀。"""
        from pangu.models.layers import PatchEmbedding

        embed_dim = 192
        patch_embed = PatchEmbedding(embed_dim=embed_dim)

        B = 1
        input_upper = torch.randn(B, 5, 13, 721, 1440)
        input_surface = torch.randn(B, 4, 721, 1440)

        with torch.no_grad():
            output = patch_embed(input_upper, input_surface)

        # Expected: (B, 8*181*360, 192)
        Z, H, W = 8, 181, 360
        assert output.shape == (B, Z * H * W, embed_dim)

    def test_load_constants(self):
        """測試載入常數遮罩。"""
        from pangu.models.layers import PatchEmbedding

        patch_embed = PatchEmbedding()

        land_mask = torch.randn(1, 1, 721, 1440)
        soil_type = torch.randn(1, 1, 721, 1440)
        topography = torch.randn(1, 1, 721, 1440)

        patch_embed.load_constants(land_mask, soil_type, topography)

        torch.testing.assert_close(patch_embed.land_mask.data, land_mask)


class TestPatchRecovery:
    """PatchRecovery 測試。"""

    def test_forward_shape(self):
        """測試 Patch Recovery 輸出形狀。"""
        from pangu.models.layers import PatchRecovery

        dim = 384  # skip connection 後的維度
        patch_recovery = PatchRecovery(dim=dim)

        B = 1
        Z, H, W = 8, 181, 360
        x = torch.randn(B, Z * H * W, dim)

        with torch.no_grad():
            output_upper, output_surface = patch_recovery(x)

        assert output_upper.shape == (B, 5, 13, 721, 1440)
        assert output_surface.shape == (B, 4, 721, 1440)


class TestDownSample:
    """DownSample 測試。"""

    def test_forward_shape(self):
        """測試 DownSample 輸出形狀。"""
        from pangu.models.upsample_downsample import DownSample

        dim = 192
        downsample = DownSample(dim=dim)

        B = 1
        Z, H, W = 8, 360, 181
        x = torch.randn(B, Z * H * W, dim)

        with torch.no_grad():
            output = downsample(x, Z, H, W)

        # Expected: (B, 8*180*91, 384)
        expected_tokens = 8 * 180 * 91
        assert output.shape == (B, expected_tokens, 2 * dim)

    def test_channel_doubling(self):
        """測試通道數加倍。"""
        from pangu.models.upsample_downsample import DownSample

        dim = 192
        downsample = DownSample(dim=dim)

        assert downsample.linear.out_features == 2 * dim


class TestUpSample:
    """UpSample 測試。"""

    def test_forward_shape(self):
        """測試 UpSample 輸出形狀。"""
        from pangu.models.upsample_downsample import UpSample

        input_dim = 384
        output_dim = 192
        upsample = UpSample(input_dim=input_dim, output_dim=output_dim)

        B = 1
        Z, H, W = 8, 180, 91
        x = torch.randn(B, Z * H * W, input_dim)

        with torch.no_grad():
            output = upsample(x)

        # Expected: (B, 8*360*181, 192)
        expected_tokens = 8 * 360 * 181
        assert output.shape == (B, expected_tokens, output_dim)

    def test_channel_halving(self):
        """測試通道數減半。"""
        from pangu.models.upsample_downsample import UpSample

        upsample = UpSample(input_dim=384, output_dim=192)

        assert upsample.linear2.out_features == 192


class TestMlp:
    """Mlp 測試。"""

    def test_forward_shape(self):
        """測試 MLP 前向傳播。"""
        from pangu.models.mlp import Mlp

        dim = 192
        mlp = Mlp(dim)

        x = torch.randn(2, 100, dim)
        output = mlp(x)

        assert output.shape == x.shape
