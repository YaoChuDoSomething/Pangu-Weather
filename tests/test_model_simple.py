"""簡化版 PanguModel 測試（避免記憶體問題）。"""

import torch
from pangu.models import create_pangu_model


def test_model_basic():
    """基本測試：模型建立與前向傳播。"""
    print("=" * 60)
    print("PanguModel 基本驗證測試")
    print("=" * 60)
    print()

    # 建立模型
    print("1. 建立模型...")
    model = create_pangu_model()
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ 模型參數: {total_params:,} ({total_params / 1e6:.2f}M)")
    print()

    # 前向傳播測試 (使用小 batch)
    print("2. 前向傳播測試...")
    B = 1
    input_upper = torch.randn(B, 5, 13, 721, 1440)
    input_surface = torch.randn(B, 4, 721, 1440)

    with torch.no_grad():
        output_upper, output_surface = model(input_upper, input_surface)

    # 驗證形狀
    assert output_upper.shape == (B, 5, 13, 721, 1440)
    assert output_surface.shape == (B, 4, 721, 1440)

    print(
        f"   ✓ 輸入: upper{list(input_upper.shape)}, surface{list(input_surface.shape)}"
    )
    print(
        f"   ✓ 輸出: upper{list(output_upper.shape)}, surface{list(output_surface.shape)}"
    )
    print()

    # 檢查輸出值域
    print("3. 輸出統計...")
    print(f"   Upper - mean: {output_upper.mean():.4f}, std: {output_upper.std():.4f}")
    print(
        f"   Surface - mean: {output_surface.mean():.4f}, std: {output_surface.std():.4f}"
    )
    print()

    print("=" * 60)
    print("✓ 所有基本測試通過！")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_model_basic()
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback

        traceback.print_exc()
        raise
