"""測試 PanguModel 的基本功能。"""

import torch
from pangu.models import PanguModel, create_pangu_model


def test_pangu_model_creation():
    """測試模型建立。"""
    model = create_pangu_model()
    assert model is not None
    assert isinstance(model, PanguModel)
    print("✓ 模型建立成功")


def test_pangu_model_forward():
    """測試前向傳播。"""
    # 建立模型
    model = create_pangu_model()
    model.eval()

    # 建立測試輸入（小尺寸以加快測試）
    B = 1
    input_upper = torch.randn(B, 5, 13, 721, 1440)
    input_surface = torch.randn(B, 4, 721, 1440)

    # 前向傳播
    with torch.no_grad():
        output_upper, output_surface = model(input_upper, input_surface)

    # 驗證輸出形狀
    assert output_upper.shape == (B, 5, 13, 721, 1440), (
        f"Expected (B, 5, 13, 721, 1440), got {output_upper.shape}"
    )
    assert output_surface.shape == (B, 4, 721, 1440), (
        f"Expected (B, 4, 721, 1440), got {output_surface.shape}"
    )

    print("✓ 前向傳播成功")
    print(f"  輸入 upper: {input_upper.shape}")
    print(f"  輸入 surface: {input_surface.shape}")
    print(f"  輸出 upper: {output_upper.shape}")
    print(f"  輸出 surface: {output_surface.shape}")


def test_model_parameters():
    """測試模型參數數量。"""
    model = create_pangu_model()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✓ 模型參數統計:")
    print(f"  總參數: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}")
    print(f"  參數量: {total_params / 1e6:.2f}M")

    assert total_params > 0
    assert trainable_params > 0


def test_gradient_flow():
    """測試梯度流動。"""
    model = create_pangu_model()
    model.train()

    # 建立測試輸入
    B = 1
    input_upper = torch.randn(B, 5, 13, 721, 1440, requires_grad=True)
    input_surface = torch.randn(B, 4, 721, 1440, requires_grad=True)

    # 前向傳播
    output_upper, output_surface = model(input_upper, input_surface)

    # 計算簡單損失
    loss = output_upper.mean() + output_surface.mean()

    # 反向傳播
    loss.backward()

    # 檢查梯度
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    assert has_grad, "模型參數沒有梯度"

    print("✓ 梯度流動正常")


if __name__ == "__main__":
    print("=" * 60)
    print("開始測試 PanguModel")
    print("=" * 60)
    print()

    try:
        test_pangu_model_creation()
        print()

        test_model_parameters()
        print()

        test_pangu_model_forward()
        print()

        test_gradient_flow()
        print()

        print("=" * 60)
        print("✓ 所有測試通過！")
        print("=" * 60)

    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ 測試失敗: {e}")
        print("=" * 60)
        import traceback

        traceback.print_exc()
        raise
