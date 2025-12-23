# 推論指南

## 快速開始

```bash
python scripts/inference.py --input data/input.nc --output data/output.nc
```

## 使用 Python API

```python
import torch
from pangu.models import create_pangu_model
from pangu.inference import WeatherPreprocessor, WeatherPostprocessor

# 載入模型
model = create_pangu_model()
model.load_state_dict(torch.load('checkpoints/best_model.pth')['model_state_dict'])
model.eval()

# 前處理器
preprocessor = WeatherPreprocessor()
preprocessor.load_stats('data/stats.npz')  # 載入正規化統計

# 後處理器
postprocessor = WeatherPostprocessor()
postprocessor.load_stats('data/stats.npz')

# 準備輸入
import numpy as np
upper = np.load('data/upper.npy')      # (5, 13, 721, 1440)
surface = np.load('data/surface.npy')  # (4, 721, 1440)

upper_t, surface_t = preprocessor.prepare_input(upper, surface, device='cuda')

# 推論
with torch.no_grad():
    out_upper, out_surface = model(upper_t, surface_t)

# 後處理
upper_np, surface_np = postprocessor.process_output(out_upper, out_surface)
```

## 自動回歸預報

對於多時刻預報（如 7 天），使用自動回歸方式：

```python
# 6 小時 → 7 天 = 28 步
outputs = []
current_upper, current_surface = upper_t, surface_t

for step in range(28):
    with torch.no_grad():
        out_upper, out_surface = model(current_upper, current_surface)
    
    outputs.append((out_upper.cpu(), out_surface.cpu()))
    
    # 將輸出作為下一步輸入
    current_upper, current_surface = out_upper, out_surface

print(f"完成 {len(outputs)} 步預報 (7 天)")
```

## ONNX 推論

如果使用 ONNX 模型：

```python
import onnxruntime as ort

session = ort.InferenceSession('models/pangu_6h.onnx')

# 準備輸入
inputs = {
    'input_upper': upper.astype(np.float32),
    'input_surface': surface.astype(np.float32)
}

# 執行推論
outputs = session.run(None, inputs)
out_upper, out_surface = outputs
```

## 輸入輸出規格

**輸入**:
- `upper`: (B, 5, 13, 721, 1440) float32
- `surface`: (B, 4, 721, 1440) float32

**輸出**:
- `output_upper`: (B, 5, 13, 721, 1440) float32  
- `output_surface`: (B, 4, 721, 1440) float32

## 效能提示

1. 使用 `torch.cuda.amp` 半精度推論
2. 批次處理多個時刻
3. 考慮使用 ONNX Runtime 加速
