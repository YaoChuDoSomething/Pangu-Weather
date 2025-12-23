# 訓練指南

## 快速開始

```bash
python scripts/train.py --config configs/train.yaml
```

## 配置說明

編輯 `configs/train.yaml`:

```yaml
training:
  epochs: 100
  batch_size: 1  # 大模型建議 1
  learning_rate: 5e-4
  weight_decay: 3e-6
  warmup_epochs: 5
  clip_grad: 1.0
  
  # Checkpoint
  save_every: 10
  checkpoint_dir: checkpoints
  
  # 分散式訓練
  distributed: false
  
  # 混合精度
  use_amp: true
```

## 使用 Python API

```python
from pangu.models import create_pangu_model
from pangu.training import Trainer, PanguLoss, TrainingConfig
from pangu.training import build_train_val_loaders, PanguDataset
import torch.optim as optim

# 建立模型
model = create_pangu_model()

# 配置
config = TrainingConfig(epochs=100, batch_size=1, learning_rate=5e-4)

# 資料集
train_dataset = PanguDataset('data/train', start_year=1979, end_year=2017)
train_loader, val_loader = build_train_val_loaders(train_dataset, batch_size=1)

# 訓練組件
criterion = PanguLoss(surface_weight=0.25)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

# 訓練器
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda',
    use_amp=True
)

# 開始訓練
trainer.fit(train_loader, val_loader, epochs=config.epochs)
```

## 損失函數

根據論文，使用加權 MAE：

```python
loss = MAE(output, target) + MAE(output_surface, target_surface) * 0.25
```

## 分散式訓練

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/train.yaml
```

## 監控訓練

使用 TensorBoard：

```bash
tensorboard --logdir logs/
```

## 恢復訓練

```python
trainer.fit(train_loader, val_loader, resume_from='checkpoints/checkpoint_epoch_50.pth')
```
