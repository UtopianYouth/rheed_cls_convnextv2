"""模型定义子包（`src/models`）。

这里包含两条主线：
- **分类（微调）模型**：`convnextv2.py`（标准 dense ConvNeXtV2）
- **自监督预训练模型**：`fcmae.py`（Fully Convolutional Masked AutoEncoder, FCMAE）
  - 其 encoder 使用 `convnextv2_sparse.py`（基于 MinkowskiEngine 的稀疏卷积实现）

你在复现时可以记住：
- `main_pretrain.py` 会从这里构建 `fcmae.<model_name>`
- `main_finetune.py` 会从这里构建 `convnextv2.<model_name>`
"""
