"""本项目训练代码包（`src`）。

你可以把 `src` 理解为“可复现实验的最小训练框架”，包含：
- **入口脚本**：`main_pretrain.py`（预训练）、`main_finetune.py`（微调/分类）
- **数据**：`datasets.py`（构建 `torchvision.datasets.ImageFolder` + 增强）
- **训练循环**：`engine_pretrain.py`、`engine_finetune.py`
- **模型**：`models/`（ConvNeXtV2 分类模型、FCMAE 预训练模型等）
- **通用工具**：`utils.py`（分布式初始化、日志、保存/恢复、LR 调度等）

常用运行方式：
- `python -m src.main_finetune ...`
- `torchrun --standalone --nproc_per_node=4 -m src.main_pretrain ...`

提示：预训练（FCMAE）依赖 `MinkowskiEngine`；微调仅依赖常规 PyTorch/timm。
"""

