# 第二章图表生成指南

本目录包含第二章实验结果的图表生成工具。

## 快速开始

### 1. 一键生成所有图表

```bash
bash scripts/generate_cp2_figures.sh
```

### 2. 查看生成结果

生成的图片会自动保存到：
- `docs/my_graduate_paper/figure/cp2/training_curves.png`
- `docs/my_graduate_paper/figure/cp2/confusion_matrix_test.png`

### 3. 在论文中引用

图片路径已在 `chapter/cp2.tex` 中配置好，直接编译即可：

```bash
cd docs/my_graduate_paper
xelatex master_pang.tex
bibtex master_pang
xelatex master_pang.tex
xelatex master_pang.tex
```

---

## 高级用法

### 手动运行Python脚本

```bash
python -m src.plot_cp2_figures \
    --exp_dir outputs/finetune_timm_20260212_145658 \
    --output_dir docs/my_graduate_paper/figure/cp2 \
    --dpi 300 \
    --normalize_cm
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp_dir` | 实验输出目录 | 必填 |
| `--output_dir` | 图表保存目录 | 必填 |
| `--dpi` | 图片分辨率 | 300 |
| `--normalize_cm` | 混淆矩阵归一化显示 | True |

### 修改实验目录

如果你的实验输出在不同目录，编辑 `scripts/generate_cp2_figures.sh`：

```bash
# 修改这一行
EXP_DIR="outputs/your_experiment_directory"
```

---

## 生成的图表

### 1. 训练曲线图 (`training_curves.png`)

双子图展示：
- **左图**：训练/验证 Loss 曲线
- **右图**：训练/验证 Accuracy 和验证 Macro-F1 曲线

用于分析：
- 模型收敛情况
- 过拟合/欠拟合
- 训练稳定性

### 2. 混淆矩阵 (`confusion_matrix_test.png`)

测试集混淆矩阵（按行归一化），展示：
- 各类别预测准确率
- 误分类模式
- 单类塌缩现象

---

## 故障排查

### 错误：实验目录不存在

```
❌ 错误：实验目录不存在: outputs/finetune_timm_20260212_145658
```

**解决方案**：
1. 检查是否已运行训练脚本
2. 确认实验输出目录名称是否正确
3. 修改 `scripts/generate_cp2_figures.sh` 中的 `EXP_DIR` 变量

### 错误：缺少训练日志文件

```
❌ 错误：训练日志不存在: outputs/.../log.txt
```

**解决方案**：
- 重新运行训练脚本：`bash scripts/run_finetune_gpu.sh`

### 错误：ModuleNotFoundError: No module named 'matplotlib'

**解决方案**：
```bash
conda activate rheed_cls_gpu
conda install matplotlib numpy
# 可选：安装seaborn美化图表
conda install seaborn
```

### 图表显示异常（中文乱码）

**解决方案**：
- 代码已自动处理中文字体回退
- 如果仍有问题，安装中文字体：
  ```bash
  sudo apt-get install fonts-wqy-zenhei
  ```

---

## 文件说明

```
rheed_cls_convnextv2/
├── src/
│   └── plot_cp2_figures.py          # 图表生成核心代码
├── scripts/
│   └── generate_cp2_figures.sh      # 一键执行脚本
├── outputs/
│   └── finetune_timm_20260212_145658/
│       ├── log.txt                  # 训练日志（输入）
│       └── test_metrics.json        # 测试指标（输入）
└── docs/my_graduate_paper/
    └── figure/cp2/
        ├── training_curves.png      # 生成的训练曲线（输出）
        └── confusion_matrix_test.png # 生成的混淆矩阵（输出）
```

---

## 代码特性

✅ **自动化**：一键生成所有论文图表  
✅ **高质量**：300 DPI 矢量图，适合论文印刷  
✅ **鲁棒性**：自动检查输入文件，友好错误提示  
✅ **兼容性**：支持有/无 seaborn 环境  
✅ **可配置**：DPI、归一化、输出路径均可自定义

---

## 维护者

生成时间：2026-02-12  
脚本版本：v1.0  
对应章节：第二章（基于ConvNeXt V2的RHEED时序图像分类）

如有问题，请检查：
1. 训练日志格式是否正确（每行一个JSON）
2. 测试指标文件是否包含 `confusion_matrix` 字段
3. Python 环境是否正确激活
