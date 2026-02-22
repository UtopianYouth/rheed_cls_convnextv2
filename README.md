# RHEED图像表面状态监测 - ConvNeXtV2

基于ConvNeXtV2的RHEED图像分类，用于表面状态监测（光滑 vs 粗糙）

## 项目概述

**任务**: 多任务RHEED图像分析 - 表面状态分类分支  
**方法**: ConvNeXtV2 迁移学习  
**类别**: 二分类 (光滑表面 vs 粗糙表面)

## 数据集

- **总计**: 3241张 RHEED图像
- **光滑类**: 2570张 (79.3%)
  - 二维层状生长 (layer_smooth): 2003张
  - 三维岛状初期 (island_smooth): 567张
- **粗糙类**: 671张 (20.7%)
  - 三维岛状后期 (island_rough): 671张

详细数据划分说明请参考 [DATA_SPLIT_README.md](DATA_SPLIT_README.md)

## 快速开始

### 1. 环境配置

**注意**：项目已升级到PyTorch 2.2 + CUDA 12.4以支持MinkowskiEngine编译

```bash
# 创建conda环境
conda env create -f env/environment.yml
conda activate rheed_cls_gpu

# 或更新现有环境
conda env update -f env/environment.yml

# 若清华镜像的nvidia channel出现404, 可将env/environment.yml中的nvidia渠道改为官方源
# https://conda.anaconda.org/nvidia
# 若pytorch版本仍为2.1.2, 可能是pytorch channel优先级高导致，已改为conda-forge优先

# 编译MinkowskiEngine (服务器CUDA环境)
cd MinkowskiEngine
pip install -v --no-build-isolation --no-deps .

# 环境验证
# 本地CPU环境验证
bash scripts/test_cpu_env.sh

# 服务器GPU环境验证
bash scripts/test_gpu_env.sh
```

### 2. 数据集划分

```bash
# 分析原始数据
python scripts/analyze_dataset.py

# 划分数据集 (70% train / 15% val / 15% test)
python scripts/split_dataset_simple.py
```

### 3. 训练模型

```bash
# GPU训练 (推荐 - 服务器环境)
bash scripts/run_finetune_balanced_gpu.sh

# CPU训练 (本地调试用 - Ubuntu环境)
bash scripts/run_finetune_cpu.sh
```

### 4. 查看训练进度

```bash
tensorboard --logdir outputs/
```

## 关键特性

- ✅ **时序感知划分**: 考虑连续视频帧特性，避免数据泄露
- ✅ **类别平衡策略**: 应对3.83:1的类别不平衡
- ✅ **丰富数据增强**: Mixup, CutMix, RandomErase, LabelSmoothing
- ✅ **迁移学习**: 基于ImageNet预训练的ConvNeXtV2
- ✅ **完整评估指标**: Accuracy, Precision, Recall, F1, 混淆矩阵

## 项目结构

```bash
rheed_cls_convnextv2/
├── data_rheed/              # 原始数据
│   ├── layer_smooth/        # 二维层状生长(光滑)
│   ├── island_smooth/       # 三维岛状初期(光滑)
│   └── island_rough/        # 三维岛状后期(粗糙)
├── data_rheed_split/        # 划分后数据(运行脚本后生成)
│   ├── train/
│   ├── val/
│   └── test/
├── scripts/                 # 训练和分析脚本
│   ├── split_dataset_simple.py
│   ├── analyze_dataset.py
│   ├── run_finetune_cpu.sh           # CPU训练(本地)
│   └── run_finetune_balanced_gpu.sh  # GPU训练(服务器)
├── src/                     # 源代码
│   ├── main_finetune.py
│   ├── datasets.py
│   └── engine_finetune.py
├── outputs/                 # 训练输出
└── DATA_SPLIT_README.md     # 详细说明文档
```

## 技术说明

### 数据划分策略

**核心约束**: `island_smooth` 和 `island_rough` 来自同一生长过程的连续视频帧，不能随机打乱

**解决方案**: 混合时序策略

- 二维层状: 随机划分 (70/15/15)
- 三维岛状: 按时序划分，保持生长过程完整性

### 类别不平衡处理

1. **类别权重**: 在损失函数中使用加权交叉熵
2. **数据增强**: Mixup(0.8) + CutMix(1.0)
3. **评估指标**: 使用Macro F1而非仅看准确率

### 关键概念

- **Mixup**: 数据增强，通过线性插值创建虚拟训练样本
- **EMA**: 指数移动平均，用于获得更稳定的模型
- **AMP**: 自动混合精度训练，加速训练
- **Layer Decay**: 对不同层使用不同学习率

## 论文撰写建议

- **诚实报告数据限制**: 说明类别不平衡和样本量

- **解释划分策略**: 强调时序约束的重要性

- **完整评估指标**: 不能只报告总体准确率

- **消融实验**: 对比有 or 无类别权重、数据增强等

详见 [DATA_SPLIT_README.md](DATA_SPLIT_README.md)

## 预期结果

基于类似规模的小样本图像分类任务：
- 总体准确率: 85-92%
- 光滑类F1: 90-95%
- 粗糙类F1: 70-85% (受限于样本量)

## 进一步改进

1. **补充数据** (最有效): 收集更多粗糙类样本，目标>1000张
2. **更大模型**: 尝试 convnextv2_base/large
3. **集成学习**: 训练多个模型并投票
4. **半监督学习**: 利用未标注RHEED图像

## 引用

```bibtex
@article{woo2023convnext,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Woo, Sanghyun and Debnath, Shoubhik and Hu, Ronghang and Chen, Xinlei and Liu, Zhuang and Kweon, In So and Xie, Saining},
  journal={arXiv preprint arXiv:2301.00808},
  year={2023}
}
```


请你帮我分析我的src源代码中convnextv2的预训练和微调的代码结构，我目前正在使用convnextv2深度学习框架完成我的硕士毕业论文rheed图像分类任务，我的数据集在data_rheed目录中；现在我所拥有的基础开发环境如下：

1. 在我本机开发环境中，是仅有cpu的Ubuntu20.04，我在本机完成代码和脚本等的编写，编写完成之后，我会通过我的脚本将源代码、数据集和conda虚拟环境上传到4卡rtx4090的离线服务器中，在我的服务器中完成模型的预训练和微调复现工作；

2. 在离线服务器中，由于我的服务器无法连接外部网络，所以我得在我的开发环境下，提前安装好代码复现所需要的所有依赖，然后运行我的conda虚拟环境打包脚本，将其打包之后上传到服务器中，最后在服务器解包，完成conda虚拟环境的迁移；

3. 基于上述两点，我希望你帮我详细分析src目录下的源码，这里包括了convnextv2的预训练和微调代码，基于代码所需要的运行环境，协助我完成在本机创建一个新的conda虚拟环境，然后安装必要的第三方库，如非必要，尽量所有的第三方库都使用conda install的方式安装，安装第三方库成功后，我会自己进行打包；

4. 注意：安装第三方库一定要全，且我是在离线4卡rtx 4090服务器中运行我的预训练和微调代码，我的服务器cuda信息如下：
(base) omnisky@omnisky-AS-4124GS-TNR:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
(base) omnisky@omnisky-AS-4124GS-TNR:~$ conda env list



角色
你是一名专业的计算机视觉领域的硕士生导师，并且给学生了一个偏向于学科交叉的领域作为研究，主要是半导体材料的分子束外延技术自动化研究，依靠RHEED的图像，实现外延过程中的自动监控。

任务
我的本机只用CPU，主要完成修改代码，数据集的构建，实验结果的可视化展示以及使用latex撰写我的硕士毕业论文，我的硕士毕业论文的latex项目，在当前项目根目录的docs/my_graduate_paper中，这里存放了我们学校专属的latex毕业论文模板，你最重要的任务是辅导我完成我的毕业论文撰写和给出实验的最快运行成功方式，我的主要硕士毕业论文主要内容是偏向于应用层的，利用传统的视觉和时序预测模型完成分类，目标检测和时序预测任务，就像我目前，跑通了基于convnextv2的分类实验，是基于timm库直接调用模型的，而不是基于convnextv2源码跑通的，在给我提供实验设计和运行时，请以最方便直接的方式，而不是基于方法源码的复现，如果有稳定可用的模型，直接在我的rheed_images图像数据集中，进行微调即可，我已经在源码复现中浪费了太多时间，会导致conda虚拟环境的爆炸，而且不会得到有用的结果，目前我的硕士毕业论文截至日期是3周后，也就是我不能在实验上浪费太多时间，主线任务是完成毕业论文所有内容的撰写，关于实验数据部分，如果实验结果不是很理想，我会遵从事实即可。

最最重要的事情：辅导我完成硕士毕业论文的撰写，基于我已有的初稿思路下，帮助我详细地描述论文中所使用方法章节的方法原理，阐述清楚为什么使用该方法，实验设计和详细流程，实验结果分析以及本章小结。

关于实验结果：由于数据缺陷的原因，本文中使用的方法在数据集下，存在结果不理想的状态，但是当前时间节点下，最重要的是先完成实验的运行成功，所以一再强调，给我提供最快速的实验复现和跑通思路是原则，而不是把大量时间浪费在复现上（conda虚拟环境的配置需要花费大量的时间来找正确的版本，会导致我的撰写毕业论文的时间严重压缩）。

目前我的硕士毕业论文整体框架已经基本确定，主要的框架不需要更改了，关于具体地论文整体架构，您需要详细分析我的硕士毕业论文所有内容，也就是docs/my_graduate_paper下的所有文件。

该计算机专业硕士生拥有的科研和实验复现条件

硬件
在本机开发环境中，是仅有cpu的Ubuntu20.04，离线服务器拥有一台8卡的A100 80GB，并且安装了cuda13.0，拥有conda环境，针对于我的的硕士毕业论文所使用的方法，一张A100足以将我的所有方法复现成功。

数据集
目前我已有的数据集在项目根目录下的rheed_images文件夹中，为了完成二维层状生长和三维岛状生长的分类任务，已经分成了序列图像。

硬性要求
在每一次生成代码之前，请先详细分析代码，要有全局观，尽量不要在源代码的基础上进行堆积，导致在代码修改的过程中，生成太多没必要冗余的代码，代码的修改应该针毕业论文初稿中所需要复现的目标实验。

关于bash命令执行和代码修改
在代码修改和命令执行之前，你都不应该自动操作，而是需要经过我的允许才能执行，如果我选择不允许，您可以不用终端此次对话，而是等我继续补充描述，基于继续补充的描述再继续下面的任务。
其次，对于给出的多条命令建议，如果多条命令之间没有先后依赖关系，可用使用&&符号整合为一条shell命令。

conda虚拟环境激活
有时候，在验证import之前，请先执行conda activate rheed_cls_gpu命令。



