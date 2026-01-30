# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""ConvNeXtV2 的稀疏版本（用于 FCMAE 预训练的 encoder）。

这个实现依赖 `MinkowskiEngine`：
- `to_sparse` 会把密集特征图转换为稀疏张量（只对未 mask 的位置计算），减少计算量。
- 每个 Block 使用 Minkowski 的 depthwise conv / linear / GELU 等。

你通常不需要直接调用它：
- `fcmae.FCMAE` 内部会构建 `SparseConvNeXtV2` 作为 encoder。
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


from src.models.utils import (
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)

# MinkowskiEngine API compatibility:
# - Some builds expose depthwise conv as `MinkowskiChannelwiseConvolution` (ME==0.5.4)
# - Some expose it as `MinkowskiDepthwiseConvolution`
try:
    from MinkowskiEngine import MinkowskiDepthwiseConvolution  # type: ignore
except Exception:
    from MinkowskiEngine import MinkowskiChannelwiseConvolution as MinkowskiDepthwiseConvolution  # type: ignore

from MinkowskiOps import (
    to_sparse,
)

class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., D=3):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(dim, kernel_size=7, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)   
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.grn = MinkowskiGRN(4  * dim)
        self.drop_path = MinkowskiDropPath(drop_path)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 D=3):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D)
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)

    def upsample_mask(self, mask, scale):
        """把 patch 级 mask 上采样到特征图分辨率。

        在 FCMAE 中，mask 是按 patch（例如 32x32）生成的二维网格。
        encoder 的 early feature map 分辨率更高，需要把 mask repeat 到对应分辨率。

        Args:
            mask: shape `(N, L)`，L 为 patch 数（
            scale: 上采样倍率（通常是 2^(num_stages-1)）。

        Returns:
            Tensor: shape `(N, H, W)` 的 mask（1=mask，0=keep）。
        """
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)


    def forward(self, x, mask):
        """稀疏 encoder 前向。

        Args:
            x: 输入图片 `(N, 3, H, W)`。
            mask: patch 级 mask `(N, L)`，0=keep，1=mask。

        Returns:
            Tensor: densify 之后的特征图（dense），供 FCMAE decoder 使用。

        实现思路：
        - 先把 patch 级 mask 上采样到 stem 输出的分辨率，并把被 mask 的位置置零（等价于“移除”）。
        - `to_sparse` 将密集特征图转成稀疏张量（只在非零/有效位置计算）。
        - 经过 4 个 stage 的 Minkowski block 后，再 `dense()` 回到普通张量。
        """
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2**(num_stages-1))        
        mask = mask.unsqueeze(1).type_as(x)
        
        # patch embedding
        x = self.downsample_layers[0](x)
        x *= (1.-mask)
        
        # sparse encoding
        x = to_sparse(x)
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)
        
        # densify
        x = x.dense()[0]
        return x
