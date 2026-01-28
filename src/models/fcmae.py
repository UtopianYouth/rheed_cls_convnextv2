# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""FCMAE（Fully Convolutional Masked AutoEncoder）预训练模型。

整体结构：
- **mask 生成**：按 patch 随机采样 mask（keep=0, mask=1）
- **encoder**：`SparseConvNeXtV2`，只对 keep 的位置做稀疏计算（依赖 MinkowskiEngine）
- **decoder**：把 encoder 输出投影到 decoder_embed_dim，填充 mask_token 后做少量 Block
- **pred**：预测每个 patch 的像素（重建目标），loss 为 mask patch 上的 MSE

在 `main_pretrain.py` 中：
- `loss, pred, mask = model(samples, labels, mask_ratio=...)`
- `labels` 不参与 loss，仅用于兼容 dataloader 输出。
"""

import torch
import torch.nn as nn

from MinkowskiEngine import (

    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)

from timm.models.layers import trunc_normal_
from src.models.convnextv2_sparse import SparseConvNeXtV2
from src.models.convnextv2 import Block

class FCMAE(nn.Module):
    """Fully Convolutional Masked Autoencoder（FCMAE）。

    你可以把它理解为“用 ConvNeXtV2 学一个图像重建任务”：
    - 输入图像被切成 patch 网格
    - 随机 mask 一部分 patch
    - encoder 在稀疏空间里提特征
    - decoder 只做浅层卷积/Block，把 mask 的区域也补齐
    - 用像素重建误差作为训练信号

    预训练完成后，通常只拿 encoder 的权重去初始化分类模型（见 `main_finetune.py --finetune`）。
    """

    def __init__(
                self,
                img_size=224,
                in_chans=3,
                depths=[3, 3, 9, 3],
                dims=[96, 192, 384, 768],
                decoder_depth=1,
                decoder_embed_dim=512,
                patch_size=32,
                mask_ratio=0.6,
                norm_pix_loss=False):
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = SparseConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims, D=2)
        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs):
        """把图片切成 patch 序列。

        Args:
            imgs: shape `(N, 3, H, W)`，要求 `H==W` 且能被 `patch_size` 整除。

        Returns:
            Tensor: shape `(N, L, patch_size**2 * 3)`，其中 `L=(H/patch_size)^2`。

        说明：这里的 patchify/unpatchify 只用于构建重建目标与计算 loss。
        """

        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """把 patch 序列还原回图片。

        Args:
            x: shape `(N, L, patch_size**2 * 3)`。

        Returns:
            Tensor: shape `(N, 3, H, W)`。

        说明：这里假设 patch 网格是正方形（
        `h = w = sqrt(L)`），并且使用与 `patchify` 对称的 reshape/einsum。
        """

        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        """生成随机 mask（patch 级别）。

        mask 定义：
        - 0：keep（保留，可见区域，会进入 encoder）
        - 1：mask（遮挡区域，encoder 不看，decoder 需要重建）

        Args:
            x: 输入图片张量（仅用其 batch size 与空间尺寸推导 patch 数）。
            mask_ratio: 遮挡比例，例如 0.6。

        Returns:
            Tensor: shape `(N, L)`，L 为 patch 数。
        """
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))


        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    
    def forward_encoder(self, imgs, mask_ratio):
        """encoder 前向：生成 mask 后提取特征。

        Args:
            imgs: `(N, 3, H, W)` 输入图像。
            mask_ratio: mask 比例。

        Returns:
            (x, mask):
            - x: encoder 输出的 dense 特征图（已由 sparse densify 回来）
            - mask: `(N, L)` patch mask（0 keep / 1 mask）
        """
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding (sparse ConvNeXtV2)
        x = self.encoder(imgs, mask)
        return x, mask


    def forward_decoder(self, x, mask):
        """decoder 前向：把 encoder 特征投影到像素重建空间。

        步骤：
        1) `proj`: 把 encoder 最后 stage 的通道数投影到 `decoder_embed_dim`
        2) 将 patch 级 mask reshape 成 `(N,1,h,w)`，并在 mask 位置填充 `mask_token`
        3) 经过浅层 `decoder` blocks
        4) `pred`: 输出每个位置对应 patch 的像素预测（通道维为 `patch_size**2 * in_chans`）

        Returns:
            pred: shape `(N, C, h, w)`，后续会 reshape 成 `(N, L, patch_dim)` 来算 loss。
        """
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred


    def forward_loss(self, imgs, pred, mask):
        """计算 MAE 重建损失（MSE，只在 mask patch 上）。

        Args:
            imgs: `[N, 3, H, W]` 原始图片。
            pred: 预测结果，既可能是 `[N, C, h, w]`（卷积输出），也可能是 `[N, L, patch_dim]`。
            mask: `[N, L]`，0=keep, 1=mask。

        Returns:
            loss: 标量张量。

        说明：
        - 若 `norm_pix_loss=True`，会对每个 patch 做归一化后再算 MSE（常见 MAE trick）。
        """

        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        """FCMAE 前向接口（供 `engine_pretrain.py` 调用）。

        Returns:
            loss, pred, mask

        备注：
        - `labels` 参数仅用于兼容 `ImageFolder` dataloader 输出；不参与计算。
        - `mask_ratio` 可以在训练时动态调节（本仓库按 args 固定）。
        """
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def convnextv2_atto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = FCMAE(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = FCMAE(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model