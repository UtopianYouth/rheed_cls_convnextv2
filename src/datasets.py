"""数据集与数据增强（面向 `main_finetune.py`）。

核心目标：
- 统一构建训练/验证集 `torchvision.datasets.ImageFolder`
- 统一构建 train/val transforms（依赖 timm 的 `create_transform`）
- 处理 RHEED 常见的灰度图：在增强链最前面把 PIL 图像转换成 RGB（3 通道），
  以匹配 ImageNet 的 mean/std 归一化。

你最常用的模式：
- `--data_set image_folder`
  - train: 读取 `--data_path`
  - val: 读取 `--eval_data_path`
"""

import os
from torchvision import datasets, transforms


from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


def _to_rgb(img):
    """Ensure images are 3-channel RGB.

    RHEED images are often grayscale (L). The default normalization here assumes 3 channels.
    """
    try:
        return img.convert("RGB")
    except Exception:
        # If it's already a tensor or non-PIL type, return as-is.
        return img


def build_dataset(is_train, args):
    """构建数据集对象（目前主要服务于微调/分类）。

    Args:
        is_train: True 构建训练集；False 构建验证/测试集。
        args: 命令行参数（至少需要 `data_set`/`data_path`/`eval_data_path`/`nb_classes` 等）。

    Returns:
        (dataset, nb_classes): dataset 为 `torchvision.datasets.*`；nb_classes 为类别数。

    重要约定（RHEED 场景）：
    - 当 `args.data_set == "image_folder"`：
      - train 读取 `args.data_path`
      - val 读取 `args.eval_data_path`
      - 两者的子目录名排序必须一致，否则 `main_finetune.py` 会报 `class_to_idx mismatch`。
    """
    transform = build_transform(is_train, args)


    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    """构建数据增强/预处理 transforms。

    训练集：使用 timm 的 `create_transform`（包含 RandomResizedCrop / RandAug / RandomErase 等）。
    验证集：Resize + CenterCrop + Normalize。

    RHEED 适配：在 transforms 的最前面插入 `_to_rgb`，把灰度图转成 RGB，
    以便后续 `Normalize(mean,std)` 使用 ImageNet 的 3 通道统计量。
    """
    resize_im = args.input_size > 32

    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)

        # Ensure grayscale images (common in RHEED) are converted to RGB before ToTensor/Normalize.
        transform.transforms.insert(0, transforms.Lambda(_to_rgb))
        return transform


    t = [transforms.Lambda(_to_rgb)]
    if resize_im:

        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
