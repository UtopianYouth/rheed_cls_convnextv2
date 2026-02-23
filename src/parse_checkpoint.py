#!/usr/bin/env python3
"""
解析checkpoint（预训练或微调，仅输出关键信息）。

用法：
    python -m src.parse_ssl_checkpoint --ckpt_path outputs/pretrain_ssl_20260221_151605/checkpoint_last.pth
"""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="解析checkpoint（预训练/微调通用）")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="checkpoint路径（.pth）",
    )
    return parser.parse_args()


def summarize_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return "<not a state_dict>"
    keys = list(state_dict.keys())
    num_keys = len(keys)
    first_keys = keys[:10]
    return {
        "num_keys": num_keys,
        "first_keys": first_keys,
    }


def summarize_checkpoint(ckpt, ckpt_path):
    print("=== Checkpoint summary ===")
    print(f"Path: {ckpt_path}")
    print(f"Top-level keys: {list(ckpt.keys())}")

    epoch = ckpt.get("epoch", None)
    print(f"Epoch: {epoch}")

    args_dict = ckpt.get("args", None)
    if isinstance(args_dict, dict):
        print(f"Args keys: {list(args_dict.keys())[:15]}")
    else:
        print("Args keys: N/A")

    candidate_keys = [
        "model",
        "ssl_model",
        "state_dict",
        "model_state",
        "model_ema",
    ]

    for key in candidate_keys:
        if key in ckpt:
            summary = summarize_state_dict(ckpt.get(key))
            print(f"State dict [{key}]: {summary}")


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint不存在：{ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        summarize_checkpoint(ckpt, ckpt_path)
    else:
        print("=== Checkpoint summary ===")
        print(f"Path: {ckpt_path}")
        print("Checkpoint is not a dict, cannot summarize.")


if __name__ == "__main__":
    main()
