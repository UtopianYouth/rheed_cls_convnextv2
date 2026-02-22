#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COCO 标注基础一致性检查脚本（RHEED 目标检测）
- 检查 images / annotations / categories 是否存在
- 检查 category_id 是否只包含 {1, 2}
- 检查 bbox 格式与数值范围
- 检查 image_id 是否可在 images 中找到

用法：
python check_coco.py --ann /path/to/instances_train.json
"""
import argparse
import json
import sys

ALLOWED_CATEGORIES = {1: "stripe", 2: "spot"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="COCO annotation json path")
    args = parser.parse_args()

    data = load_json(args.ann)
    if "images" not in data or "annotations" not in data or "categories" not in data:
        print("[ERROR] COCO json must contain images/annotations/categories.")
        sys.exit(1)

    images = data["images"]
    annos = data["annotations"]
    cats = data["categories"]

    image_ids = {img["id"] for img in images}
    cat_ids = {c["id"] for c in cats}

    # category id check
    invalid_cat_ids = cat_ids - set(ALLOWED_CATEGORIES.keys())
    if invalid_cat_ids:
        print(f"[ERROR] Invalid category ids found: {invalid_cat_ids}")
        sys.exit(1)

    # annotation check
    for a in annos:
        if a.get("image_id") not in image_ids:
            print(f"[ERROR] annotation image_id not found: {a.get('image_id')}")
            sys.exit(1)
        if a.get("category_id") not in ALLOWED_CATEGORIES:
            print(f"[ERROR] invalid category_id: {a.get('category_id')}")
            sys.exit(1)
        bbox = a.get("bbox")
        if not bbox or len(bbox) != 4:
            print(f"[ERROR] invalid bbox: {bbox}")
            sys.exit(1)
        x, y, w, h = bbox
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            print(f"[ERROR] bbox values out of range: {bbox}")
            sys.exit(1)

    print("[OK] COCO json check passed.")


if __name__ == "__main__":
    main()
