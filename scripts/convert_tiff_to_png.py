"""Convert TIFF images under a directory tree to PNG, preserving folder structure."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


SUPPORTED_EXTS = (".tif", ".tiff")


def parse_args():
    parser = argparse.ArgumentParser("Convert TIFF to PNG")
    parser.add_argument("--src", type=str, required=True, help="源目录（包含tif/tiff）")
    parser.add_argument("--dst", type=str, required=True, help="目标目录（输出png）")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有png")
    return parser.parse_args()


def _convert_one(src_path: Path, dst_path: Path, overwrite: bool):
    if dst_path.exists() and not overwrite:
        return "skip"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        img.save(dst_path, format="PNG", optimize=True)
    return "ok"


def main():
    args = parse_args()
    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise SystemExit(f"src not found: {src_root}")

    tiff_files = [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not tiff_files:
        raise SystemExit("no tif/tiff found under src")

    total = len(tiff_files)
    print(f"Found {total} tiff files. Converting to png...")

    ok = 0
    skip = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = []
        for src_path in tiff_files:
            rel = src_path.relative_to(src_root)
            dst_path = (dst_root / rel).with_suffix(".png")
            futures.append(executor.submit(_convert_one, src_path, dst_path, args.overwrite))

        for i, fut in enumerate(as_completed(futures), 1):
            status = fut.result()
            if status == "ok":
                ok += 1
            else:
                skip += 1
            if i % 500 == 0 or i == total:
                print(f"Progress: {i}/{total} | ok={ok} skip={skip}")

    print(f"Done. ok={ok}, skip={skip}, output={dst_root}")


if __name__ == "__main__":
    main()
