#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_DIR=${1:-"data_rheed/train"}
OUTDIR=${OUTDIR:-"outputs/dataset_check_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$OUTDIR"

echo "data_dir=$DATA_DIR"
echo "outdir=$OUTDIR"

echo "python: $(command -v python)"
python -V

python - <<PY
from pathlib import Path
import sys

DATA_DIR = Path("$DATA_DIR")
OUTDIR = Path("$OUTDIR")

try:
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image
except Exception as e:
    print("[FAIL] missing dependency in current env:", repr(e))
    print("Tip: make sure you activated the correct conda env (the one that has numpy/torch/torchvision/Pillow).")
    sys.exit(2)

if not DATA_DIR.exists():
    print(f"[FAIL] data dir not found: {DATA_DIR}")
    sys.exit(3)

files = sorted(list(DATA_DIR.rglob('*.tiff')) + list(DATA_DIR.rglob('*.tif')))
print("files", len(files))
if not files:
    print("[FAIL] no tif/tiff found")
    sys.exit(4)

# Mirror main_pretrain's preprocessing (excluding random crop/flip)
post = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def to_rgb(im):
    try:
        return im.convert('RGB')
    except Exception:
        return im

bad = []
out = []
mode_cnt = {}

for i, p in enumerate(files, 1):
    try:
        with Image.open(p) as im:
            mode_cnt[im.mode] = mode_cnt.get(im.mode, 0) + 1

            # raw array stats
            a = np.array(im)
            fin_raw = bool(np.isfinite(a).all())
            mn_raw = float(np.nanmin(a))
            mx_raw = float(np.nanmax(a))

            # tensor stats (rgb + toTensor + normalize)
            im2 = to_rgb(im)
            x = post(im2)
            fin_x = bool(torch.isfinite(x).all().item())
            mn_x = float(x.min().item())
            mx_x = float(x.max().item())

            if (not fin_raw) or (not fin_x):
                bad.append((str(p), im.mode, fin_raw, fin_x, mn_raw, mx_raw, mn_x, mx_x))

            # extreme normalized values are suspicious (typically within ~[-3, +3])
            if abs(mn_x) > 50 or abs(mx_x) > 50:
                out.append((max(abs(mn_x), abs(mx_x)), str(p), im.mode, mn_x, mx_x, mn_raw, mx_raw))

    except Exception as e:
        bad.append((str(p), 'EXC', repr(e)))

    if i % 500 == 0:
        print("scanned", i)

print("modes", sorted(mode_cnt.items(), key=lambda kv: -kv[1])[:10])
print("nonfinite_or_exc", len(bad))
print("extreme_norm_outliers", len(out))

bad_path = OUTDIR / 'bad_files.txt'
out_path = OUTDIR / 'outliers.txt'

with bad_path.open('w', encoding='utf-8') as f:
    for r in bad:
        f.write(str(r) + '\n')
with out_path.open('w', encoding='utf-8') as f:
    for r in sorted(out, reverse=True):
        f.write(str(r) + '\n')

print('written', bad_path, out_path)

# If we found truly non-finite values or exceptions, return non-zero
if bad:
    sys.exit(10)
PY

echo "Done."