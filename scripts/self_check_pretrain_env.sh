#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# reduce noisy warnings / avoid CPU oversubscription during checks
: "${OMP_NUM_THREADS:=8}"
: "${MKL_NUM_THREADS:=8}"
export OMP_NUM_THREADS MKL_NUM_THREADS

DATA_PATH=""
ALLOW_CPU=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/self_check_pretrain_env.sh [--data-path PATH] [--allow-cpu]

What it checks:
  - Python imports: torch / torchvision / timm / tensorboardX / MinkowskiEngine
  - CUDA availability in torch
  - CUDA availability in MinkowskiEngine
  - timm API: timm.optim.optim_factory.add_weight_decay
  - (optional) dataset layout: PATH/train exists

Notes:
  - Recommended to run after activating your packed env via:
      CONDA_BASE=$(conda info --base)
      source "$CONDA_BASE/envs/rheed_cls_gpu/bin/activate"
      conda-unpack
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-path)
      DATA_PATH="${2:-}"; shift 2 ;;
    --allow-cpu)
      ALLOW_CPU=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

echo "== Basic info =="
echo "pwd: $(pwd)"
echo "python: $(command -v python || true)"
python -V || true

echo

echo "== Python package / CUDA checks =="
python - <<'PY'
import os
import sys

def _try_import(name):
    try:
        mod = __import__(name)
        return mod, None
    except Exception as e:
        return None, e

pkgs = ["torch", "torchvision", "timm", "tensorboardX", "MinkowskiEngine"]
loaded = {}
failed = {}
for p in pkgs:
    mod, err = _try_import(p)
    if err is None:
        loaded[p] = mod
    else:
        failed[p] = repr(err)

if failed:
    print("[FAIL] import errors:")
    for k, v in failed.items():
        print(f"  - {k}: {v}")
    sys.exit(10)

import torch
import torchvision
import timm
import tensorboardX
import MinkowskiEngine as ME

print(f"torch: {torch.__version__} | torch.version.cuda={torch.version.cuda} | cuda.is_available={torch.cuda.is_available()}")
print(f"torchvision: {torchvision.__version__}")
print(f"timm: {timm.__version__}")
print(f"tensorboardX: {getattr(tensorboardX, '__version__', 'unknown')}")
print(f"MinkowskiEngine: {ME.__version__} | ME.is_cuda_available={ME.is_cuda_available()}")

try:
    import timm.optim.optim_factory as of
    has_add = hasattr(of, "add_weight_decay")
    has_pgw = hasattr(of, "param_groups_weight_decay")
    print(f"timm.optim.optim_factory.add_weight_decay: {has_add}")
    print(f"timm.optim.optim_factory.param_groups_weight_decay: {has_pgw}")
    if not (has_add or has_pgw):
        sys.exit(11)
except Exception as e:
    print(f"[FAIL] timm.optim.optim_factory import failed: {e!r}")
    sys.exit(12)

if torch.cuda.is_available():
    try:
        n = torch.cuda.device_count()
        print(f"cuda.device_count: {n}")
        for i in range(min(n, 8)):
            print(f"  - cuda:{i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"[WARN] cuda device query failed: {e!r}")

print("[OK] core imports and APIs look good")
PY

if [[ -n "$DATA_PATH" ]]; then
  echo
  echo "== Dataset layout check =="
  if [[ ! -d "$DATA_PATH" ]]; then
    echo "[FAIL] --data-path not found: $DATA_PATH" >&2
    exit 20
  fi
  if [[ ! -d "$DATA_PATH/train" ]]; then
    echo "[FAIL] missing train/ under: $DATA_PATH" >&2
    echo "Expected: $DATA_PATH/train/<any_class_name>/*" >&2
    exit 21
  fi
  echo "[OK] found: $DATA_PATH/train"
fi

# Optional strictness: require CUDA unless --allow-cpu
if [[ "$ALLOW_CPU" -eq 0 ]]; then
  echo
  echo "== CUDA strict check =="
  python - <<'PY'
import sys
import torch
import MinkowskiEngine as ME
if not torch.cuda.is_available():
    print("[FAIL] torch.cuda.is_available() is False")
    sys.exit(30)
if not ME.is_cuda_available():
    print("[FAIL] MinkowskiEngine ME.is_cuda_available() is False")
    sys.exit(31)
print("[OK] CUDA is available for both torch and MinkowskiEngine")
PY
fi

echo

echo "== (Optional) main_pretrain import check =="
python -c "import importlib; importlib.import_module('src.main_pretrain'); print('[OK] import src.main_pretrain')"

echo

echo "All checks passed."
