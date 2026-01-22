import os
from pathlib import Path
import numpy as np
from PIL import Image

def make_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_rand_jpg(path: Path, size=224):
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=95)

def main():
    root = Path("data_row/fake_imagenet")
    for split, n_per_class in [("train", 20), ("val", 10)]:
        for cls in ["class0", "class1"]:
            d = root / split / cls
            make_dir(d)
            for i in range(n_per_class):
                save_rand_jpg(d / f"{i:06d}.jpg", size=224)

    print(f"Fake ImageNet created at: {root.resolve()}")

if __name__ == "__main__":
    main()
