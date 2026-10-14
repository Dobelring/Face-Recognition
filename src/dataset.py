"""FER2013 数据加载。

支持两种数据布局（--data 指向的目录下二选一，自动识别）:
1. CSV 版:  fer2013.csv（列: emotion, pixels, Usage）或 icml_face_data.csv
2. 图片版:  train/<类别名>/xxx.jpg 与 test/<类别名>/xxx.jpg
   类别子目录支持英文名（angry 等 7 类）或数字名（0~6）

所有标签统一映射回 FER2013 官方顺序（见 models/labels.py）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

from models.labels import EMOTION_CLASSES

IMG_SIZE = 48

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomCrop(IMG_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize([0.5], [0.5]),  # -> [-1,1]
    ]
)

# 推理/评估统一使用该变换；recognizer.py 也复用它，保证训练与推理预处理一致
EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


class _RemapTargets:
    """把 ImageFolder 的字典序标签映射回 FER2013 官方类别顺序（可被 pickle）。"""

    def __init__(self, mapping: dict):
        self.mapping = dict(mapping)

    def __call__(self, target: int) -> int:
        return self.mapping[target]


class Fer2013CSVDataset(Dataset):
    """CSV 版 FER2013：pixels 列为 2304 个空格分隔的灰度像素值。"""

    def __init__(self, df: pd.DataFrame, transform):
        pixels = df["pixels"].astype(str).tolist()
        self.images = np.stack(
            [np.array(p.split(), dtype=np.uint8).reshape(IMG_SIZE, IMG_SIZE) for p in pixels]
        )
        self.labels = df["emotion"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.images[idx], mode="L")
        return self.transform(img), int(self.labels[idx])


def _canonical_index(folder_name: str) -> int:
    name = folder_name.strip().lower()
    if name in EMOTION_CLASSES:
        return EMOTION_CLASSES.index(name)
    if name.isdigit():
        return int(name)
    raise ValueError(
        f"无法识别的类别目录名: {folder_name!r}，请使用 {'/'.join(EMOTION_CLASSES)} 或 0~6 命名"
    )


def _find_csv(root: Path) -> Path:
    candidates = sorted(root.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"{root} 下未找到 CSV 文件")
    for c in candidates:  # 优先匹配常见的命名
        if "fer2013" in c.name.lower() or "icml" in c.name.lower():
            return c
    return candidates[0]


def _folder_class_map(class_to_idx: dict) -> dict:
    return {orig: _canonical_index(name) for name, orig in class_to_idx.items()}


def load_datasets(data_root: str | Path, val_fraction: float = 0.1, seed: int = 42):
    """返回 (train_ds, val_ds, test_ds)。

    CSV 版: 按 Usage 列划分（Training / PublicTest / PrivateTest）。
    图片版: train/ 内部按比例切出验证集，test/ 作为测试集。
    """
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}，请先下载数据集（见 data/README.md）")

    if any(root.glob("*.csv")):
        csv_path = _find_csv(root)
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]  # icml 文件列名带前导空格
        df["Usage"] = df["Usage"].astype(str).str.strip()
        by_usage = {
            "train": (df[df["Usage"] == "Training"], TRAIN_TRANSFORM),
            "val": (df[df["Usage"] == "PublicTest"], EVAL_TRANSFORM),
            "test": (df[df["Usage"] == "PrivateTest"], EVAL_TRANSFORM),
        }
        return tuple(Fer2013CSVDataset(d, tf) for d, tf in by_usage.values())

    train_dir, test_dir = root / "train", root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"{root} 下既没有 CSV 也没有 train/test 子目录，数据布局见 data/README.md"
        )

    # 用 train 目录确定类别映射，三份子集共用同一映射
    probe = datasets.ImageFolder(str(train_dir))
    mapper = _RemapTargets(_folder_class_map(probe.class_to_idx))

    base_train = datasets.ImageFolder(str(train_dir), transform=TRAIN_TRANSFORM, target_transform=mapper)
    base_eval = datasets.ImageFolder(str(train_dir), transform=EVAL_TRANSFORM, target_transform=mapper)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(base_train), generator=g).tolist()
    n_val = max(1, int(len(base_train) * val_fraction))
    train_ds = Subset(base_train, idx[n_val:])
    val_ds = Subset(base_eval, idx[:n_val])
    test_ds = datasets.ImageFolder(str(test_dir), transform=EVAL_TRANSFORM, target_transform=mapper)
    return train_ds, val_ds, test_ds
