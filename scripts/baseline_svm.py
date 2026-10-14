"""传统方法对比基线：HOG 特征 + 线性 SVM。

用于课设报告中与 CNN 方法做对比分析（精度差距、各类别表现差异）。

用法（在项目根目录）:
  uv run python scripts/baseline_svm.py
  uv run python scripts/baseline_svm.py --max-train 8000   # 抽样训练，加快速度

产出:
  results/baseline_svm_report.txt       分类报告
  results/confusion_matrix_svm.png      混淆矩阵图
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.labels import EMOTION_CLASSES, cn
from skimage.feature import hog
from src.dataset import load_datasets
from src.plot_utils import save_confusion_matrix


def extract_hog(loader) -> tuple[np.ndarray, np.ndarray]:
    """遍历数据集（已归一化到 [-1,1]），反归一化后提取 HOG 特征。"""
    xs, ys = [], []
    for images, targets in tqdm(loader, desc="提取HOG", ncols=90):
        imgs = ((images.numpy()[:, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
        for img, y in zip(imgs, targets.numpy()):
            feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            xs.append(feat)
            ys.append(int(y))
    return np.asarray(xs), np.asarray(ys)


def main():
    ap = argparse.ArgumentParser(description="HOG+SVM 对比基线")
    ap.add_argument("--data", default="data/fer2013")
    ap.add_argument("--max-train", type=int, default=0, help="训练子集上限，0=全部")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    train_ds, _, test_ds = load_datasets(args.data)
    train_loader = DataLoader(train_ds, batch_size=256, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=0)

    X_train, y_train = extract_hog(train_loader)
    X_test, y_test = extract_hog(test_loader)
    if args.max_train and len(X_train) > args.max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), args.max_train, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    print(f"训练 SVM: {X_train.shape[0]} 样本 x {X_train.shape[1]} 维特征")
    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto"))
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"训练完成，用时 {time.time() - t0:.0f}s")

    y_pred = clf.predict(X_test)
    acc = float((y_pred == y_test).mean())
    names = [f"{en}({cn(en)})" for en in EMOTION_CLASSES]
    labels7 = list(range(len(names)))
    report = classification_report(
        y_test, y_pred, labels=labels7, target_names=names, digits=4, zero_division=0
    )
    print(f"\n[HOG+SVM 基线] 测试准确率: {acc:.4f}\n\n{report}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "baseline_svm_report.txt").write_text(
        f"HOG+SVM 测试准确率: {acc:.4f}\n训练样本: {len(y_train)}\n\n{report}", encoding="utf-8"
    )
    save_confusion_matrix(
        confusion_matrix(y_test, y_pred, labels=labels7),
        names,
        out_dir / "confusion_matrix_svm.png",
        title=f"HOG+SVM 基线混淆矩阵（准确率 {acc:.1%}）",
    )
    print(f"评估报告: {out_dir / 'baseline_svm_report.txt'}")
    print(f"混淆矩阵: {out_dir / 'confusion_matrix_svm.png'}")


if __name__ == "__main__":
    main()
