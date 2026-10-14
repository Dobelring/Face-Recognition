"""在测试集上评估：准确率、分类报告、混淆矩阵。

用法（在项目根目录）:
  uv run python scripts/evaluate.py
  uv run python scripts/evaluate.py --weights weights/best_model.pth --model cnn

产出:
  results/evaluation_report.txt      分类报告（各类精确率/召回率/F1）
  results/confusion_matrix.png       混淆矩阵图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.labels import EMOTION_CLASSES, cn
from src.dataset import load_datasets
from src.plot_utils import save_confusion_matrix
from src.recognizer import ExpressionRecognizer


def main():
    ap = argparse.ArgumentParser(description="测试集评估")
    ap.add_argument("--weights", default="weights/best_model.pth")
    ap.add_argument("--data", default="data/fer2013")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    recognizer = ExpressionRecognizer(args.weights)
    device = recognizer.device
    print(f"模型: {recognizer.model_type}  权重: {args.weights}  设备: {device}")

    _, _, test_ds = load_datasets(args.data)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred = [], []
    for images, targets in tqdm(loader, desc="评估", ncols=90):
        preds = recognizer.model(images.to(device)).argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(targets.numpy().tolist())

    names = [f"{en}({cn(en)})" for en in EMOTION_CLASSES]
    labels7 = list(range(len(names)))
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    report = classification_report(
        y_true, y_pred, labels=labels7, target_names=names, digits=4, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels7)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    print(f"\n总体准确率: {acc:.4f}\n\n{report}")
    (out_dir / "evaluation_report.txt").write_text(
        f"权重: {args.weights}\n测试样本: {len(y_true)}\n总体准确率: {acc:.4f}\n\n{report}",
        encoding="utf-8",
    )
    save_confusion_matrix(
        cm, names, out_dir / "confusion_matrix.png", title=f"测试集混淆矩阵（准确率 {acc:.1%}）"
    )
    print(f"混淆矩阵: {out_dir / 'confusion_matrix.png'}")
    print(f"评估报告: {out_dir / 'evaluation_report.txt'}")


if __name__ == "__main__":
    main()
