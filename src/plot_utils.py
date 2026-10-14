"""绘图工具：统一中文字体，保存训练曲线与混淆矩阵（Agg 后端，无窗口）。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_cjk_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def save_training_curves(history: dict, out_path: str | Path):
    set_cjk_font()
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, history["train_loss"], label="训练损失")
    axes[0].plot(epochs, history["val_loss"], label="验证损失")
    axes[0].set_xlabel("轮次")
    axes[0].set_ylabel("损失")
    axes[0].set_title("损失曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="训练准确率")
    axes[1].plot(epochs, history["val_acc"], label="验证准确率")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("准确率曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    out_path: str | Path,
    title: str = "混淆矩阵",
    normalize: bool = True,
):
    """混淆矩阵图：单元格显示样本数，括号内为按真实类归一化的比例。"""
    set_cjk_font()
    n = len(class_names)
    cm = np.asarray(cm)
    ratio = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    im = ax.imshow(ratio, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n), class_names, rotation=30, ha="right")
    ax.set_yticks(range(n), class_names)
    for i in range(n):
        for j in range(n):
            text = f"{cm[i, j]}\n({ratio[i, j]:.0%})" if normalize else f"{cm[i, j]}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=8,
                color="white" if ratio[i, j] > 0.5 else "black",
            )
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
