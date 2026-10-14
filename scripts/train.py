"""训练入口。

用法（在项目根目录）:
  uv run python scripts/train.py --model cnn --epochs 50
  uv run python scripts/train.py --model resnet18 --epochs 30

产出:
  weights/best_model.pth          验证准确率最高的权重（GUI 与评估脚本使用）
  results/training_curves_<model>.png  训练/验证损失与准确率曲线
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MODEL_TYPES, build_model
from models.labels import EMOTION_CLASSES
from src.dataset import load_datasets
from src.plot_utils import save_training_curves


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, desc=""):
    """optimizer 为 None 时执行评估（不更新参数）。返回 (平均损失, 准确率)。"""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False, ncols=90)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, targets)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
        pbar.set_postfix(loss=f"{total_loss / max(total, 1):.4f}", acc=f"{correct / max(total, 1):.4f}")
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="FER2013 表情识别训练")
    ap.add_argument("--data", default="data/fer2013", help="数据目录（含 CSV 或 train/test 图片）")
    ap.add_argument("--model", choices=MODEL_TYPES, default="cnn")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=0, help="Windows 下建议 0")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[1/4] 加载数据: {args.data}")
    train_ds, val_ds, test_ds = load_datasets(args.data, seed=args.seed)
    print(f"      训练 {len(train_ds)} / 验证 {len(val_ds)} / 测试 {len(test_ds)}")
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    print(f"[2/4] 构建模型: {args.model} (device={device})")
    model = build_model(args.model, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"[3/4] 开始训练，共 {args.epochs} 轮")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            desc=f"epoch {epoch}/{args.epochs} train",
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, device, desc=f"epoch {epoch}/{args.epochs} val  "
        )
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        star = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_type": args.model,
                    "classes": list(EMOTION_CLASSES),
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": va_acc,
                },
                out_dir / "best_model.pth",
            )
            star = "  <- 保存最优"
        print(
            f"epoch {epoch:>3}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.0f}s{star}"
        )

    print("[4/4] 保存结果并做测试集快评")
    save_training_curves(history, results_dir / f"training_curves_{args.model}.png")
    print(f"训练曲线: {results_dir / f'training_curves_{args.model}.png'}")

    ckpt = torch.load(out_dir / "best_model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    te_loss, te_acc = run_epoch(model, test_loader, criterion, device, desc="test")
    print(f"测试集: loss {te_loss:.4f} acc {te_acc:.4f}（混淆矩阵等详细分析请运行 scripts/evaluate.py）")
    print(f"最优权重: {out_dir / 'best_model.pth'}（验证准确率 {best_val_acc:.4f}）")


if __name__ == "__main__":
    main()
