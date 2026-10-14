"""临时冒烟测试：验证依赖、模型、检测器、数据管线、GUI 均可正常工作。"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

print("[1] 依赖导入")
import cv2
import torchvision
from PyQt5 import QtWidgets

print(f"    torch {torch.__version__}  cuda可用={torch.cuda.is_available()}")
print(f"    opencv {cv2.__version__}  torchvision {torchvision.__version__}")

print("[2] 模型前向")
from models import MODEL_TYPES, build_model

for t in MODEL_TYPES:
    m = build_model(t, pretrained=False)
    out = m(torch.randn(2, 1, 48, 48))
    assert out.shape == (2, 7), f"{t} 输出形状错误: {out.shape}"
    n_params = sum(p.numel() for p in m.parameters())
    print(f"    {t}: 输出 {tuple(out.shape)}, 参数量 {n_params / 1e6:.2f}M")

print("[3] Haar 人脸检测器")
from src.detector import FaceDetector

det = FaceDetector()
print(f"    级联加载成功: {Path(det.cascade.__class__.__module__)}, min_size={det.min_size}")
# 造一张纯噪声图，确认 detect 返回列表（不报错即可）
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
boxes = det.detect(img)
print(f"    噪声图检出人脸框: {len(boxes)}（预期 0）")
assert boxes == []

print("[4] 数据管线（伪造 CSV 端到端）")
import pandas as pd

rng = np.random.default_rng(0)
rows = []
for i in range(60):
    px = " ".join(str(v) for v in rng.integers(0, 256, 2304))
    rows.append(
        {
            "emotion": int(i % 7),
            "pixels": px,
            "Usage": ["Training", "PublicTest", "PrivateTest"][i % 3],
        }
    )
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    pd.DataFrame(rows).to_csv(root / "fer2013.csv", index=False)
    from src.dataset import EVAL_TRANSFORM, load_datasets

    train_ds, val_ds, test_ds = load_datasets(root)
    x, y = train_ds[0]
    assert x.shape == (1, 48, 48) and x.min() >= -1 and x.max() <= 1, f"样本异常: {x.shape}"
    print(f"    train/val/test = {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print(f"    单样本: 形状 {tuple(x.shape)}, 范围 [{x.min():.2f}, {x.max():.2f}], 标签 {y}")
    from torch.utils.data import DataLoader

    dl = DataLoader(train_ds, batch_size=8)
    xb, yb = next(iter(dl))
    assert xb.shape == (8, 1, 48, 48)
    print(f"    DataLoader 批: {tuple(xb.shape)}")

print("[5] 识别器 + 权重保存/加载回路（随机权重，仅验证接口）")
from models.labels import EMOTION_CLASSES

model = build_model("cnn", pretrained=False)
ckpt = {
    "model_type": "cnn",
    "classes": list(EMOTION_CLASSES),
    "state_dict": model.state_dict(),
    "epoch": 0,
    "val_acc": 0.0,
}
tmp_weights = Path(tempfile.gettempdir()) / "_smoke_best_model.pth"
torch.save(ckpt, tmp_weights)
from src.recognizer import ExpressionRecognizer

rec = ExpressionRecognizer(tmp_weights, device="cpu")
gray48 = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
idx, label, conf, probs = rec.predict_gray(gray48)
assert probs.shape == (7,) and abs(probs.sum() - 1.0) < 1e-4
print(f"    predict_gray -> {label}({rec.cn_name(label)}) conf={conf:.3f}, 概率和={probs.sum():.4f}")

print("[6] 绘图工具")
from src.plot_utils import save_confusion_matrix, save_training_curves

tmp_png = Path(tempfile.gettempdir()) / "_smoke_cm.png"
save_confusion_matrix(np.array([[30, 1, 0, 2, 1, 0, 1]] * 7), EMOTION_CLASSES, tmp_png)
hist = {k: [0.5 - i * 0.01 for i in range(10)] for k in ("train_loss", "val_loss")}
hist.update({k: [0.3 + i * 0.03 for i in range(10)] for k in ("train_acc", "val_acc")})
save_training_curves(hist, Path(tempfile.gettempdir()) / "_smoke_curve.png")
print(f"    混淆矩阵/训练曲线保存成功: {tmp_png.exists()}")

print("[7] GUI 构建（离屏模式）")
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QtWidgets.QApplication([])
from src.gui.main_window import MainWindow

win = MainWindow(recognizer=rec)
assert win.centralWidget().count() == 3
print(f"    主窗口构建成功，页签数={win.centralWidget().count()}，状态栏: {win.statusBar().currentMessage()}")
win.camera_tab.stop()
win.close()

print("[8] 时序平滑（EMA + 去抖 + IoU 跟踪）")
from src.smoothing import FaceTracker, ProbSmoother

sm = ProbSmoother(alpha=0.5, switch_patience=3)
onehot = np.eye(7)
seq = [3] * 5 + [0, 0] + [3] * 4 + [0] * 4  # 中间 2 帧干扰不应切换；连续 4 帧应切换
labels = [sm.update(onehot[p] + 1e-6)[0] for p in seq]
assert labels[0] == 3 and all(v == 3 for v in labels[:7]), f"干扰帧未被去抖: {labels}"
assert labels[-1] == 0, f"持续新标签未切换: {labels}"
print(f"    去抖序列 {labels}: 2 帧干扰被抑制，4 帧后正常切换")

tracker = FaceTracker()
r1 = tracker.update([(100, 100, 120, 120)], [onehot[2]])
r2 = tracker.update([(104, 102, 120, 120)], [onehot[2]])  # 微移，应为同一目标
assert len(r1) == 1 and len(r2) == 1 and r2[0][1] == 2
r3 = tracker.update([], [])  # 空帧不崩溃，目标保留
r4 = tracker.update([(100, 100, 120, 120)], [onehot[2]])
assert len(r4) == 1 and r4[0][1] == 2
print("    IoU 跨帧匹配 / 空帧容忍: OK")

tmp_weights.unlink(missing_ok=True)
print("\n全部冒烟测试通过 ✔")
