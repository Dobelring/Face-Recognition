"""表情识别推理封装：加载训练好的权重，输出表情标签、置信度与各类概率。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from models import build_model
from models.labels import EMOTION_CLASSES, cn
from src.dataset import EVAL_TRANSFORM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "best_model.pth"


class ExpressionRecognizer:
    def __init__(self, weights_path: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model_type: str = ckpt["model_type"]
        self.classes: list[str] = list(ckpt.get("classes", EMOTION_CLASSES))
        self.model = build_model(self.model_type, num_classes=len(self.classes), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

    @property
    def label(self) -> str:
        return f"{self.model_type}（{self.device.type}）"

    @torch.no_grad()
    def predict_gray(self, gray48: np.ndarray) -> tuple[int, str, float, np.ndarray]:
        """输入 48x48 单通道 uint8 灰度人脸图。

        返回 (类别索引, 英文标签, 置信度, 各类概率数组)。
        """
        tensor = EVAL_TRANSFORM(Image.fromarray(gray48, mode="L")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return idx, self.classes[idx], float(probs[idx]), probs

    @staticmethod
    def cn_name(name_en: str) -> str:
        return cn(name_en)
