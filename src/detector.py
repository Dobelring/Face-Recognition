"""人脸检测：OpenCV Haar 级联（随 opencv-python 自带，无需额外下载模型文件）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CASCADE_PATH = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(
        self,
        scale_factor: float = 1.15,
        min_neighbors: int = 6,
        min_size: tuple[int, int] = (60, 60),
    ):
        self.cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        if self.cascade.empty():
            raise RuntimeError(f"Haar 级联加载失败: {CASCADE_PATH}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        """返回人脸框列表 [(x, y, w, h), ...]，按面积从大到小排序。"""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def crop_face(
        self,
        image_bgr: np.ndarray,
        box: tuple[int, int, int, int],
        margin: float = 0.12,
        size: tuple[int, int] = (48, 48),
    ) -> np.ndarray:
        """按框裁剪人脸（四周外扩 margin 比例保留背景），缩放到 size，返回单通道灰度图。"""
        h_img, w_img = image_bgr.shape[:2]
        x, y, w, h = box
        dx, dy = int(w * margin), int(h * margin)
        x1, y1 = max(0, x - dx), max(0, y - dy)
        x2, y2 = min(w_img, x + w + dx), min(h_img, y + h + dy)
        crop = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
