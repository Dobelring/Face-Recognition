"""可视化工具：在画面上绘制识别结果、BGR 帧转 QPixmap（供 GUI 使用）。

OpenCV 的 putText 不支持中文，画面上统一使用英文标签；中文只出现在 GUI 控件里。
"""

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

EMO_COLORS_BGR = {
    "angry": (0, 0, 230),
    "disgust": (160, 0, 160),
    "fear": (230, 216, 0),
    "happy": (60, 180, 0),
    "sad": (0, 120, 255),
    "surprise": (0, 165, 255),
    "neutral": (180, 180, 180),
}

EMO_COLORS_HEX = {
    "angry": "#e53935",
    "disgust": "#8e24aa",
    "fear": "#00d8e6",
    "happy": "#2db400",
    "sad": "#ff7814",
    "surprise": "#ffa500",
    "neutral": "#9e9e9e",
}


def draw_face_results(frame_bgr: np.ndarray, results: list) -> np.ndarray:
    """results: [(box, label_en, conf), ...]。在副本上绘制人脸框与标签后返回。"""
    out = frame_bgr.copy()
    for box, label, conf in results:
        x, y, w, h = box
        color = EMO_COLORS_BGR.get(label, (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = y - 8 if y - th - 12 > 0 else y + h + th + 8
        cv2.rectangle(out, (x - 1, ty - th - 4), (x + tw + 5, ty + 4), color, -1)
        cv2.putText(out, text, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    """BGR ndarray 转 QPixmap（拷贝数据，避免原数组被释放后 QImage 失效）。"""
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
