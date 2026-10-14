"""PyQt5 图形界面：图片识别 / 摄像头实时识别 / 批量识别 三个页签。"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch  # 必须先于 PyQt5 导入，否则 Windows 下 c10.dll 初始化失败
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from models.labels import EMOTION_CLASSES, cn
from src.detector import FaceDetector
from src.smoothing import FaceTracker
from src.visualize import EMO_COLORS_HEX, draw_face_results, to_pixmap

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIEW_STYLE = "border: 1px solid #ccc; background: #1e1e1e; color: #999;"


# --------------------------------------------------------------------------- #
# 结果面板：各类别概率条
# --------------------------------------------------------------------------- #
class ResultPanel(QWidget):
    def __init__(self, title: str = "识别结果", parent=None):
        super().__init__(parent)
        self._base_title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.title_label = QLabel(f"<b>{title}</b>（未识别）")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16pt;")
        layout.addWidget(self.title_label)
        self.bars: dict[str, QProgressBar] = {}
        for name in EMOTION_CLASSES:
            row = QHBoxLayout()
            tag = QLabel(f"{cn(name)} {name}")
            tag.setFixedWidth(120)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
            bar.setFixedHeight(18)
            bar.setStyleSheet(
                "QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; }"
                f"QProgressBar::chunk {{ background-color: {EMO_COLORS_HEX[name]}; border-radius: 3px; }}"
            )
            row.addWidget(tag)
            row.addWidget(bar, 1)
            layout.addLayout(row)
            self.bars[name] = bar

    def update_result(self, probs: np.ndarray, classes: list[str], top_label: str, top_conf: float):
        for i, name in enumerate(classes):
            if name in self.bars:
                self.bars[name].setValue(int(round(float(probs[i]) * 100)))
        self.title_label.setText(f"<b>{cn(top_label)} {top_conf:.1%}</b>（{top_label}）")

    def clear(self):
        for bar in self.bars.values():
            bar.setValue(0)
        self.title_label.setText(f"<b>{self._base_title}</b>（未识别）")


# --------------------------------------------------------------------------- #
# 页签 1：单张图片识别
# --------------------------------------------------------------------------- #
class ImageTab(QWidget):
    def __init__(self, detector: FaceDetector, recognizer, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.recognizer = recognizer
        self._image: np.ndarray | None = None
        self._vis: np.ndarray | None = None

        self.image_label = QLabel("请先打开一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(560, 420)
        self.image_label.setStyleSheet(VIEW_STYLE)

        btn_open = QPushButton("打开图片…")
        btn_open.clicked.connect(self.on_open)
        btn_redo = QPushButton("重新识别")
        btn_redo.clicked.connect(self.run)

        self.panel = ResultPanel()
        side = QVBoxLayout()
        side.addWidget(self.panel)
        side.addStretch(1)
        side.addWidget(btn_open)
        side.addWidget(btn_redo)
        side.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(self.image_label)
        root = QHBoxLayout(self)
        root.addLayout(left, 3)
        root.addLayout(side, 1)

    def on_open(self):
        if self.recognizer is None:
            QMessageBox.warning(self, "提示", "模型未加载：请先运行 scripts/train.py 生成 weights/best_model.pth")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "错误", f"无法读取图片:\n{path}")
            return
        self._image = img
        self.run()

    def run(self):
        if self._image is None or self.recognizer is None:
            return
        img = self._image
        results = []
        for box in self.detector.detect(img):
            gray48 = self.detector.crop_face(img, box)
            _, label, conf, probs = self.recognizer.predict_gray(gray48)
            results.append((box, label, conf, probs))
        self._vis = draw_face_results(img, [r[:3] for r in results])
        self._show(self._vis)
        if results:
            best = max(results, key=lambda r: r[2])
            self.panel.update_result(best[3], self.recognizer.classes, best[1], best[2])
        else:
            self.panel.clear()
            QMessageBox.information(self, "提示", "未检出人脸，请换一张更清晰的正面照片。")

    def _show(self, bgr: np.ndarray):
        pm = to_pixmap(bgr).scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pm)

    def resizeEvent(self, event):  # 窗口缩放时按当前尺寸重绘
        super().resizeEvent(event)
        if self._vis is not None:
            self._show(self._vis)


# --------------------------------------------------------------------------- #
# 页签 2：摄像头实时识别
# --------------------------------------------------------------------------- #
class CameraTab(QWidget):
    def __init__(self, detector: FaceDetector, recognizer, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.recognizer = recognizer
        self.cap: cv2.VideoCapture | None = None
        self._t_prev: float | None = None
        self._fps = 0.0
        # 时序平滑：alpha 越小画面表情越稳（延迟越高），switch_patience 越大标签越不易闪变
        self.tracker = FaceTracker(smoother_alpha=0.35, switch_patience=3)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_frame)

        self.view = QLabel("摄像头未开启")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(640, 480)
        self.view.setStyleSheet(VIEW_STYLE)

        self.combo = QComboBox()
        self.combo.addItems(["摄像头 0", "摄像头 1", "摄像头 2"])
        self.btn = QPushButton("开启摄像头")
        self.btn.clicked.connect(self.toggle)
        self.fps_label = QLabel("FPS: -")

        controls = QHBoxLayout()
        controls.addWidget(self.combo)
        controls.addWidget(self.btn)
        controls.addStretch(1)
        controls.addWidget(self.fps_label)

        self.panel = ResultPanel("实时识别")

        body = QHBoxLayout()
        body.addWidget(self.view, 3)
        side = QVBoxLayout()
        side.addWidget(self.panel)
        side.addStretch(1)
        body.addLayout(side, 1)
        root = QVBoxLayout(self)
        root.addLayout(controls)
        root.addLayout(body, 1)

    def toggle(self):
        if self.cap is None:
            if self.recognizer is None:
                QMessageBox.warning(self, "提示", "模型未加载：请先运行 scripts/train.py 生成 weights/best_model.pth")
                return
            idx = self.combo.currentIndex()
            api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, api)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                cap.release()
                QMessageBox.warning(self, "错误", f"无法打开摄像头 {idx}")
                return
            self.cap = cap
            self._t_prev = None
            self.tracker.reset()
            self.combo.setEnabled(False)
            self.btn.setText("关闭摄像头")
            self.timer.start(33)
        else:
            self.stop()

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.combo.setEnabled(True)
        self.btn.setText("开启摄像头")
        self.view.clear()
        self.view.setText("摄像头未开启")
        self.fps_label.setText("FPS: -")

    def on_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        results = []
        try:
            dets = []
            for box in self.detector.detect(frame):
                gray48 = self.detector.crop_face(frame, box)
                _, _, _, probs = self.recognizer.predict_gray(gray48)
                dets.append((box, probs))
            # 跨帧匹配 + EMA 平滑 + 标签去抖，消除表情闪变
            results = self.tracker.update(
                [b for b, _ in dets], [p for _, p in dets]
            )
        except Exception as exc:  # 单帧失败不中断视频
            self.fps_label.setText(f"推理异常: {exc}")
            return

        now = time.time()
        if self._t_prev is not None:
            dt = now - self._t_prev
            if dt > 0:
                fps_inst = 1.0 / dt
                self._fps = fps_inst if self._fps == 0 else 0.9 * self._fps + 0.1 * fps_inst
        self._t_prev = now
        self.fps_label.setText(f"FPS: {self._fps:.0f}   人脸: {len(results)}")

        # (box, 稳定标签索引, 平滑概率, 平滑后稳定标签概率)
        draw_items = [
            (box, self.recognizer.classes[label_idx], conf)
            for box, label_idx, _, conf in results
        ]
        vis = draw_face_results(frame, draw_items)
        self.view.setPixmap(
            to_pixmap(vis).scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        if results:
            best = max(results, key=lambda r: r[3])
            top_label = self.recognizer.classes[best[1]]
            self.panel.update_result(best[2], self.recognizer.classes, top_label, best[3])
        else:
            self.panel.clear()


# --------------------------------------------------------------------------- #
# 页签 3：批量识别
# --------------------------------------------------------------------------- #
class BatchTab(QWidget):
    def __init__(self, detector: FaceDetector, recognizer, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.recognizer = recognizer

        self.in_edit = QLineEdit()
        self.in_edit.setReadOnly(True)
        self.in_edit.setPlaceholderText("选择包含图片的文件夹")
        btn_in = QPushButton("选择文件夹…")
        btn_in.clicked.connect(self.pick_dir)
        self.btn_run = QPushButton("开始批量识别")
        self.btn_run.clicked.connect(self.run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)

        form = QHBoxLayout()
        form.addWidget(QLabel("图片目录:"))
        form.addWidget(self.in_edit, 1)
        form.addWidget(btn_in)
        form.addWidget(self.btn_run)
        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self.progress)
        root.addWidget(self.log, 1)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if d:
            self.in_edit.setText(d)

    def run(self):
        if self.recognizer is None:
            QMessageBox.warning(self, "提示", "模型未加载：请先运行 scripts/train.py 生成 weights/best_model.pth")
            return
        src = Path(self.in_edit.text())
        if not src.is_dir():
            QMessageBox.warning(self, "提示", "请先选择图片文件夹")
            return
        files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            QMessageBox.warning(self, "提示", "该文件夹中没有图片")
            return

        out_dir = Path("results") / "batch"
        out_dir.mkdir(parents=True, exist_ok=True)
        self.progress.setRange(0, len(files))
        self.btn_run.setEnabled(False)
        self.log.clear()
        rows = [("文件", "人脸数", "表情(中文)", "表情(英文)", "置信度")]
        try:
            for i, f in enumerate(files, 1):
                img = cv2.imread(str(f))
                if img is None:
                    self.log.appendPlainText(f"[跳过] 无法读取 {f.name}")
                    self.progress.setValue(i)
                    continue
                results = []
                for box in self.detector.detect(img):
                    gray48 = self.detector.crop_face(img, box)
                    _, label, conf, _ = self.recognizer.predict_gray(gray48)
                    results.append((box, label, conf))
                vis = draw_face_results(img, results)
                cv2.imwrite(str(out_dir / f"{f.stem}_pred{f.suffix}"), vis)
                if results:
                    _, label, conf = max(results, key=lambda r: r[2])
                    rows.append((f.name, len(results), cn(label), label, f"{conf:.2%}"))
                    self.log.appendPlainText(f"{f.name}: {cn(label)} {conf:.0%}（{len(results)} 张人脸）")
                else:
                    rows.append((f.name, 0, "未检出", "-", "-"))
                    self.log.appendPlainText(f"{f.name}: 未检出人脸")
                QApplication.processEvents()
                self.progress.setValue(i)
        finally:
            self.btn_run.setEnabled(True)

        summary = out_dir / "summary.csv"
        with open(summary, "w", newline="", encoding="utf-8-sig") as fh:
            csv.writer(fh).writerows(rows)
        self.log.appendPlainText(f"\n完成：标注图与 summary.csv 已保存到 {out_dir.resolve()}")
        QMessageBox.information(self, "完成", f"已处理 {len(files)} 张图片\n结果目录: {out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# 主窗口与入口
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, recognizer):
        super().__init__()
        self.setWindowTitle("人脸表情识别系统")
        self.resize(1100, 680)
        self.recognizer = recognizer
        self.detector = FaceDetector()

        tabs = QTabWidget()
        self.camera_tab = CameraTab(self.detector, recognizer)
        tabs.addTab(ImageTab(self.detector, recognizer), "图片识别")
        tabs.addTab(self.camera_tab, "摄像头实时")
        tabs.addTab(BatchTab(self.detector, recognizer), "批量识别")
        self.setCentralWidget(tabs)

        status = "模型: " + (recognizer.label if recognizer else "未加载（请先运行 scripts/train.py）")
        self.statusBar().showMessage(status)

    def closeEvent(self, event):
        self.camera_tab.stop()
        super().closeEvent(event)


def run_app(recognizer=None):
    app = QApplication(sys.argv)
    win = MainWindow(recognizer)
    win.show()
    if recognizer is None:
        QMessageBox.warning(
            win,
            "模型未加载",
            "未找到 weights/best_model.pth，界面可浏览但无法识别。\n"
            "请先训练模型:\n    uv run python scripts/train.py",
        )
    sys.exit(app.exec_())
