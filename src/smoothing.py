"""实时识别的时序平滑。

摄像头逐帧独立预测时，单帧的小抖动会直接显示为表情"闪变"。
这里用两层手段稳定输出：
1. ProbSmoother: 对单个目标的 7 类概率做指数滑动平均（EMA），
   并要求新标签连续若干帧稳定后才切换显示（去抖）。
2. FaceTracker: 按 IoU 把相邻帧的人脸框匹配成同一目标，
   让每个目标持有独立的平滑历史（多人场景互不干扰）。
"""

from __future__ import annotations

import numpy as np


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """两个 (x, y, w, h) 框的交并比。"""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class ProbSmoother:
    """概率序列 EMA 平滑 + 标签切换去抖。

    alpha: 新帧权重，越小越平滑（0.2~0.4 常用）。
    switch_patience: 新标签需在平滑概率中连续领先这么多帧才切换显示。
    """

    def __init__(self, alpha: float = 0.35, switch_patience: int = 3):
        self.alpha = alpha
        self.switch_patience = max(1, switch_patience)
        self.ema: np.ndarray | None = None
        self.stable_label: int | None = None
        self._candidate: int | None = None
        self._candidate_count = 0

    def update(self, probs: np.ndarray) -> tuple[int, np.ndarray]:
        """输入本帧概率，返回 (稳定标签索引, 平滑后概率)。"""
        probs = np.asarray(probs, dtype=float)
        if self.ema is None:
            self.ema = probs.copy()
        else:
            self.ema = self.alpha * probs + (1.0 - self.alpha) * self.ema
        top = int(self.ema.argmax())
        if top == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = top
            self._candidate_count = 1
        # 首个目标出现时立即显示；之后只有连续稳定才切换，避免闪变
        if self.stable_label is None or self._candidate_count >= self.switch_patience:
            self.stable_label = top
        return self.stable_label, self.ema


class FaceTracker:
    """跨帧匹配人脸框（IoU 贪心），每个目标持有独立 ProbSmoother。"""

    def __init__(
        self,
        max_age: int = 6,
        iou_thresh: float = 0.2,
        smoother_alpha: float = 0.35,
        switch_patience: int = 3,
    ):
        self.max_age = max_age  # 连续多少帧没匹配上才删除目标
        self.iou_thresh = iou_thresh
        self.smoother_kwargs = {"alpha": smoother_alpha, "switch_patience": switch_patience}
        self._tracks: dict[int, dict] = {}  # track_id -> {"box", "smoother", "age"}
        self._next_id = 0

    def reset(self):
        self._tracks.clear()

    def update(self, boxes: list, probs_list: list) -> list:
        """输入本帧的框列表与对应概率，返回目标列表:

        [(box, 稳定标签索引, 平滑概率, 稳定标签的平滑概率), ...]
        """
        matched: dict[int, int] = {}  # track_id -> 检测下标
        used: set[int] = set()
        pairs = []
        for tid, tr in self._tracks.items():
            for di, box in enumerate(boxes):
                v = iou(tr["box"], box)
                if v >= self.iou_thresh:
                    pairs.append((v, tid, di))
        for v, tid, di in sorted(pairs, key=lambda p: p[0], reverse=True):
            if tid in matched or di in used:
                continue
            matched[tid] = di
            used.add(di)

        out = []
        for tid, tr in self._tracks.items():
            if tid in matched:
                di = matched[tid]
                tr["box"] = boxes[di]
                tr["age"] = 0
                label, ema = tr["smoother"].update(probs_list[di])
                out.append((boxes[di], label, ema, float(ema[label])))
            else:
                tr["age"] += 1

        for di, box in enumerate(boxes):
            if di not in used:  # 新目标
                smoother = ProbSmoother(**self.smoother_kwargs)
                label, ema = smoother.update(probs_list[di])
                self._tracks[self._next_id] = {"box": box, "smoother": smoother, "age": 0}
                out.append((box, label, ema, float(ema[label])))
                self._next_id += 1

        self._tracks = {tid: tr for tid, tr in self._tracks.items() if tr["age"] <= self.max_age}
        return out
