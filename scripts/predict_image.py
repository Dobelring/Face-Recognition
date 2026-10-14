"""对单张图片或整个文件夹做人脸检测 + 表情识别，保存标注结果图。

用法（在项目根目录）:
  uv run python scripts/predict_image.py path/to/photo.jpg
  uv run python scripts/predict_image.py path/to/photos/ --out results/predictions
  uv run python scripts/predict_image.py photo.jpg --show   # 额外弹窗预览
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import FaceDetector
from src.recognizer import ExpressionRecognizer
from src.visualize import draw_face_results

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="单图/文件夹表情识别")
    ap.add_argument("inputs", nargs="+", help="图片文件或目录")
    ap.add_argument("--weights", default="weights/best_model.pth")
    ap.add_argument("--out", default="results/predictions")
    ap.add_argument("--show", action="store_true", help="弹窗显示结果（按任意键关闭）")
    args = ap.parse_args()

    detector = FaceDetector()
    recognizer = ExpressionRecognizer(args.weights)

    files: list[Path] = []
    for p in map(Path, args.inputs):
        if p.is_dir():
            files.extend(q for q in sorted(p.iterdir()) if q.suffix.lower() in IMAGE_EXTS)
        else:
            files.append(p)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[跳过] 无法读取 {f}")
            continue
        results = []
        for box in detector.detect(img):
            gray48 = detector.crop_face(img, box)
            _, label, conf, _ = recognizer.predict_gray(gray48)
            results.append((box, label, conf))
        vis = draw_face_results(img, results)
        dst = out_dir / f"{f.stem}_pred{f.suffix}"
        cv2.imwrite(str(dst), vis)
        tags = (
            ", ".join(f"{lb}({recognizer.cn_name(lb)}) {cf:.0%}" for _, lb, cf in results)
            or "未检出人脸"
        )
        print(f"{f.name}: {tags}  ->  {dst}")
        if args.show:
            cv2.imshow("prediction (press any key)", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
