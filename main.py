"""软件入口：启动 PyQt5 图形界面。

用法（在项目根目录）: uv run python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.recognizer import DEFAULT_WEIGHTS, ExpressionRecognizer  # 先导入 torch
from src.gui.main_window import run_app


def main() -> None:
    recognizer = None
    if DEFAULT_WEIGHTS.exists():
        try:
            recognizer = ExpressionRecognizer(DEFAULT_WEIGHTS)
            print(f"已加载模型: {DEFAULT_WEIGHTS}")
        except Exception as exc:
            print(f"模型加载失败: {exc}")
    else:
        print("未找到权重 weights/best_model.pth，请先运行: uv run python scripts/train.py")
    run_app(recognizer)


if __name__ == "__main__":
    main()
