# 人脸表情识别软件的设计与实现（课程设计）

基于 PyTorch + FER2013 的人脸表情识别系统，识别 7 种典型表情：
**生气 / 厌恶 / 恐惧 / 开心 / 难过 / 惊讶 / 中性**

## 功能

- **图片识别**：检测人脸并标注表情与置信度，侧栏显示各类别概率条
- **摄像头实时识别**：逐帧检测 + 识别，EMA 概率平滑与标签去抖消除表情闪变（[src/smoothing.py](src/smoothing.py)）
- **批量识别**：整目录处理，输出标注图与 `summary.csv` 汇总
- **模型对比**：自建小型 CNN vs ResNet18 迁移学习；另有 HOG+SVM 传统方法基线
- **实验分析**：准确率、分类报告（精确率/召回率/F1）、混淆矩阵、训练曲线

## 环境搭建（uv）

```bash
uv sync          # 自动创建 .venv（Python 3.12）并安装全部依赖
```

依赖含 CUDA 12.6 版 PyTorch（适配 RTX 显卡）。若需 CPU 版，把
`pyproject.toml` 中 `[[tool.uv.index]]` 的 url 改为 `https://download.pytorch.org/whl/cpu` 后重新 `uv sync`。

安装完成后可自检环境（依赖导入 / 模型前向 / 检测器 / 数据管线 / GUI 构建）：

```bash
uv run python scripts/smoke_test.py
```

## 数据准备

下载 FER2013 数据集，CSV 版或图片版二选一，放到 `data/fer2013/`，详见 [data/README.md](data/README.md)。

## 训练与评估（均在项目根目录执行）

```bash
uv run python scripts/train.py --model cnn --epochs 50        # 自建 CNN（CPU 也能训）
uv run python scripts/train.py --model resnet18 --epochs 30   # 迁移学习，首次需联网下载预训练权重
uv run python scripts/evaluate.py                             # 测试集评估 + 混淆矩阵
uv run python scripts/baseline_svm.py                         # HOG+SVM 传统方法对比
uv run python scripts/predict_image.py path/to/photo.jpg      # 命令行单图预测
```

产出：`weights/best_model.pth`（GUI 使用的权重）、`results/` 下的训练曲线、混淆矩阵、评估报告。

## 启动软件

```bash
uv run python main.py
```

三个页签：图片识别 / 摄像头实时 / 批量识别。模型未训练时界面可打开但无法识别。

## 目录结构

```
├── main.py               # 软件入口（GUI）
├── models/               # 模型定义（cnn.py / resnet.py / labels.py）
├── src/
│   ├── detector.py       # Haar 人脸检测与裁剪
│   ├── recognizer.py     # 推理封装（加载权重 → 标签 + 置信度）
│   ├── dataset.py        # FER2013 数据加载（CSV / 图片两种格式）
│   ├── visualize.py      # 结果绘制、帧转 QPixmap
│   ├── plot_utils.py     # 训练曲线 / 混淆矩阵绘图（中文字体）
│   └── gui/main_window.py  # PyQt5 界面
├── scripts/
│   ├── train.py          # 训练入口
│   ├── evaluate.py       # 测试集评估
│   ├── predict_image.py  # 命令行预测
│   └── baseline_svm.py   # HOG+SVM 对比基线
├── data/                 # 数据集（见 data/README.md）
├── weights/              # 训练好的 .pth 权重
└── results/              # 曲线、混淆矩阵、批量识别结果
```

## 预期指标（FER2013 测试集）

| 方法 | 准确率 |
|------|--------|
| HOG + SVM（基线） | 约 45~50% |
| 自建 CNN | 约 60~65% |
| ResNet18 迁移学习 | 约 68~72% |

## 常见问题

- **训练提示 device=cpu**：说明 CUDA 不可用，检查是否按上文安装了 cu126 版 PyTorch。
- **resnet18 首次训练卡在下载**：正在从 download.pytorch.org 下载 ImageNet 预训练权重（约 45MB），需网络。
- **摄像头打不开**：在"摄像头实时"页切换摄像头编号；笔记本通常为 0。
- **厌恶类召回率低**：数据集本身该类样本极少（约 600 张），属正常现象，可作为报告分析点。
