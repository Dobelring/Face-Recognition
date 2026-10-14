# 数据准备

本项目的数据目录约定如下。FER2013 数据集请从 Kaggle 下载（需注册账号），**两种格式二选一**，放到 `data/fer2013/` 下即可，代码会自动识别。

## 方式一：CSV 版（官方原始格式）

下载地址（Kaggle 比赛 "Challenges in Representation Learning: Facial Expression Recognition" 的 Data 页）:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

下载 `fer2013.tar.gz` 解压后得到 `fer2013.csv`，放到:

```
data/fer2013/fer2013.csv
```

要求 CSV 包含 `emotion, pixels, Usage` 三列（`icml_face_data.csv` 同样支持，列顺序不限）。

## 方式二：图片文件夹版

下载地址（Kaggle 数据集 "FER2013"，按 train/test 组织的图片版）:
https://www.kaggle.com/datasets/msambare/fer2013

解压后把 `train` 和 `test` 两个文件夹放到:

```
data/fer2013/train/<类别子目录>/xxx.jpg   （约 28709 张）
data/fer2013/test/<类别子目录>/xxx.jpg    （约 7178 张）
```

类别子目录名可以是英文（`angry / disgust / fear / happy / sad / surprise / neutral`）或数字（`0` ~ `6`）。

## 数据规模

FER2013 共 35887 张 48x48 灰度人脸图，7 类表情:
生气(0) 厌恶(1) 恐惧(2) 开心(3) 难过(4) 惊讶(5) 中性(6)
其中厌恶类样本最少（约 600 张），训练时表现为该类召回率偏低，可作为报告中的分析点。
