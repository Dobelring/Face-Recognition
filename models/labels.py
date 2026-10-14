"""表情类别定义。

索引顺序与 FER2013 官方标签一致，训练/推理/评估均依赖此顺序，勿改动。
"""

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_CN = {
    "angry": "生气",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "开心",
    "sad": "难过",
    "surprise": "惊讶",
    "neutral": "中性",
}


def cn(name: str) -> str:
    return EMOTION_CN.get(name, name)
