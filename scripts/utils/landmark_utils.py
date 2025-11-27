"""
landmark_utils.py
=================

從 MediaPipe Pose 的輸出結果，依照訓練時的 feature_columns
組成一個 numpy 特徵向量，給模型做推論。
"""

import numpy as np


def build_feature_vector_from_pose(results, feature_columns):
    """
    給定 MediaPipe Pose 的 results + 模型需要的 feature_columns，
    回傳 shape = (1, num_features) 的 numpy array。

    feature_columns 通常像：
        ["x_0", "y_0", "z_0", "v_0", "x_1", "y_1", ...]
    """
    if not results.pose_landmarks:
        return None

    lm_dict = {}
    for i, lm in enumerate(results.pose_landmarks.landmark):
        lm_dict[f"x_{i}"] = lm.x
        lm_dict[f"y_{i}"] = lm.y
        lm_dict[f"z_{i}"] = lm.z
        lm_dict[f"v_{i}"] = lm.visibility

    feat = []
    for col in feature_columns:
        feat.append(lm_dict.get(col, 0.0))  # 沒找到就用 0.0 補

    feat = np.array(feat, dtype=np.float32).reshape(1, -1)
    return feat
