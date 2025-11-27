"""
feature_utils.py
================

共用的特徵欄位工具：
- 根據欄位名稱自動抓出 x_*, y_*, z_*, v_* 當作特徵
"""

from typing import List
import pandas as pd


def get_feature_columns(df: pd.DataFrame, extra_exclude=None) -> List[str]:
    """
    從 DataFrame 中自動挑出特徵欄位（x_*, y_*, z_*, v_*），
    排除一些 meta 欄位，以及 caller 額外指定的欄位。

    回傳：list[str]
    """
    if extra_exclude is None:
        extra_exclude = []

    exclude_cols = {
        "video_path",
        "frame_idx",
        "category",
        "sub_label",
        "label",
        "is_augmented",
        "aug_id",
    }
    exclude_cols.update(extra_exclude)

    feature_cols = [
        c
        for c in df.columns
        if (
            (c.startswith("x_") or c.startswith("y_") or c.startswith("z_") or c.startswith("v_"))
            and c not in exclude_cols
        )
    ]

    if not feature_cols:
        raise ValueError("找不到任何特徵欄位 (x_*, y_*, z_*, v_*)，請確認資料格式。")

    print(f"[INFO] 使用的特徵欄位數量：{len(feature_cols)}")
    return feature_cols
