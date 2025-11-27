"""
augment_dataset.py
==================

用途：
    對從 extract_landmarks.py 產生的 landmark CSV 進行資料增強（Data Augmentation）。
    包含平移 / 旋轉 / 縮放 / 加噪音，使模型更能適應不同場景與姿勢變化。

使用方式（請在 VSCode 的終端機，且位置在專案根目錄）：

    # 先跑 landmark 抽取（一定要先有 all_landmarks.csv）
    python scripts/extract_landmarks.py

    # 再跑資料增強，每筆原始資料生成 3 筆新資料
    python scripts/augment_dataset.py --num-aug 3

執行後會產生：
    data_augmented/all_landmarks_augmented.csv

欄位說明：
    is_augmented = 0 -> 原始資料
    is_augmented = 1 -> 增強資料
    aug_id = 第幾次增強（0,1,2,...）

==============================================
    📁 輸入： data_landmarks/all_landmarks.csv
    📁 輸出： data_augmented/all_landmarks_augmented.csv
==============================================
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# 專案根目錄（scripts/ 的上一層）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_LANDMARKS_DIR = PROJECT_ROOT / "data_landmarks"
DATA_AUGMENTED_DIR = PROJECT_ROOT / "data_augmented"


def parse_args():
    parser = argparse.ArgumentParser(description="Pose landmarks 資料增強")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_LANDMARKS_DIR / "all_landmarks.csv"),
        help="輸入 CSV 路徑（預設：data_landmarks/all_landmarks.csv）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_AUGMENTED_DIR / "all_landmarks_augmented.csv"),
        help="輸出 CSV 路徑（預設：data_augmented/all_landmarks_augmented.csv）",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=3,
        help="每一筆原始資料要產生幾筆增強樣本（預設 3）",
    )
    return parser.parse_args()


def get_landmark_indices(columns):
    """
    從欄位名稱抓出所有 x_i, y_i, z_i 的欄位清單。
    回傳：
        x_cols, y_cols, z_cols
    """
    x_cols = [c for c in columns if c.startswith("x_")]
    y_cols = [c for c in columns if c.startswith("y_")]
    z_cols = [c for c in columns if c.startswith("z_")]
    return x_cols, y_cols, z_cols


def augment_single_sample(row: pd.Series, x_cols, y_cols, z_cols, num_aug: int):
    """
    對單一 row 做 num_aug 次資料增強。
    回傳：list[pd.Series]

    增強動作包含：
        - 以中心點為原點
        - 旋轉（-5° ~ 5°）
        - 縮放（0.9 ~ 1.1）
        - 平移（-0.02 ~ 0.02）
        - 高斯噪音（0.005）
    """
    augmented_rows = []

    xs = row[x_cols].values.astype(np.float32)
    ys = row[y_cols].values.astype(np.float32)

    # 計算中心點（所有 landmark 平均）
    center_x = xs.mean()
    center_y = ys.mean()

    xs_centered = xs - center_x
    ys_centered = ys - center_y

    for k in range(num_aug):

        # 隨機縮放
        scale = np.random.uniform(0.9, 1.1)

        # 隨機旋轉（角度）
        angle_deg = np.random.uniform(-5, 5)
        angle_rad = angle_deg * math.pi / 180.0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 隨機平移
        shift_x = np.random.uniform(-0.02, 0.02)
        shift_y = np.random.uniform(-0.02, 0.02)

        # 高斯噪音
        noise_x = np.random.normal(loc=0.0, scale=0.005, size=xs.shape)
        noise_y = np.random.normal(loc=0.0, scale=0.005, size=ys.shape)

        # 旋轉 + 縮放
        x_rot = xs_centered * cos_a - ys_centered * sin_a
        y_rot = xs_centered * sin_a + ys_centered * cos_a

        x_aug = x_rot * scale + center_x + shift_x + noise_x
        y_aug = y_rot * scale + center_y + shift_y + noise_y

        new_row = row.copy()
        new_row[x_cols] = x_aug
        new_row[y_cols] = y_aug

        # z 加少量噪音
        if len(z_cols) > 0:
            zs = row[z_cols].values.astype(np.float32)
            new_row[z_cols] = zs + np.random.normal(
                loc=0.0, scale=0.005, size=zs.shape
            )

        new_row["is_augmented"] = 1
        new_row["aug_id"] = k

        augmented_rows.append(new_row)

    return augmented_rows


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] 讀取：{input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"找不到輸入檔案：{input_path}")

    DATA_AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"[INFO] 原始資料筆數：{len(df)}")

    x_cols, y_cols, z_cols = get_landmark_indices(df.columns)

    df_original = df.copy()
    df_original["is_augmented"] = 0
    df_original["aug_id"] = -1

    augmented_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="資料增強中"):
        augmented_rows.extend(
            augment_single_sample(row, x_cols, y_cols, z_cols, num_aug=args.num_aug)
        )

    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows)
        df_all = pd.concat([df_original, df_aug], ignore_index=True)
    else:
        df_all = df_original

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] 已輸出增強後資料：{output_path}")
    print(f"[INFO] 增強後總筆數（含原始）：{len(df_all)}")


if __name__ == "__main__":
    main()
