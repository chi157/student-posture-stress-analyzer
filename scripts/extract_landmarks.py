"""
extract_landmarks.py

從 data_raw/ 底下的影片 (mp4) 讀取，使用 MediaPipe Pose 擷取關鍵點，
輸出成 CSV 檔到 data_landmarks/ 對應結構，並產生一份 all_landmarks.csv 總表。

資料夾假設結構：
data_raw/
  posture/good_posture/*.mp4
  posture/bad_posture/*.mp4
  ...
  stress/head_touch/*.mp4
  ...
  baseline/focus_good/*.mp4
  ...

輸出：
data_landmarks/
  posture/*.csv
  stress/*.csv
  baseline/*.csv
  all_landmarks.csv
"""

import os
import sys
from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm


# 專案根目錄（scripts/ 的上一層）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_LANDMARKS_DIR = PROJECT_ROOT / "data_landmarks"


def discover_videos(root: Path):
    """
    遍歷 data_raw，找到所有 mp4 檔案。

    回傳 list[Path]
    """
    videos = list(root.rglob("*.mp4"))
    return videos


def get_category_and_label(video_path: Path):
    """
    根據檔案路徑推出 category / sub_label / full_label

    例如：
    data_raw/posture/good_posture/posture_good_01.mp4
      -> category = posture
      -> sub_label = good_posture
      -> label = posture_good_posture
    """
    # .../data_raw/<category>/<sub_label>/<file>
    try:
        sub_label = video_path.parent.name
        category = video_path.parent.parent.name  # posture / stress / baseline
    except Exception:
        category = "unknown"
        sub_label = "unknown"

    label = f"{category}_{sub_label}"
    return category, sub_label, label


def extract_pose_landmarks_from_video(video_path: Path):
    """
    讀取單一影片，使用 MediaPipe Pose 抽取每一幀的關鍵點。

    回傳：
        rows: list[dict]  每一個 dict 代表一幀
    """
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] 無法開啟影片: {video_path}")
        return []

    rows = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # OpenCV BGR -> RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                # 若沒偵測到人，這一幀就略過
                continue

            row = {
                "video_path": str(video_path.relative_to(PROJECT_ROOT)),
                "frame_idx": frame_idx,
            }

            # 取出 33 個 pose landmark
            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f"x_{i}"] = lm.x
                row[f"y_{i}"] = lm.y
                row[f"z_{i}"] = lm.z
                row[f"v_{i}"] = lm.visibility

            rows.append(row)

    cap.release()
    return rows


def main():
    print(f"[INFO] 專案根目錄: {PROJECT_ROOT}")
    print(f"[INFO] 掃描影片路徑: {DATA_RAW_DIR}")

    if not DATA_RAW_DIR.exists():
        print(f"[ERROR] data_raw 資料夾不存在: {DATA_RAW_DIR}")
        sys.exit(1)

    DATA_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)

    video_paths = discover_videos(DATA_RAW_DIR)
    if not video_paths:
        print("[WARN] data_raw 底下沒有找到任何 .mp4 影片")
        sys.exit(0)

    print(f"[INFO] 共找到 {len(video_paths)} 支影片")

    all_dfs = []

    for vp in tqdm(video_paths, desc="處理影片"):
        category, sub_label, label = get_category_and_label(vp)

        print(f"\n[INFO] 處理影片: {vp}")
        print(f"       category={category}, sub_label={sub_label}, label={label}")

        rows = extract_pose_landmarks_from_video(vp)
        if not rows:
            print(f"[WARN] 影片無有效關鍵點資料: {vp}")
            continue

        # 加上標籤欄位
        for r in rows:
            r["category"] = category
            r["sub_label"] = sub_label
            r["label"] = label

        df = pd.DataFrame(rows)

        # 個別影片輸出到 data_landmarks/<category>/<basename>.csv
        out_subdir = DATA_LANDMARKS_DIR / category
        out_subdir.mkdir(parents=True, exist_ok=True)

        out_csv = out_subdir / f"{vp.stem}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] 已輸出: {out_csv}")

        all_dfs.append(df)

    # 整合成一份 all_landmarks.csv
    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_csv = DATA_LANDMARKS_DIR / "all_landmarks.csv"
        all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
        print(f"\n[INFO] 已輸出總表: {all_csv}")
        print(f"[INFO] 總筆數: {len(all_df)}")
    else:
        print("[WARN] 沒有任何 landmarks 資料被產生。")


if __name__ == "__main__":
    main()
