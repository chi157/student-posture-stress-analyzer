"""
realtime_demo.py
================

用途：
    使用 webcam + MediaPipe Pose 即時抓取姿態 landmarks，
    並載入已訓練好的：
        - models/posture_model.joblib
        - models/stress_model.joblib
    做「姿勢分類」與「壓力行為分類」，即時顯示在畫面上。

前置條件：
    1. 已經有影片資料並執行過：
        python scripts/extract_landmarks.py
        python scripts/augment_dataset.py --num-aug 3

    2. 已訓練並產生模型：
        python scripts/train_posture_model.py
        python scripts/train_stress_model.py

    3. 專案結構要是：
        project_root/
          ├─ data_raw/
          ├─ data_landmarks/
          ├─ data_augmented/
          ├─ models/
          │    ├─ posture_model.joblib
          │    └─ stress_model.joblib
          └─ scripts/
               └─ realtime_demo.py

使用方式（在專案根目錄執行，例如 VSCode 終端機）：

    python scripts/realtime_demo.py

    # 如果想指定 model 路徑或 webcam index：
    python scripts/realtime_demo.py ^
        --posture-model models/posture_model.joblib ^
        --stress-model models/stress_model.joblib ^
        --camera-index 0

操作：
    - 按下鍵盤 'q' 離開視窗。

需要套件（建議寫進 requirements.txt）：
    - opencv-python
    - mediapipe
    - pandas（只在某些 debug 情境用到，可選）
    - numpy
    - scikit-learn
    - joblib
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import joblib


# 專案根目錄（scripts/ 的上一層）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam 即時姿勢 / 壓力行為偵測 Demo")

    parser.add_argument(
        "--posture-model",
        type=str,
        default=str(MODELS_DIR / "posture_model.joblib"),
        help="姿勢模型路徑（預設：models/posture_model.joblib）",
    )

    parser.add_argument(
        "--stress-model",
        type=str,
        default=str(MODELS_DIR / "stress_model.joblib"),
        help="壓力模型路徑（預設：models/stress_model.joblib）",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index（預設 0）",
    )

    return parser.parse_args()


def load_model_package(path: Path):
    """
    載入 train_*_model.py 存的 joblib 包裝：
        {
            "model": clf,
            "label_encoder": label_encoder,
            "feature_columns": feature_cols,
            "target_col": target_col,
        }
    """
    if not path.exists():
        print(f"[WARN] 找不到模型檔：{path}")
        return None

    pkg = joblib.load(path)
    required_keys = ["model", "label_encoder", "feature_columns"]
    for k in required_keys:
        if k not in pkg:
            raise ValueError(f"模型檔缺少 key: {k}")

    print(f"[INFO] 已載入模型：{path}")
    print(f"       使用特徵數量：{len(pkg['feature_columns'])}")
    print(f"       類別：{list(pkg['label_encoder'].classes_)}")

    return pkg


def build_feature_vector_from_pose(results, feature_columns):
    """
    給定 MediaPipe Pose 的結果 + 模型需要的 feature_columns，
    回傳 shape = (1, num_features) 的 numpy array。

    feature_columns 通常像：
        ["x_0", "y_0", "z_0", "v_0", "x_1", "y_1", ...]
    """
    if not results.pose_landmarks:
        return None

    # 先用 dict 存所有已知的 landmark 值
    lm_dict = {}
    for i, lm in enumerate(results.pose_landmarks.landmark):
        lm_dict[f"x_{i}"] = lm.x
        lm_dict[f"y_{i}"] = lm.y
        lm_dict[f"z_{i}"] = lm.z
        lm_dict[f"v_{i}"] = lm.visibility

    # 依照 feature_columns 順序組成 feature 向量
    feat = []
    for col in feature_columns:
        feat.append(lm_dict.get(col, 0.0))  # 若沒找到就給 0.0

    feat = np.array(feat, dtype=np.float32).reshape(1, -1)
    return feat


def main():
    args = parse_args()

    posture_model_path = Path(args.posture_model)
    stress_model_path = Path(args.stress_model)

    # 1. 載入模型
    posture_pkg = load_model_package(posture_model_path)
    stress_pkg = load_model_package(stress_model_path)

    if posture_pkg is None and stress_pkg is None:
        print("[ERROR] 姿勢與壓力模型都沒載入成功，無法執行 demo。")
        return

    # 2. 初始化 MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    # 3. 開啟攝影機
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟攝影機 index={args.camera_index}")
        return

    print("[INFO] 按 'q' 關閉視窗並退出。")

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
                print("[WARN] 無法讀取畫面，結束。")
                break

            # OpenCV 讀進來是 BGR，轉成 RGB 給 MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            # 畫姿態骨架在原始 frame（BGR）
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            # 準備用於顯示的文字
            posture_text = "Posture: (no model)"
            stress_text = "Stress: (no model)"

            # 4. 若有姿勢模型，就做推論
            if posture_pkg is not None and results.pose_landmarks:
                feat = build_feature_vector_from_pose(
                    results, posture_pkg["feature_columns"]
                )
                if feat is not None:
                    clf = posture_pkg["model"]
                    le = posture_pkg["label_encoder"]
                    pred_idx = clf.predict(feat)[0]
                    pred_label = le.inverse_transform([pred_idx])[0]
                    posture_text = f"Posture: {pred_label}"
                else:
                    posture_text = "Posture: no pose"

            # 5. 若有壓力模型，就做推論
            if stress_pkg is not None and results.pose_landmarks:
                feat_s = build_feature_vector_from_pose(
                    results, stress_pkg["feature_columns"]
                )
                if feat_s is not None:
                    clf_s = stress_pkg["model"]
                    le_s = stress_pkg["label_encoder"]
                    pred_idx_s = clf_s.predict(feat_s)[0]
                    pred_label_s = le_s.inverse_transform([pred_idx_s])[0]
                    stress_text = f"Stress: {pred_label_s}"
                else:
                    stress_text = "Stress: no pose"

            # 6. 簡單推一個「專注狀態」示意（可依你的類別名稱微調）
            focus_text = ""
            if posture_pkg is not None and results.pose_landmarks:
                # 這裡直接用 posture_pred 來判斷
                # 例如 good_posture / look_left / look_right / lookdown_...
                if "good" in posture_text:
                    focus_text = "Focus: GOOD"
                elif any(x in posture_text.lower() for x in ["lookleft", "lookright", "lookdown"]):
                    focus_text = "Focus: DISTRACTED"
                else:
                    focus_text = "Focus: UNKNOWN"

            # 7. 在畫面上畫文字
            h, w, _ = frame.shape
            org1 = (10, 30)
            org2 = (10, 60)
            org3 = (10, 90)

            cv2.rectangle(frame, (5, 5), (400, 110), (0, 0, 0), thickness=-1)  # 半透明背景就先簡單黑底
            cv2.putText(frame, posture_text, org1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, stress_text, org2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if focus_text:
                cv2.putText(frame, focus_text, org3, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 8. 顯示畫面
            cv2.imshow("Realtime Posture & Stress Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Demo 結束。")


if __name__ == "__main__":
    main()
