"""
realtime_demo.py
================

用途：
    使用 webcam + MediaPipe Pose 即時抓取姿態 landmarks，
    並載入：
        - models/posture_model.joblib
        - models/stress_model.joblib
    做「姿勢分類」與「壓力行為分類」，顯示在畫面上。

    額外統計：
        - 連續專注秒數（目前這一段連續專注的時間）
        - 累積專注秒數（本次執行以來，所有專注時間總和）
        - 壓力行為事件次數（每當壓力類別改變一次就 +1）

專注判定邏輯（簡易版，依姿勢模型輸出文字）：
    - Posture label 內含 "good"      -> Focus: GOOD
    - Posture label 內含 "lookleft" / "lookright" / "lookdown"
                                    -> Focus: DISTRACTED
    - 其他                             -> Focus: UNKNOWN

使用方式（建議在專案根目錄執行）：

    # 最標準方式（確保 scripts 是 package）
    python -m scripts.realtime_demo

    # 或指定攝影機 index / 模型路徑：
    python -m scripts.realtime_demo --camera-index 0 ^
        --posture-model models/posture_model.joblib ^
        --stress-model models/stress_model.joblib

操作：
    - 在顯示視窗時按下 'q' 離開。

前置條件：
    - 已經執行過：
        python scripts/extract_landmarks.py
        python scripts/augment_dataset.py --num-aug 3
        python -m scripts.train_posture_model
        python -m scripts.train_stress_model

需要套件（建議寫在 requirements.txt）：
    - opencv-python
    - mediapipe
    - numpy
    - joblib
"""

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import joblib

from scripts.utils.landmark_utils import build_feature_vector_from_pose

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam 即時姿勢 / 壓力行為偵測 Demo（含專注統計）")

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
    載入 train_*_model.py 存好的 joblib 包裝：
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


def main():
    args = parse_args()

    posture_model_path = Path(args.posture_model)
    stress_model_path = Path(args.stress_model)

    posture_pkg = load_model_package(posture_model_path)
    stress_pkg = load_model_package(stress_model_path)

    if posture_pkg is None and stress_pkg is None:
        print("[ERROR] 姿勢與壓力模型都沒載入成功，無法執行 demo。")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[ERROR] 無法開啟攝影機 index={args.camera_index}")
        return

    print("[INFO] 按 'q' 離開。")

    # ===== 專注與壓力相關的狀態變數 =====
    focus_state = "UNKNOWN"           # "GOOD" / "DISTRACTED" / "UNKNOWN"
    focus_start_time = None           # 目前這一段專注開始的時間戳
    continuous_focus_seconds = 0.0    # 目前這一段連續專注秒數
    total_focus_seconds = 0.0         # 本次執行累積專注秒數
    last_time = time.time()           # 用來估計每幀時間間隔

    stress_event_count = 0            # 壓力事件次數
    last_stress_label = None          # 上一幀的壓力類別（用來判斷「事件」次數）

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

            now = time.time()
            dt = now - last_time
            if dt < 0:
                dt = 0.0
            last_time = now

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

            posture_text = "Posture: (no model)"
            stress_text = "Stress: (no model)"
            focus_text = ""

            # ===== 姿勢推論 =====
            posture_label_raw = None  # 只存 label，不加前綴字串，方便判斷

            if posture_pkg is not None and results.pose_landmarks:
                feat = build_feature_vector_from_pose(results, posture_pkg["feature_columns"])
                if feat is not None:
                    clf = posture_pkg["model"]
                    le = posture_pkg["label_encoder"]
                    pred_idx = clf.predict(feat)[0]
                    pred_label = le.inverse_transform([pred_idx])[0]
                    posture_label_raw = str(pred_label)
                    posture_text = f"Posture: {posture_label_raw}"
                else:
                    posture_text = "Posture: no pose"

            # ===== 壓力推論 & 統計壓力事件次數 =====
            stress_label_raw = None

            if stress_pkg is not None and results.pose_landmarks:
                feat_s = build_feature_vector_from_pose(results, stress_pkg["feature_columns"])
                if feat_s is not None:
                    clf_s = stress_pkg["model"]
                    le_s = stress_pkg["label_encoder"]
                    pred_idx_s = clf_s.predict(feat_s)[0]
                    pred_label_s = le_s.inverse_transform([pred_idx_s])[0]
                    stress_label_raw = str(pred_label_s)
                    stress_text = f"Stress: {stress_label_raw}"
                else:
                    stress_text = "Stress: no pose"

            # 壓力事件計數邏輯：
            #   每當「壓力類別」與上一幀不同，就視為一個新的事件。
            #   （例如：none -> head_touch，或 head_touch -> chin_support）
            if stress_label_raw is not None:
                if last_stress_label is None:
                    # 第一次有預測，不算事件，只記錄
                    last_stress_label = stress_label_raw
                else:
                    if stress_label_raw != last_stress_label:
                        stress_event_count += 1
                        last_stress_label = stress_label_raw

            # ===== 專注狀態變化與時間統計 =====
            # 依姿勢 label 粗略判斷：
            #   good_*      -> GOOD
            #   lookleft/ookright/lookdown -> DISTRACTED
            #   其他        -> UNKNOWN
            prev_focus_state = focus_state

            if posture_label_raw is None:
                focus_state = "UNKNOWN"
            else:
                label_lower = posture_label_raw.lower()
                if "good" in label_lower:
                    focus_state = "GOOD"
                elif any(k in label_lower for k in ["lookleft", "look_right", "lookright", "lookdown"]):
                    focus_state = "DISTRACTED"
                else:
                    focus_state = "UNKNOWN"

            # 更新專注時間
            if focus_state == "GOOD":
                if prev_focus_state != "GOOD":
                    # 剛剛進入專注狀態
                    focus_start_time = now
                    continuous_focus_seconds = 0.0
                else:
                    # 持續專注：更新連續專注秒數
                    if focus_start_time is not None:
                        continuous_focus_seconds = now - focus_start_time
                # 無論是不是剛進入 / 持續，只要這一幀是 GOOD，就把 dt 累加到 total_focus_seconds
                total_focus_seconds += dt
            else:
                # 非 GOOD 狀態，連續專注歸零
                focus_start_time = None
                continuous_focus_seconds = 0.0

            # 專注狀態文字
            if focus_state == "GOOD":
                focus_text = "Focus: GOOD"
            elif focus_state == "DISTRACTED":
                focus_text = "Focus: DISTRACTED"
            else:
                focus_text = "Focus: UNKNOWN"

            # ===== 畫 HUD（左上：狀態；右下：統計） =====
            h, w, _ = frame.shape

            # 左上狀態框
            cv2.rectangle(frame, (5, 5), (440, 120), (0, 0, 0), thickness=-1)
            cv2.putText(frame, posture_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, stress_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, focus_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # 右下統計框
            stats_box_w = 420
            stats_box_h = 100
            stats_box_x1 = w - stats_box_w - 5
            stats_box_y1 = h - stats_box_h - 5
            stats_box_x2 = w - 5
            stats_box_y2 = h - 5

            cv2.rectangle(frame,
                          (stats_box_x1, stats_box_y1),
                          (stats_box_x2, stats_box_y2),
                          (0, 0, 0),
                          thickness=-1)

            # 統計文字內容
            cont_focus_str = f"Continuous focus: {int(continuous_focus_seconds)} s"
            total_focus_str = f"Total focus:      {int(total_focus_seconds)} s"
            stress_count_str = f"Stress events:    {stress_event_count}"

            cv2.putText(frame, cont_focus_str, (stats_box_x1 + 10, stats_box_y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
            cv2.putText(frame, total_focus_str, (stats_box_x1 + 10, stats_box_y1 + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            cv2.putText(frame, stress_count_str, (stats_box_x1 + 10, stats_box_y1 + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)

            cv2.imshow("Realtime Posture & Stress Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Demo 結束。")


if __name__ == "__main__":
    main()
