"""
realtime_demo.py
================

用途：
    使用 webcam + MediaPipe Pose 即時抓取姿態 landmarks，
    並載入：
        - models/posture_model.joblib
        - models/stress_model.joblib
    做「姿勢分類」與「壓力行為分類」，顯示在畫面上。

使用方式（專案根目錄）：
    python -m scripts.realtime_demo

或：
    python -m scripts.realtime_demo --camera-index 0
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import joblib

from scripts.utils.landmark_utils import build_feature_vector_from_pose

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

            # 姿勢推論
            if posture_pkg is not None and results.pose_landmarks:
                feat = build_feature_vector_from_pose(results, posture_pkg["feature_columns"])
                if feat is not None:
                    clf = posture_pkg["model"]
                    le = posture_pkg["label_encoder"]
                    pred_idx = clf.predict(feat)[0]
                    pred_label = le.inverse_transform([pred_idx])[0]
                    posture_text = f"Posture: {pred_label}"
                else:
                    posture_text = "Posture: no pose"

            # 壓力推論
            if stress_pkg is not None and results.pose_landmarks:
                feat_s = build_feature_vector_from_pose(results, stress_pkg["feature_columns"])
                if feat_s is not None:
                    clf_s = stress_pkg["model"]
                    le_s = stress_pkg["label_encoder"]
                    pred_idx_s = clf_s.predict(feat_s)[0]
                    pred_label_s = le_s.inverse_transform([pred_idx_s])[0]
                    stress_text = f"Stress: {pred_label_s}"
                else:
                    stress_text = "Stress: no pose"

            # 簡易專注狀態（示意）
            if posture_pkg is not None and results.pose_landmarks:
                if "good" in posture_text.lower():
                    focus_text = "Focus: GOOD"
                elif any(x in posture_text.lower() for x in ["lookleft", "lookright", "lookdown"]):
                    focus_text = "Focus: DISTRACTED"
                else:
                    focus_text = "Focus: UNKNOWN"

            # 畫文字
            cv2.rectangle(frame, (5, 5), (420, 110), (0, 0, 0), thickness=-1)
            cv2.putText(frame, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, stress_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if focus_text:
                cv2.putText(frame, focus_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Realtime Posture & Stress Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Demo 結束。")


if __name__ == "__main__":
    main()
