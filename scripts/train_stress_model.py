"""
train_stress_model.py
=====================

用途：
    使用 MediaPipe Pose / Face 抽出的 landmarks（含資料增強）來訓練「壓力行為分類模型」。
    只會使用 category == 'stress' 的資料（例如：head_touch, forehead_rub, chin_support, frown）。

資料來源優先順序：
    1) data_augmented/all_landmarks_augmented.csv （建議先跑 augment_dataset.py）
    2) 若上面不存在，退回使用 data_landmarks/all_landmarks.csv（無增強）

訓練邏輯重點：
    - 只用「原始資料」（is_augmented == 0）來切 train / val / test
    - 增強資料（is_augmented == 1）只加入 train，不進 val / test
    - 避免資料洩漏（data leakage），驗證與測試更真實

輸出：
    - models/stress_model.joblib   （內含：model, label_encoder, feature_columns）

使用方式（請在專案根目錄執行，例如 VSCode 終端機）：

    # 1. 先抽 landmarks（如果還沒做）
    python scripts/extract_landmarks.py

    # 2. 再做資料增強（可選，但很推薦）
    python scripts/augment_dataset.py --num-aug 3

    # 3. 訓練壓力行為模型（會自動找 augmented 檔案）
    python scripts/train_stress_model.py

    # 如果想指定輸入 / 模型輸出檔名：
    python scripts/train_stress_model.py ^
        --input data_augmented/all_landmarks_augmented.csv ^
        --model-out models/stress_model.joblib

需要套件（建議寫進 requirements.txt）：
    - pandas
    - numpy
    - scikit-learn
    - joblib
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 專案根目錄（scripts/ 的上一層）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_LANDMARKS_DIR = PROJECT_ROOT / "data_landmarks"
DATA_AUGMENTED_DIR = PROJECT_ROOT / "data_augmented"
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(description="訓練壓力行為分類模型（stress model）")

    parser.add_argument(
        "--input",
        type=str,
        default=str(DATA_AUGMENTED_DIR / "all_landmarks_augmented.csv"),
        help=(
            "輸入 CSV 檔路徑。預設為 data_augmented/all_landmarks_augmented.csv，"
            "若不存在則自動改用 data_landmarks/all_landmarks.csv"
        ),
    )

    parser.add_argument(
        "--model-out",
        type=str,
        default=str(MODELS_DIR / "stress_model.joblib"),
        help="輸出模型檔案路徑（預設：models/stress_model.joblib）",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="測試集比例（預設 0.15）",
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="驗證集比例（預設 0.15）",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="隨機種子（預設 42）",
    )

    return parser.parse_args()


def load_dataset(input_path: Path) -> pd.DataFrame:
    """
    嘗試讀取 input_path，如果不存在，改讀 data_landmarks/all_landmarks.csv。
    """
    if input_path.exists():
        print(f"[INFO] 使用輸入檔案：{input_path}")
        df = pd.read_csv(input_path)
    else:
        fallback = DATA_LANDMARKS_DIR / "all_landmarks.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"找不到輸入檔案：{input_path}，也找不到備用檔案：{fallback}"
            )
        print(f"[WARN] 找不到 {input_path}，改用：{fallback}")
        df = pd.read_csv(fallback)

    print(f"[INFO] 原始總筆數：{len(df)}")
    return df


def select_stress_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    只保留 category == 'stress' 的資料作為壓力模型的資料。
    """
    if "category" not in df.columns:
        raise ValueError("資料中找不到 'category' 欄位，請確認 extract_landmarks.py 的輸出格式。")

    df_stress = df[df["category"] == "stress"].copy()
    print(f"[INFO] 壓力資料筆數（category == 'stress'）：{len(df_stress)}")

    if len(df_stress) == 0:
        raise ValueError("沒有任何 category == 'stress' 的資料，請確認 data_raw/stress 有影片。")

    return df_stress


def split_original_and_augmented(df_stress: pd.DataFrame):
    """
    將壓力資料分為原始資料 & 增強資料。
    如果沒有 is_augmented 欄位，全部視為原始資料。
    """
    if "is_augmented" not in df_stress.columns:
        print("[WARN] 找不到 is_augmented 欄位，視所有資料為原始資料。")
        df_orig = df_stress.copy()
        df_aug = df_stress.iloc[0:0].copy()
    else:
        df_orig = df_stress[df_stress["is_augmented"] == 0].copy()
        df_aug = df_stress[df_stress["is_augmented"] == 1].copy()
        print(f"[INFO] 原始資料筆數：{len(df_orig)}")
        print(f"[INFO] 增強資料筆數：{len(df_aug)}")

        if len(df_orig) == 0:
            print("[WARN] 沒有 is_augmented == 0 的資料，全部視為原始資料。")
            df_orig = df_stress.copy()
            df_aug = df_stress.iloc[0:0].copy()

    return df_orig, df_aug


def get_feature_columns(df: pd.DataFrame):
    """
    取出特徵欄位（x_*, y_*, z_*, v_*），排除 meta 欄位。
    """
    exclude_cols = {
        "video_path",
        "frame_idx",
        "category",
        "sub_label",
        "label",
        "is_augmented",
        "aug_id",
    }

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


def main():
    args = parse_args()

    input_path = Path(args.input)
    model_out_path = Path(args.model_out)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 讀資料
    df = load_dataset(input_path)

    # 2. 專注於壓力資料
    df_stress = select_stress_rows(df)

    # 3. 切原始 & 增強
    df_orig, df_aug = split_original_and_augmented(df_stress)

    # 4. 決定特徵欄位
    feature_cols = get_feature_columns(df_stress)

    # 5. 設定標籤（用 sub_label 做多類別，例如 head_touch, forehead_rub ...）
    target_col = "sub_label"
    if target_col not in df_stress.columns:
        raise ValueError("資料中找不到 'sub_label' 欄位，請確認 extract_landmarks.py 的輸出格式。")

    # 只用原始資料來做切分
    X_orig = df_orig[feature_cols].values.astype(np.float32)
    y_orig = df_orig[target_col].astype(str).values

    # 6. 切 train / test  (先切出 test)
    test_size = args.test_size
    val_size = args.val_size

    if test_size + val_size >= 0.9:
        raise ValueError("test_size + val_size 太大，請調小一點。")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_orig,
        y_orig,
        test_size=test_size,
        random_state=args.random_state,
        stratify=y_orig,
    )

    # 7. 再從 trainval 裡面切出 val
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_ratio,
        random_state=args.random_state,
        stratify=y_trainval,
    )

    print(f"[INFO] Train 原始資料筆數：{len(X_train)}")
    print(f"[INFO] Val   原始資料筆數：{len(X_val)}")
    print(f"[INFO] Test  原始資料筆數：{len(X_test)}")

    # 8. 如果有增強資料，加入 train
    if len(df_aug) > 0:
        X_aug = df_aug[feature_cols].values.astype(np.float32)
        y_aug = df_aug[target_col].astype(str).values

        print(f"[INFO] 將增強資料 {len(X_aug)} 筆加入 Train")
        X_train = np.concatenate([X_train, X_aug], axis=0)
        y_train = np.concatenate([y_train, y_aug], axis=0)

    print(f"[INFO] Train 總筆數（含增強）：{len(X_train)}")

    # 9. Label 編碼
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    print("[INFO] 類別對應：")
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        print(f"  {cls_idx}: {cls_name}")

    # 10. 建立模型（RandomForest）
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=args.random_state,
        n_jobs=-1,
    )

    print("[INFO] 開始訓練壓力模型（RandomForestClassifier）...")
    clf.fit(X_train, y_train_enc)

    # 11. 評估：Val
    print("\n[INFO] 驗證集結果 (Val)：")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val_enc, y_val_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix (Val):")
    print(confusion_matrix(y_val_enc, y_val_pred))

    # 12. 評估：Test
    print("\n[INFO] 測試集結果 (Test)：")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test_enc, y_test_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test_enc, y_test_pred))

    # 13. 存模型（包含 label encoder & feature cols）
    model_package = {
        "model": clf,
        "label_encoder": label_encoder,
        "feature_columns": feature_cols,
        "target_col": target_col,
    }

    joblib.dump(model_package, model_out_path)
    print(f"\n[INFO] 模型已儲存到：{model_out_path}")


if __name__ == "__main__":
    main()
