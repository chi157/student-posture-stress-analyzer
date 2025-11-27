"""
data_utils.py
=============

共用的資料處理工具：
- 讀取 landmarks 資料（含 fallback）
- 依 category 篩選（posture / stress / baseline ...）
- 拆成「原始資料」與「增強資料」
"""

from pathlib import Path
import pandas as pd

# PROJECT_ROOT: student_posture_stress
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_LANDMARKS_DIR = PROJECT_ROOT / "data_landmarks"


def load_dataset(input_path: Path, fallback_name: str = "all_landmarks.csv") -> pd.DataFrame:
    """
    嘗試讀取 input_path，如果不存在，就改讀 data_landmarks/<fallback_name>。

    參數：
        input_path:    優先使用的路徑
        fallback_name: 備援檔名（預設 all_landmarks.csv）

    回傳：
        pandas.DataFrame
    """
    if input_path.exists():
        print(f"[INFO] 使用輸入檔案：{input_path}")
        df = pd.read_csv(input_path)
    else:
        fallback = DATA_LANDMARKS_DIR / fallback_name
        if not fallback.exists():
            raise FileNotFoundError(
                f"找不到輸入檔案：{input_path}，也找不到備用檔案：{fallback}"
            )
        print(f"[WARN] 找不到 {input_path}，改用：{fallback}")
        df = pd.read_csv(fallback)

    print(f"[INFO] 原始總筆數：{len(df)}")
    return df


def select_category_rows(df: pd.DataFrame, category_name: str) -> pd.DataFrame:
    """
    只保留指定 category 的資料。

    例如：
        category_name = "posture" / "stress" / "baseline"
    """
    if "category" not in df.columns:
        raise ValueError("資料中找不到 'category' 欄位，請確認 extract_landmarks.py 的輸出格式。")

    df_cat = df[df["category"] == category_name].copy()
    print(f"[INFO] 資料筆數（category == '{category_name}'）：{len(df_cat)}")

    if len(df_cat) == 0:
        raise ValueError(f"沒有任何 category == '{category_name}' 的資料，請確認 data_raw/{category_name} 有影片。")

    return df_cat


def split_original_and_augmented(df_cat: pd.DataFrame):
    """
    將某一個 category 的資料拆成：
        - 原始資料 df_orig
        - 增強資料 df_aug

    若沒 is_augmented 欄位，則全部視為原始資料。
    """
    if "is_augmented" not in df_cat.columns:
        print("[WARN] 找不到 is_augmented 欄位，視所有資料為原始資料。")
        df_orig = df_cat.copy()
        df_aug = df_cat.iloc[0:0].copy()  # 空 DataFrame
    else:
        df_orig = df_cat[df_cat["is_augmented"] == 0].copy()
        df_aug = df_cat[df_cat["is_augmented"] == 1].copy()
        print(f"[INFO] 原始資料筆數：{len(df_orig)}")
        print(f"[INFO] 增強資料筆數：{len(df_aug)}")

        if len(df_orig) == 0:
            print("[WARN] 沒有 is_augmented == 0 的資料，全部視為原始資料。")
            df_orig = df_cat.copy()
            df_aug = df_cat.iloc[0:0].copy()

    return df_orig, df_aug
