# 📘 Student Posture & Stress Detection

基於 MediaPipe + RandomForest + 自建資料集的**學生坐姿偵測**、**專注度偵測**、**壓力行為偵測**專案。

本專案提供：

* 🧍 **姿勢偵測**（含：良好坐姿、聳肩、向左看、向右看、低頭…）
* 😣 **壓力行為偵測**（抓頭、撐額頭、托腮、皺眉…）
* 👀 **視線 / 分心偵測**
* 📊 **專注度時長統計**
* 🔥 **Webcam 即時偵測 DEMO**

並且支援：

* 自己錄影建立資料集
* 資料增強（Data Augmentation）
* 以 CPU/GPU 執行
* 可在 Windows / macOS / Linux 運行

---

# 📁 目錄架構

```
student_posture_stress/
│
├─ data_raw/                  # 原始錄影資料
│   ├─ posture/
│   ├─ stress/
│   └─ baseline/
│
├─ data_landmarks/            # extract_landmarks.py 輸出
│   └─ all_landmarks.csv
│
├─ data_augmented/            # augment_dataset.py 輸出
│   └─ all_landmarks_augmented.csv
│
├─ models/
│   ├─ posture_model.joblib
│   └─ stress_model.joblib
│
├─ scripts/
│   ├─ extract_landmarks.py
│   ├─ augment_dataset.py
│   ├─ train_posture_model.py
│   ├─ train_stress_model.py
│   ├─ realtime_demo.py
│   └─ utils/
│       ├─ __init__.py
│       ├─ data_utils.py
│       ├─ feature_utils.py
│       └─ landmark_utils.py
│
└─ README.md
```

---

# 🎥 資料收集指引

每種行為拍 **三段影片**（每段 10–15 秒）。

## 姿勢（posture）

| 行為              | 用途說明        |
| --------------- | ----------- |
| good_posture    | 正常坐姿、挺直     |
| slouch          | 駝背 / 彎腰     |
| look_left       | 把臉往左偏       |
| look_right      | 把臉往右偏       |
| look_down       | 低頭看書 / 桌面   |
| lean_left       | 身體歪到左邊      |
| lean_right      | 身體歪到右邊      |
| far_from_screen | 人遠離鏡頭（距離拉開） |

---

## 壓力行為（stress）

| 行為           | 用途說明      |
| ------------ | --------- |
| head_touch   | 抓頭        |
| forehead_rub | 揉額頭       |
| chin_support | 手托下巴      |
| frown        | 皺眉，搭配臉部肌肉 |

---

## baseline（非必要）

| 行為      | 用途       |
| ------- | -------- |
| neutral | 靜坐、不明顯動作 |

---

# 🧠 **完整流程（一步一步照做即可）**

## 1️⃣ 安裝環境

```
pip install -r requirements.txt
```

---

## 2️⃣ 收集資料 → 放在 data_raw/

例如：

```
data_raw/posture/good_posture_1.mp4
data_raw/stress/head_touch_1.mp4
```

---

## 3️⃣ 抽取 landmarks

```
python scripts/extract_landmarks.py
```

輸出：

```
data_landmarks/all_landmarks.csv
```

---

## 4️⃣ 資料增強（推薦）

```
python scripts/augment_dataset.py --num-aug 3
```

輸出：

```
data_augmented/all_landmarks_augmented.csv
```

---

## 5️⃣ 訓練姿勢模型

```
python scripts/train_posture_model.py
```

---

## 6️⃣ 訓練壓力模型

```
python scripts/train_stress_model.py
```

---

## 7️⃣ 使用 Webcam 即時偵測

```
python scripts/realtime_demo.py
```

可同時輸出：

* 姿勢分類
* 壓力行為分類
* 專注狀態（良好 / 分心）
* 專注度時長統計（秒）
* 壓力行為次數（次）

---

# 🖥️ 目前 GPU / CPU 支援說明

MediaPipe Pose **基本上依賴 CPU**，沒有硬性要求 GPU。

但如果你未來換深度學習模型（如 MoveNet / BlazePose GPU 最佳化），建議以下規格：

## ✔ 你目前的硬體

假設你目前使用 **筆電 / 桌電 CPU 就能跑**
標準 MediaPipe Pose 近乎 30fps。

## 建議 GPU（非必要）：

| GPU                | 使用感受             |
| ------------------ | ---------------- |
| GTX 1650 / 1660    | 夠用，30–60fps      |
| RTX 2060 / 3050    | 更適合姿勢偵測專案        |
| RTX 3060 / 3070 以上 | 如果你未來要做深度模型訓練會更快 |

> ⚠ MediaPipe Pose 本身不是 GPU-heavy，所以 GPU → 加速有限。
> 只有你想自己改成 TensorRT / ONNX 才會需要。


---

# 📊 專注度統計 & 壓力事件統計（Realtime Demo 功能）

系統在 Webcam 即時偵測中，會自動計算三種資訊，這些數據都會即時顯示在螢幕右下角的 HUD（Head-Up Display）中。

---

## 1️⃣ 連續專注秒數（Continuous Focus Time）

* 表示「目前這一段時間」使用者已經連續維持專注狀態多久
* 當姿勢模型判定為 **良好坐姿 / good_posture 類** 時，就屬於專注狀態
* 若發生分心（如看左、看右、低頭），計時將自動歸零並重新開始

**用途：**

* 查看學生是否持續維持注意力
* 可用於設定「專注多久才算達標」（例如連續專注 10 秒即視為成功）

畫面顯示範例：

```
Continuous focus: 12 s
```

---

## 2️⃣ 累積專注秒數（Total Focus Time）

* 表示從即時偵測開始到目前為止
  **所有 GOOD 專注狀態的累積秒數**
* 不會因為中途分心而歸零
* 若重新啟動程式，計時會重新開始

**用途：**

* 適合用於課堂專注度分析
* 可觀察「整節課累積專注多少時間」

畫面顯示範例：

```
Total focus: 86 s
```

---

## 3️⃣ 壓力事件次數（Stress Events）

* 表示偵測到「壓力行為」時的事件次數
* 每當壓力模型輸出類別**改變**（例如從無 → 抓頭）即視為事件 +1
* 若行為在多個 frame 內都一樣，不會不斷增加計數
* 可判定學生是否在某段時間內壓力上升

支援的壓力行為（視你資料集而定）：

* 抓頭（head_touch）
* 揉額頭（forehead_rub）
* 托腮（chin_support）
* 皺眉（frown）

畫面顯示範例：

```
Stress events: 3
```

---

## 📘 HUD 視覺呈現（右下角）

系統會在畫面右下角顯示：

* **連續專注秒數**（即時變化）
* **累積專注秒數**（總和）
* **壓力事件次數**（行為變更次數）

示意：

```
Continuous focus: 5 s
Total focus: 34 s
Stress events: 2
```

---



