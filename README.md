# 📘 Student Posture & Stress Detection

基於 **MediaPipe** 的**學生坐姿偵測**、**專注度偵測**、**壓力行為偵測**專案。

本專案提供：

* 🧍 **姿勢偵測**（含：良好坐姿、聳肩、向左看、向右看、低頭…）
* 😣 **壓力行為偵測**（抓頭、撐額頭、托腮、皺眉…）
* 👀 **視線 / 分心偵測**
* 📊 **專注度時長統計**
* 🔥 **Webcam 即時偵測 DEMO**

並且支援：

* 純 MediaPipe 規則式檢測，無需訓練模型
* 以 CPU/GPU 執行
* 可在 Windows / macOS / Linux 運行

---

# 📁 目錄架構

```
student_posture_stress/
│
├─ scripts/
│   ├─ realtime_demo.py        # 即時偵測主程式
│   └─ utils/
│       ├─ __init__.py
│       ├─ posture_detector.py # 姿勢檢測邏輯
│       ├─ stress_detector.py  # 壓力行為檢測邏輯
│       └─ landmark_utils.py   # MediaPipe 相關工具
│
└─ README.md
```

---

# � 偵測功能說明

## 姿勢偵測（Posture Detection）

透過 MediaPipe Pose 關鍵點分析，自動偵測：

| 姿勢              | 偵測方式                |
| --------------- | ------------------- |
| good_posture    | 身體挺直、肩膀水平           |
| slouch          | 肩膀明顯下沉、駝背           |
| look_left       | 臉部朝向左側              |
| look_right      | 臉部朝向右側              |
| look_down       | 頭部向下傾斜              |
| lean_left       | 身體重心偏左              |
| lean_right      | 身體重心偏右              |
| far_from_screen | 人體關鍵點距離鏡頭過遠（整體縮小） |

---

## 壓力行為偵測（Stress Detection）

透過 MediaPipe Holistic（手部 + 臉部）關鍵點分析：

| 行為           | 偵測方式        |
| ------------ | ----------- |
| head_touch   | 手部接觸頭部區域    |
| forehead_rub | 手部在額頭區域移動   |
| chin_support | 手部支撐下巴位置    |
| frown        | 臉部眉毛區域變化（選用） |

---

# 🧠 **完整流程（一步一步照做即可）**

## 1️⃣ 安裝環境

```
pip install -r requirements.txt
```快速開始（兩步驟即可運行）**

## 1️⃣ 安裝環境

```bash
pip install -r requirements.txt
```

需要的套件：
- mediapipe
- opencv-python
- numpy

---

## 2️⃣ 啟動 Webcam 即時偵測

```bash
python scripts/realtime_demo.py
```

系統會自動開啟 Webcam 並即時顯示：

* ✅ **姿勢分類**（良好坐姿 / 駝背 / 向左看 / 向右看 / 低頭等）
* ⚠️ **壓力行為分類**（抓頭 / 揉額頭 / 托腮等）
* 🎯 **專注狀態**（FOCUSED / DISTRACTED）
* ⏱️ **連續專注時長**（秒）
* 📊 **累積專注時長**（秒）
* 🔔 **壓力事件次數**（次）

---

## 🔧 偵測邏輯說明

本專案使用 **MediaPipe** 提供的人體姿態和手部關鍵點，透過**規則式判斷**進行偵測：

- **姿勢偵測**：分析肩膀、頭部、身體角度和距離
- **壓力行為偵測**：計算手部與臉部的相對位置
- **專注度判斷**：根據姿勢是否為良好坐姿來計算

**優點**：
- 無需訓練資料集
- 無需訓練模型
- 即裝即用
- 適合快速原型開發ose **基本上依賴 CPU**，沒有硬性要求 GPU。

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

* 表示「目硬體需求

MediaPipe **主要使用 CPU 運算**，一般筆電 / 桌電即可流暢運行。

## ✔ 建議配置

| 硬體    | 需求           |
| ----- | ------------ |
| CPU   | Intel i5 或以上 |
| RAM   | 4GB 以上      |
| Webcam | 任何 USB 攝影機  |

## 效能表現

- 標準筆電：可達 **20-30 FPS**
- 桌上型電腦：可達 **30+ FPS**

> 💡 **提示**：MediaPipe 已經過高度優化，GPU 加速效果有限。CPU 版本即可滿足即時偵測需求
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



