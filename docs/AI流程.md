OK，我幫你把整個「自己演＋資料增強」→ 做出壓力/姿勢/專注偵測系統的流程，拆成一條完整開發線，你可以照著一步步做。

---

## 0. 先決定你到底要「辨識哪些狀態」

建議一開始不要太多類別，先列出你**真的會用到的標籤**，例如：

1. **專注狀態（Attention）**

   * `focus_good`：看螢幕、坐好
   * `focus_bad`：東張西望 / 低頭用手機 / 看旁邊

2. **姿勢（Posture）**

   * `posture_good`：背挺直、距離正常
   * `posture_bad`：駝背 / 太靠近螢幕 / 整個人滑下去

3. **壓力行為（Stress gestures）**

   * `stress_none`：沒有明顯壓力動作
   * `stress_head_touch`：抓頭 / 扶額 / 摸頭髮
   * （皺眉可以先暫緩或跟 head_touch 一起算「壓力高」）

一開始可以只做兩個模型：

* **模型 A：專注度 + 姿勢（好 / 壞）**
* **模型 B：有沒有壓力動作（none / head_touch）**

之後要再細分類別，再慢慢加。

---

## 1. 錄製資料 — 自己演 / 找朋友演

### 1-1. 規劃要錄的情境

針對每個標籤，設計幾段 10–30 秒的演戲：

例：

* `focus_good + posture_good`

  * 正常坐好，看螢幕，偶爾眨眼
* `focus_bad`

  * 一直看旁邊、抬頭放空、低頭看手機
* `posture_bad`

  * 越坐越前傾、駝背、整個頭靠近螢幕
* `stress_head_touch`

  * 抓頭、揉額頭、雙手撐頭、抓頭髮

**小技巧：**

* 可以一次錄「連續情境」，例如：先坐好 10 秒 → 然後開始駝背 10 秒 → 抓頭 10 秒
  之後用時間段去切割與標記。

### 1-2. 錄影工具與格式

用 Python + OpenCV 直接錄，或用一般相機錄好再丟進電腦都可以，但建議：

* 固定解析度（例如 1280×720）
* 固定 fps（例如 30 fps）
* 相機位置固定（你之後上課時也希望差不多位置）

**資料夾建議結構：**

```text
data_raw/
  session1_alex/
    focus_good_posture_good_01.mp4
    focus_bad_01.mp4
    posture_bad_01.mp4
    stress_head_touch_01.mp4
  session2_friend/
    ...
```

之後標籤可以寫在檔名或用一個 `labels.csv` 管理。

---

## 2. 用 MediaPipe 把影片轉成「landmarks 資料集」

這一步是關鍵：**影片 → 每一幀的關鍵點座標 + 標籤**。

流程概念：

1. 寫一個 script：

   * 讀取每一個 `.mp4`
   * 用 MediaPipe Holistic / Pose + FaceMesh 取得：

     * 臉 landmark（或至少：鼻子、雙眼、嘴角、眉毛幾點）
     * 身體 landmark（肩、耳、鼻、臀）

2. 每一幀輸出一筆紀錄（或滑動視窗的一段連續幀一筆）：

```text
video_id, frame_idx, x1, y1, z1, x2, y2, z2, ..., label_focus, label_posture, label_stress
```

3. 存成 `CSV` 或 `numpy` / `parquet`。

> 標籤的對應方式：
>
> * 最簡單：用「整段影片」都當成同一個標籤（例如這段影片就是 `focus_good`）。
> * 如果你錄的是「一段影片裡有多個情境」，就要記錄每個情境開始時間 & 結束時間，用 frame index 判斷。

---

## 3. 做特徵工程（不要直接用原始座標）

原始座標會受畫面大小與位置影響，建議：

### 3-1. 正規化

* 先把所有點的座標**轉成相對位置**：

  * 以某個基準點為中心（例如：hip 或 nose）
  * 或用「肩膀距離」做縮放，讓不同距離的人也能對齊

例：

```text
x_norm = (x - x_center) / shoulder_width
y_norm = (y - y_center) / shoulder_width
```

### 3-2. 把重要的幾個距離/角度抽出來

比方說：

* 頭與肩的角度（判斷駝背）
* 手腕到頭部的距離（判斷抓頭）
* 頭部相對臀部的水平偏移、前傾程度
* 臉占畫面的比例（可用臉 bounding box 寬/高 vs 全畫面）

你可以讓每一幀最後變成一個「特徵向量」，例如：

```text
[head_tilt_angle,
 distance_head_to_screen_proxy,
 hand_to_head_distance_left,
 hand_to_head_distance_right,
 ...]
```

這樣特徵更有「語意」，模型會更好學。

### 3-3. 使用時間資訊（可先略過）

如果要考慮「一段連續幀」的動作，可以：

* 用「N 幀的平均值 / 最大值」作為一筆樣本
* 或用 RNN / LSTM 等模型（比較複雜，可以晚點再加）

一開始可以直接 **每幀算一次 + 再用簡單平滑** 就好。

---

## 4. 資料增強（Data Augmentation）— 在 landmark 上做

你已經有每幀的 landmark 特徵後，可以在程式裡做這些變形：

### 4-1. 幾何變形

* 小幅平移整個 skeleton（模擬人稍微左右偏）
* 小幅縮放（模擬距離稍有不同）
* 小幅旋轉（模擬相機角度 / 頭微歪）

注意 **不要變太誇張**，不然 label 會失真。

### 4-2. 加噪音（noise）

* 在每個座標上加一點高斯噪音，模擬偵測誤差，避免模型太敏感。

### 4-3. 時間切片 / 重複

* 若你用多幀 window，可以用「隨機起點」取同一段影片不同片段，增加樣本。

這些通常都是在「訓練時的 dataloader 裡」做，而不是預先存很多版本的檔案。

---

## 5. 訓練簡單模型（先從 Scikit-Learn 開始）

### 5-1. 切資料

* 把所有樣本切成：

  * 訓練集（Train）
  * 驗證集（Val）
  * 測試集（Test）

建議依「影片」切，不要把同一支影片拆去 train/test，避免洩漏。

### 5-2. 選模型

對於這種 tabular 特徵（距離、角度），非常適合：

* **Random Forest**
* **XGBoost / LightGBM**
* 或簡單的 **MLP（fully connected NN）**

例如對「姿勢好壞」：

* Input：一幀的特徵向量
* Output：`0 = bad`, `1 = good`

對「壓力動作」：

* Input：一幀的手-頭距離、手的動作特徵
* Output：`0 = 無`, `1 = head_touch`

你可以做兩個獨立的分類器，簡化問題。

### 5-3. 評估

看幾個指標：

* Accuracy（整體正確率）
* Precision / Recall（尤其是壓力動作，避免誤報太多）
* 看混淆矩陣：模型容易把什麼誤判成什麼

如果效果不理想，可以回去：

* 再調整特徵（加/減一些關鍵距離或角度）
* 或重新設計標籤（例如只分「正常 / 異常」）

---

## 6. 實時系統整合（線上推論）

當你有訓練好的模型（例如 `.pkl` / `.pt` 檔），就可以：

1. 用 MediaPipe 即時抓 webcam 畫面

2. 每幀：

   * 取得 landmarks
   * 做同樣的特徵轉換（正規化、計算角度、距離）
   * 丟進模型 `predict` → 得到：

     * 姿勢好/壞
     * 有無壓力動作
     * 是否專注

3. 做「時間平滑」：

   * 使用最近 N 幀的結果做 majority vote
     → 避免一兩幀抖動就觸發提醒

4. 邏輯例子：

* 若「姿勢壞」連續超過 5 秒 → 顯示「請挺直背」
* 若「專注度低」持續 10 秒 → 顯示「請回到螢幕」
* 若「壓力動作」過於頻繁（例如 1 分鐘內 > X 次）→ 標記為「可能壓力高」

5. 同時記錄 log：

* 每分鐘專注秒數
* 姿勢不良次數與總時長
* 壓力動作次數

可以之後做統計或畫圖。

---

## 7. 迭代優化

跑一陣子後，你會發現：

* 哪些情況常被誤判
* 哪些人（高個子、戴眼鏡、鏡頭位置不同）特別不準

接下來你可以：

1. 特別錄那種「常錯的情境」再加到資料裡
2. 加入簡單的「個人校正」：

   * 第一次使用請他坐好 5 秒 → 設那幾秒為個人 baseline
   * 之後所有判斷都以「偏離 baseline」為主，而不是絕對角度

---

## 8. 總流程快速 recap（給你當 checklist）

1. **定義標籤**：focus / posture / stress 分類
2. **錄影**：自己 & 朋友演不同情境，存 `data_raw/`
3. **跑 MediaPipe**：把影片轉成 landmarks（每幀）
4. **特徵工程**：正規化 + 重要距離/角度 + 一些時間統計
5. **資料增強**：在 landmark 特徵上做平移/旋轉/縮放/加噪音
6. **訓練模型**：用 Random Forest 等簡單分類器各訓練一個任務
7. **評估 & 調參**：看錯在哪裡，調特徵與 rule
8. **即時整合**：webcam + MediaPipe + 模型推論 + 時間平滑 + 提醒/統計
9. **迭代**：依實際使用情況再錄新資料補強

---

如果你接下來有打算用某個具體技術棧（例如 Python + MediaPipe + scikit-learn），我可以直接幫你寫一份「專案資料夾結構＋關鍵程式檔要做什麼」的範本，甚至加上 pseudo-code，讓你照著填就好。




接下來你整個流程就是：

錄好所有影片 → 放進對應 data_raw/... 資料夾

抽 landmarks：

python scripts/extract_landmarks.py


資料增強：

python scripts/augment_dataset.py --num-aug 3


訓練姿勢模型：

python scripts/train_posture_model.py


訓練壓力模型：

python scripts/train_stress_model.py


即時 demo：

python scripts/realtime_demo.py