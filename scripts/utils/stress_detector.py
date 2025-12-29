"""
壓力行為偵測模組
使用規則式判斷來偵測壓力相關的行為
"""

import numpy as np


class StressDetector:
    """壓力行為偵測器 - 偵測抓頭、揉額頭、托腮等壓力行為"""
    
    # MediaPipe Pose 關鍵點索引
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    
    # MediaPipe Hand 關鍵點索引
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    def __init__(self):
        """初始化壓力行為偵測器"""
        self.current_behavior = "neutral"
        self.confidence = 0.0
        self.last_behavior = "neutral"
        
        # 皺眉基準值（在專注狀態時記錄）
        self.baseline_brow_eye_dist = None
        self.baseline_brow_distance = None
        self.baseline_samples = []
        self.baseline_set = False
        
        # 皺眉動作追蹤（偵測靠近->鬆開的動作）
        self.is_frowning = False        # 是否正在皺眉（眉頭靠近）
        self.frown_start_time = None    # 皺眉開始時間
        self.frown_duration = 0.0       # 皺眉持續時間
        self.min_frown_duration = 0.5   # 最小皺眉持續時間（0.5秒）
    
    def detect_stress_behavior(self, pose_landmarks, left_hand, right_hand, face_landmarks):
        """
        偵測壓力行為
        
        參數:
            pose_landmarks: 姿態關鍵點
            left_hand: 左手關鍵點
            right_hand: 右手關鍵點
            face_landmarks: 臉部關鍵點
            
        回傳:
            behavior: 壓力行為類別
            confidence: 信心度
        """
        if pose_landmarks is None:
            return "neutral", 0.0
        
        # 檢查各種壓力行為
        behavior, confidence = self._check_head_touch(pose_landmarks, left_hand, right_hand)
        if behavior:
            return behavior, confidence
        
        behavior, confidence = self._check_forehead_rub(pose_landmarks, left_hand, right_hand, face_landmarks)
        if behavior:
            return behavior, confidence
        
        behavior, confidence = self._check_chin_support(pose_landmarks, left_hand, right_hand)
        if behavior:
            return behavior, confidence
        
        behavior, confidence = self._check_frown(pose_landmarks, face_landmarks)
        if behavior:
            return behavior, confidence
        
        # 預設為無壓力行為
        return "neutral", 0.9
    
    def _check_head_touch(self, pose_landmarks, left_hand, right_hand):
        """
        檢測是否在抓頭
        判斷依據：手部接觸頭頂區域
        """
        try:
            # 取得頭部位置（使用鼻子作為參考）
            nose = pose_landmarks[self.NOSE]
            head_top_y = nose['y'] - 0.15  # 估計頭頂位置
            head_x = nose['x']
            
            # 檢查左手
            if left_hand is not None:
                wrist = left_hand[self.WRIST]
                middle_tip = left_hand[self.MIDDLE_TIP]
                
                # 如果手指在頭頂區域附近
                if (middle_tip['y'] < head_top_y + 0.1 and 
                    abs(middle_tip['x'] - head_x) < 0.15):
                    return "head_touch", 0.85
            
            # 檢查右手
            if right_hand is not None:
                wrist = right_hand[self.WRIST]
                middle_tip = right_hand[self.MIDDLE_TIP]
                
                # 如果手指在頭頂區域附近
                if (middle_tip['y'] < head_top_y + 0.1 and 
                    abs(middle_tip['x'] - head_x) < 0.15):
                    return "head_touch", 0.85
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_forehead_rub(self, pose_landmarks, left_hand, right_hand, face_landmarks):
        """
        檢測是否在揉額頭
        判斷依據：手部在額頭區域移動
        """
        try:
            # 取得額頭區域位置
            nose = pose_landmarks[self.NOSE]
            forehead_y = nose['y'] - 0.08  # 額頭位置
            forehead_x = nose['x']
            
            # 檢查左手
            if left_hand is not None:
                index_tip = left_hand[self.INDEX_TIP]
                middle_tip = left_hand[self.MIDDLE_TIP]
                
                # 計算手指平均位置
                hand_y = (index_tip['y'] + middle_tip['y']) / 2
                hand_x = (index_tip['x'] + middle_tip['x']) / 2
                
                # 如果手在額頭區域
                if (abs(hand_y - forehead_y) < 0.05 and 
                    abs(hand_x - forehead_x) < 0.1):
                    return "forehead_rub", 0.8
            
            # 檢查右手
            if right_hand is not None:
                index_tip = right_hand[self.INDEX_TIP]
                middle_tip = right_hand[self.MIDDLE_TIP]
                
                # 計算手指平均位置
                hand_y = (index_tip['y'] + middle_tip['y']) / 2
                hand_x = (index_tip['x'] + middle_tip['x']) / 2
                
                # 如果手在額頭區域
                if (abs(hand_y - forehead_y) < 0.05 and 
                    abs(hand_x - forehead_x) < 0.1):
                    return "forehead_rub", 0.8
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_chin_support(self, pose_landmarks, left_hand, right_hand):
        """
        檢測是否托腮
        判斷依據：手部支撐下巴位置
        """
        try:
            # 取得下巴位置（使用鼻子 + 偏移估計）
            nose = pose_landmarks[self.NOSE]
            chin_y = nose['y'] + 0.08  # 估計下巴位置
            chin_x = nose['x']
            
            # 檢查左手
            if left_hand is not None:
                wrist = left_hand[self.WRIST]
                thumb_tip = left_hand[self.THUMB_TIP]
                index_tip = left_hand[self.INDEX_TIP]
                
                # 計算手掌中心
                palm_y = (wrist['y'] + index_tip['y']) / 2
                palm_x = (wrist['x'] + index_tip['x']) / 2
                
                # 如果手掌在下巴區域下方（支撐狀態）
                if (abs(palm_y - chin_y) < 0.08 and 
                    abs(palm_x - chin_x) < 0.15 and
                    wrist['y'] > chin_y):
                    return "chin_support", 0.85
            
            # 檢查右手
            if right_hand is not None:
                wrist = right_hand[self.WRIST]
                thumb_tip = right_hand[self.THUMB_TIP]
                index_tip = right_hand[self.INDEX_TIP]
                
                # 計算手掌中心
                palm_y = (wrist['y'] + index_tip['y']) / 2
                palm_x = (wrist['x'] + index_tip['x']) / 2
                
                # 如果手掌在下巴區域下方（支撐狀態）
                if (abs(palm_y - chin_y) < 0.08 and 
                    abs(palm_x - chin_x) < 0.15 and
                    wrist['y'] > chin_y):
                    return "chin_support", 0.85
            
            return None, 0.0
        except:
            return None, 0.0
    
    def update_baseline(self, face_landmarks):
        """
        更新皺眉基準值（在專注狀態時調用）
        收集10個樣本後取平均作為基準值
        
        參數:
            face_landmarks: 臉部關鍵點
        """
        try:
            if face_landmarks is None:
                return
            
            # MediaPipe 臉部關鍵點索引
            LEFT_EYEBROW_INNER = 70
            RIGHT_EYEBROW_INNER = 300
            LEFT_EYE_TOP = 159
            RIGHT_EYE_TOP = 386
            
            # 取得關鍵點
            left_eyebrow_inner = face_landmarks[LEFT_EYEBROW_INNER]
            right_eyebrow_inner = face_landmarks[RIGHT_EYEBROW_INNER]
            left_eye_top = face_landmarks[LEFT_EYE_TOP]
            right_eye_top = face_landmarks[RIGHT_EYE_TOP]
            
            # 計算當前值
            left_brow_eye_dist = abs(left_eyebrow_inner['y'] - left_eye_top['y'])
            right_brow_eye_dist = abs(right_eyebrow_inner['y'] - right_eye_top['y'])
            avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
            
            brow_distance = abs(right_eyebrow_inner['x'] - left_eyebrow_inner['x'])
            
            # 收集樣本
            self.baseline_samples.append({
                'brow_eye_dist': avg_brow_eye_dist,
                'brow_distance': brow_distance
            })
            
            # 收集10個樣本後計算平均值
            if len(self.baseline_samples) >= 10:
                avg_brow_eye = sum(s['brow_eye_dist'] for s in self.baseline_samples) / len(self.baseline_samples)
                avg_brow_dist = sum(s['brow_distance'] for s in self.baseline_samples) / len(self.baseline_samples)
                
                self.baseline_brow_eye_dist = avg_brow_eye
                self.baseline_brow_distance = avg_brow_dist
                self.baseline_set = True
                
                # 清空樣本，準備下次更新
                self.baseline_samples = []
        except:
            pass
    
    def _check_frown(self, pose_landmarks, face_landmarks):
        """
        檢測是否皺眉
        判斷依據：眉頭靠近一段時間後鬆開，才算皺眉一次
        注意：只在正面時偵測，避免轉頭時誤判
        """
        import time
        
        try:
            if face_landmarks is None or pose_landmarks is None:
                # 沒有臉部資料時，重置狀態
                if self.is_frowning:
                    self.is_frowning = False
                    self.frown_start_time = None
                    self.frown_duration = 0.0
                return None, 0.0
            
            # 如果還沒設定基準值，不進行判斷
            if not self.baseline_set:
                return None, 0.0
            
            # 檢查臉部是否為正面（避免轉頭時誤判）
            # 使用左右耳朵可見度判斷
            LEFT_EAR = 7
            RIGHT_EAR = 8
            
            if LEFT_EAR in pose_landmarks and RIGHT_EAR in pose_landmarks:
                left_ear_vis = pose_landmarks[LEFT_EAR]['visibility']
                right_ear_vis = pose_landmarks[RIGHT_EAR]['visibility']
                
                # 如果左右耳可見度差異太大，表示正在轉頭，不偵測皺眉
                ear_vis_diff = abs(left_ear_vis - right_ear_vis)
                if ear_vis_diff > 0.15:  # 轉頭角度太大
                    # 重置皺眉狀態
                    if self.is_frowning:
                        self.is_frowning = False
                        self.frown_start_time = None
                        self.frown_duration = 0.0
                    return None, 0.0
            
            # MediaPipe 臉部關鍵點索引
            LEFT_EYEBROW_INNER = 70
            RIGHT_EYEBROW_INNER = 300
            LEFT_EYE_TOP = 159
            RIGHT_EYE_TOP = 386
            
            # 取得關鍵點
            left_eyebrow_inner = face_landmarks[LEFT_EYEBROW_INNER]
            right_eyebrow_inner = face_landmarks[RIGHT_EYEBROW_INNER]
            left_eye_top = face_landmarks[LEFT_EYE_TOP]
            right_eye_top = face_landmarks[RIGHT_EYE_TOP]
            
            # 計算當前值
            left_brow_eye_dist = abs(left_eyebrow_inner['y'] - left_eye_top['y'])
            right_brow_eye_dist = abs(right_eyebrow_inner['y'] - right_eye_top['y'])
            avg_brow_eye_dist = (left_brow_eye_dist + right_brow_eye_dist) / 2
            
            brow_distance = abs(right_eyebrow_inner['x'] - left_eyebrow_inner['x'])
            
            # 計算相對於基準值的變化（百分比）
            brow_eye_change = (self.baseline_brow_eye_dist - avg_brow_eye_dist) / self.baseline_brow_eye_dist
            brow_dist_change = (self.baseline_brow_distance - brow_distance) / self.baseline_brow_distance
            
            # 判斷當前是否眉頭靠近（正在皺眉）
            is_brows_close = False
            if brow_eye_change > 0.12 or brow_dist_change > 0.01:  # 降低閾值使其更靈敏
                is_brows_close = True
            
            current_time = time.time()
            
            # 狀態機轉換
            if is_brows_close:
                # 眉頭靠近：開始或繼續皺眉
                if not self.is_frowning:
                    # 剛開始皺眉
                    self.is_frowning = True
                    self.frown_start_time = current_time
                    self.frown_duration = 0.0
                else:
                    # 持續皺眉，累加時間
                    self.frown_duration = current_time - self.frown_start_time
                
                return None, 0.0  # 皺眉期間不回傳皺眉狀態
            
            else:
                # 眉頭鬆開：檢查是否完成一次皺眉動作
                if self.is_frowning:
                    # 皺眉結束，檢查持續時間
                    if self.frown_duration >= self.min_frown_duration:
                        # 持續時間足夠，認定為皺眉一次
                        self.is_frowning = False
                        self.frown_start_time = None
                        self.frown_duration = 0.0
                        return "frown", 0.80
                    else:
                        # 持續時間不足，不認定為皺眉
                        self.is_frowning = False
                        self.frown_start_time = None
                        self.frown_duration = 0.0
                
                return None, 0.0
            
        except:
            return None, 0.0
    
    def is_behavior_changed(self, new_behavior):
        """
        檢查行為是否改變（用於事件計數）
        只有當從 neutral 或不同的壓力行為切換到新的壓力行為時才計數
        
        參數:
            new_behavior: 新的行為類別
            
        回傳:
            changed: 是否改變
        """
        # 只有從非該行為狀態切換到該壓力行為時才計數一次
        if new_behavior != "neutral" and new_behavior != self.last_behavior:
            self.last_behavior = new_behavior
            return True
        
        # 回到 neutral 狀態時更新記錄
        if new_behavior == "neutral" and self.last_behavior != "neutral":
            self.last_behavior = new_behavior
        
        return False
    
    def get_behavior_label(self, behavior):
        """
        取得壓力行為的中文標籤
        
        參數:
            behavior: 行為類別
            
        回傳:
            label: 中文標籤
        """
        labels = {
            "neutral": "無壓力行為",
            "head_touch": "抓頭",
            "forehead_rub": "揉額頭",
            "chin_support": "托腮",
            "frown": "皺眉"
        }
        return labels.get(behavior, behavior)
