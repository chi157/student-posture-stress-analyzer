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
    
    def is_behavior_changed(self, new_behavior):
        """
        檢查行為是否改變（用於事件計數）
        
        參數:
            new_behavior: 新的行為類別
            
        回傳:
            changed: 是否改變
        """
        if new_behavior != self.last_behavior and new_behavior != "neutral":
            self.last_behavior = new_behavior
            return True
        
        if new_behavior == "neutral":
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
