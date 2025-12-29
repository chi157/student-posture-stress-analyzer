"""
姿勢偵測模組
使用規則式判斷來偵測各種坐姿狀態
"""

import numpy as np


class PostureDetector:
    """姿勢偵測器 - 使用規則判斷學生坐姿"""
    
    # MediaPipe Pose 關鍵點索引
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_HIP = 23
    RIGHT_HIP = 24
    
    def __init__(self):
        """初始化姿勢偵測器"""
        self.current_posture = "unknown"
        self.confidence = 0.0
        # 添加穩定性追蹤
        self.posture_history = []
        self.history_size = 3  # 記錄最近3幀
    
    def detect_posture(self, pose_landmarks):
        """
        偵測當前姿勢
        
        參數:
            pose_landmarks: 姿態關鍵點字典
            
        回傳:
            posture: 姿勢類別
            confidence: 信心度
        """
        if pose_landmarks is None:
            return "no_person", 0.0
        
        # 檢查各種姿勢（依優先順序）
        posture, confidence = self._check_far_from_screen(pose_landmarks)
        if posture:
            return self._stabilize_posture(posture, confidence)
        
        # 優先檢查轉頭動作（看上、看下、看左、看右）
        posture, confidence = self._check_look_up(pose_landmarks)
        if posture:
            return self._stabilize_posture(posture, confidence)
        
        posture, confidence = self._check_look_down(pose_landmarks)
        if posture:
            return self._stabilize_posture(posture, confidence)
        
        posture, confidence = self._check_look_left_right(pose_landmarks)
        if posture:
            return self._stabilize_posture(posture, confidence)
        
        posture, confidence = self._check_lean(pose_landmarks)
        if posture:
            return self._stabilize_posture(posture, confidence)
        
        posture, confidence = self._check_slouch(pose_landmarks)
        if posture:
            return self._stabilize_posture(posture, confidence)
        
        # 預設為良好坐姿
        return self._stabilize_posture("good_posture", 0.8)
    
    def _check_far_from_screen(self, landmarks):
        """
        檢測是否遠離螢幕
        判斷依據：整體關鍵點的深度（z 值）較大
        """
        try:
            # 取得肩膀的深度值（z 越大表示離鏡頭越遠）
            left_shoulder_z = landmarks[self.LEFT_SHOULDER]['z']
            right_shoulder_z = landmarks[self.RIGHT_SHOULDER]['z']
            avg_z = (left_shoulder_z + right_shoulder_z) / 2
            
            # 如果平均深度大於閾值，判定為遠離螢幕（降低閾值使其更靈敏）
            if avg_z > 0.3:
                return "far_from_screen", 0.9
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_look_up(self, landmarks):
        """
        檢測是否抬頭向上看
        判斷依據：鼻子明顯高於眼睛，或下巴抬高
        """
        try:
            nose = landmarks[self.NOSE]
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            # 計算眼睛中心點
            eye_center_y = (left_eye['y'] + right_eye['y']) / 2
            
            # 計算鼻子到眼睛的垂直距離（抬頭時鼻子會高於眼睛）
            nose_to_eye_distance = eye_center_y - nose['y']
            
            # 抬頭的判斷條件：鼻子明顯高於眼睛中心（降低閾值）
            if nose_to_eye_distance > 0.02:
                return "look_up", 0.85
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_look_down(self, landmarks):
        """
        檢測是否低頭
        判斷依據：鼻子 y 座標明顯低於眼睛，或頭部傾斜角度
        """
        try:
            nose = landmarks[self.NOSE]
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            
            # 計算眼睛中心點
            eye_center_y = (left_eye['y'] + right_eye['y']) / 2
            
            # 計算肩膀中心點
            shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            
            # 計算鼻子到眼睛的垂直距離（正常應該是鼻子在眼睛下方一點點）
            nose_to_eye_distance = nose['y'] - eye_center_y
            
            # 計算鼻子到肩膀的距離（用來判斷頭是否過低）
            nose_to_shoulder_distance = shoulder_center_y - nose['y']
            
            # 低頭的判斷條件（降低閾值提高靈敏度）：
            # 1. 鼻子明顯低於眼睛（正常應該只低一點點，約0.02-0.05）
            # 2. 鼻子太接近肩膀（頭垂得很低）
            if nose_to_eye_distance > 0.1 or nose_to_shoulder_distance < 0.15:
                return "look_down", 0.85
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_look_left_right(self, landmarks):
        """
        檢測是否向左或向右看
        判斷依據：左右眼睛與鼻子的相對位置、耳朵可見度
        """
        try:
            nose = landmarks[self.NOSE]
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_ear = landmarks[self.LEFT_EAR]
            right_ear = landmarks[self.RIGHT_EAR]
            
            # 計算左右眼睛和耳朵的可見度
            left_eye_vis = landmarks[self.LEFT_EYE]['visibility']
            right_eye_vis = landmarks[self.RIGHT_EYE]['visibility']
            left_ear_vis = landmarks[self.LEFT_EAR]['visibility']
            right_ear_vis = landmarks[self.RIGHT_EAR]['visibility']
            
            # 計算鼻子相對於眼睛中心的偏移
            eye_center_x = (left_eye['x'] + right_eye['x']) / 2
            nose_offset = nose['x'] - eye_center_x
            
            # 計算耳朵可見度差異
            ear_vis_diff = left_ear_vis - right_ear_vis
            
            # 計算眼睛可見度差異
            eye_vis_diff = left_eye_vis - right_eye_vis
            
            # 注意：因為影像已經鏡像翻轉，所以左右邏輯需要對調
            # 當 left_ear_vis 高時，使用者實際上是在看右邊（自己的右邊）
            # 當 right_ear_vis 高時，使用者實際上是在看左邊（自己的左邊）
            
            # 使用者看左邊的判斷（鏡像後 right_ear 更可見）
            # 大幅降低閾值，使用 OR 邏輯，任一條件滿足即可
            if (ear_vis_diff < -0.05) or \
               (nose_offset < -0.01) or \
               (eye_vis_diff < -0.04):
                return "look_left", 0.75
            
            # 使用者看右邊的判斷（鏡像後 left_ear 更可見）
            # 大幅降低閾值，使用 OR 邏輯，任一條件滿足即可
            if (ear_vis_diff > 0.05) or \
               (nose_offset > 0.01) or \
               (eye_vis_diff > 0.04):
                return "look_right", 0.75
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_lean(self, landmarks):
        """
        檢測身體是否向左或向右傾斜
        判斷依據：肩膀連線與水平線的夾角
        """
        try:
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            left_hip = landmarks[self.LEFT_HIP]
            right_hip = landmarks[self.RIGHT_HIP]
            
            # 計算肩膀傾斜角度
            shoulder_slope = (right_shoulder['y'] - left_shoulder['y']) / (right_shoulder['x'] - left_shoulder['x'] + 1e-6)
            shoulder_angle = np.degrees(np.arctan(shoulder_slope))
            
            # 計算身體中心線偏移
            shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            hip_center_x = (left_hip['x'] + right_hip['x']) / 2
            body_offset = shoulder_center_x - hip_center_x
            
            # 向左傾斜（降低角度閾值和偏移閾值）
            if shoulder_angle > 7 or body_offset < -0.04:
                return "lean_left", 0.85
            
            # 向右傾斜（降低角度閾值和偏移閾值）
            if shoulder_angle < -7 or body_offset > 0.04:
                return "lean_right", 0.85
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_slouch(self, landmarks):
        """
        檢測是否駝背
        判斷依據：肩膀明顯低於正常位置，或背部彎曲
        """
        try:
            left_shoulder = landmarks[self.LEFT_SHOULDER]
            right_shoulder = landmarks[self.RIGHT_SHOULDER]
            left_hip = landmarks[self.LEFT_HIP]
            right_hip = landmarks[self.RIGHT_HIP]
            nose = landmarks[self.NOSE]
            
            # 計算肩膀與臀部的垂直距離
            shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hip_y = (left_hip['y'] + right_hip['y']) / 2
            torso_length = hip_y - shoulder_y
            
            # 計算鼻子與肩膀的垂直距離
            nose_shoulder_distance = shoulder_y - nose['y']
            
            # 如果軀幹過短（駝背壓縮）或頭部過度前傾（降低閾值提高靈敏度）
            if torso_length < 0.28 or nose_shoulder_distance < 0.12:
                return "slouch", 0.8
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _stabilize_posture(self, posture, confidence):
        """
        穩定姿勢偵測，減少镘3爍
        只有當新姿勢在歷史中出現超過一定次數時才確認
        
        參數:
            posture: 新偵測到的姿勢
            confidence: 信心度
            
        回傳:
            posture: 穩定後的姿勢
            confidence: 信心度
        """
        # 將新姿勢加入歷史
        self.posture_history.append(posture)
        
        # 保持歷史長度
        if len(self.posture_history) > self.history_size:
            self.posture_history.pop(0)
        
        # 如果歷史不足，直接返回當前姿勢
        if len(self.posture_history) < 2:
            self.current_posture = posture
            self.confidence = confidence
            return posture, confidence
        
        # 計算每個姿勢在歷史中出現的次數
        from collections import Counter
        posture_counts = Counter(self.posture_history)
        most_common_posture, count = posture_counts.most_common(1)[0]
        
        # 如果最常出現的姿勢出現至少2次，使用它（更快反應）
        if count >= 2:
            self.current_posture = most_common_posture
            self.confidence = confidence
            return most_common_posture, confidence
        else:
            # 否則保持上一次的姿勢
            return self.current_posture, self.confidence
    
    def is_focused(self, posture):
        """
        判斷當前姿勢是否屬於專注狀態
        
        參數:
            posture: 姿勢類別
            
        回傳:
            is_focused: 是否專注
        """
        focused_postures = ["good_posture"]
        return posture in focused_postures
    
    
