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
            return posture, confidence
        
        posture, confidence = self._check_look_down(pose_landmarks)
        if posture:
            return posture, confidence
        
        posture, confidence = self._check_look_left_right(pose_landmarks)
        if posture:
            return posture, confidence
        
        posture, confidence = self._check_lean(pose_landmarks)
        if posture:
            return posture, confidence
        
        posture, confidence = self._check_slouch(pose_landmarks)
        if posture:
            return posture, confidence
        
        # 預設為良好坐姿
        return "good_posture", 0.8
    
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
            
            # 如果平均深度大於閾值，判定為遠離螢幕
            if avg_z > 0.5:
                return "far_from_screen", 0.9
            
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
            
            # 如果鼻子明顯低於眼睛中心，或接近肩膀，判定為低頭
            if nose['y'] > eye_center_y + 0.05 or (nose['y'] - shoulder_center_y) < 0.15:
                return "look_down", 0.85
            
            return None, 0.0
        except:
            return None, 0.0
    
    def _check_look_left_right(self, landmarks):
        """
        檢測是否向左或向右看
        判斷依據：左右眼睛與鼻子的相對位置
        """
        try:
            nose = landmarks[self.NOSE]
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_ear = landmarks[self.LEFT_EAR]
            right_ear = landmarks[self.RIGHT_EAR]
            
            # 計算左右眼睛的可見度
            left_eye_vis = landmarks[self.LEFT_EYE]['visibility']
            right_eye_vis = landmarks[self.RIGHT_EYE]['visibility']
            
            # 計算臉部中心線偏移
            eye_center_x = (left_eye['x'] + right_eye['x']) / 2
            face_offset = nose['x'] - eye_center_x
            
            # 向左看：右眼可見度高，左眼可見度低，或鼻子偏右
            if right_eye_vis > left_eye_vis + 0.2 or face_offset > 0.03:
                return "look_left", 0.8
            
            # 向右看：左眼可見度高，右眼可見度低，或鼻子偏左
            if left_eye_vis > right_eye_vis + 0.2 or face_offset < -0.03:
                return "look_right", 0.8
            
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
            
            # 向左傾斜
            if shoulder_angle > 10 or body_offset < -0.05:
                return "lean_left", 0.85
            
            # 向右傾斜
            if shoulder_angle < -10 or body_offset > 0.05:
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
            
            # 如果軀幹過短（駝背壓縮）或頭部過度前傾
            if torso_length < 0.25 or nose_shoulder_distance < 0.1:
                return "slouch", 0.8
            
            return None, 0.0
        except:
            return None, 0.0
    
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
    
    def get_posture_label(self, posture):
        """
        取得姿勢的中文標籤
        
        參數:
            posture: 姿勢類別
            
        回傳:
            label: 中文標籤
        """
        labels = {
            "good_posture": "良好坐姿",
            "slouch": "駝背",
            "look_left": "向左看",
            "look_right": "向右看",
            "look_down": "低頭",
            "lean_left": "向左傾斜",
            "lean_right": "向右傾斜",
            "far_from_screen": "遠離螢幕",
            "no_person": "未偵測到人",
            "unknown": "未知"
        }
        return labels.get(posture, posture)
