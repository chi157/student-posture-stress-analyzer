"""
MediaPipe 關鍵點提取工具
提供人體姿態、手部、臉部關鍵點的提取功能
"""

import cv2
import mediapipe as mp
import numpy as np


class LandmarkExtractor:
    """MediaPipe 關鍵點提取器"""
    
    def __init__(self):
        """初始化 MediaPipe 模型"""
        # 初始化 MediaPipe Holistic（包含姿態、手部、臉部）
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 建立 Holistic 模型（提高追蹤信心值減少跳動）
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,   # 最小偵測信心值
            min_tracking_confidence=0.9,    # 提高追蹤信心值到 0.9
            model_complexity=1,             # 模型複雜度 (0, 1, 2)
            smooth_landmarks=True           # 啟用平滑處理
        )
    
    def process_frame(self, frame):
        """
        處理單一影格，提取關鍵點
        
        參數:
            frame: BGR 格式的影像
            
        回傳:
            results: MediaPipe 處理結果
        """
        # 轉換 BGR 到 RGB（MediaPipe 需要 RGB）
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 設定為不可寫入以提升效能
        image_rgb.flags.writeable = False
        
        # 進行偵測
        results = self.holistic.process(image_rgb)
        
        # 恢復可寫入狀態
        image_rgb.flags.writeable = True
        
        return results
    
    def draw_landmarks(self, frame, results):
        """
        在影像上繪製關鍵點
        
        參數:
            frame: BGR 格式的影像
            results: MediaPipe 處理結果
            
        回傳:
            frame: 繪製後的影像
        """
        # 繪製臉部關鍵點
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
        
        # 繪製姿態關鍵點
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_pose_landmarks_style()
            )
        
        # 繪製左手關鍵點
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_hand_landmarks_style()
            )
        
        # 繪製右手關鍵點
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles
                .get_default_hand_landmarks_style()
            )
        
        return frame
    
    def get_pose_landmarks(self, results):
        """
        取得姿態關鍵點座標
        
        參數:
            results: MediaPipe 處理結果
            
        回傳:
            landmarks: 關鍵點字典 {名稱: (x, y, z, visibility)}
        """
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return landmarks
    
    def get_hand_landmarks(self, results, hand='left'):
        """
        取得手部關鍵點座標
        
        參數:
            results: MediaPipe 處理結果
            hand: 'left' 或 'right'
            
        回傳:
            landmarks: 關鍵點字典 {名稱: (x, y, z)}
        """
        hand_landmarks = results.left_hand_landmarks if hand == 'left' else results.right_hand_landmarks
        
        if not hand_landmarks:
            return None
        
        landmarks = {}
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
        
        return landmarks
    
    def get_face_landmarks(self, results):
        """
        取得臉部關鍵點座標
        
        參數:
            results: MediaPipe 處理結果
            
        回傳:
            landmarks: 關鍵點字典 {名稱: (x, y, z)}
        """
        if not results.face_landmarks:
            return None
        
        landmarks = {}
        for idx, landmark in enumerate(results.face_landmarks.landmark):
            landmarks[idx] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            }
        
        return landmarks
    
    def calculate_angle(self, point1, point2, point3):
        """
        計算三點之間的角度
        
        參數:
            point1, point2, point3: 座標點 (x, y)
            
        回傳:
            angle: 角度（度數）
        """
        # 計算向量
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
        
        # 計算角度
        cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_distance(self, point1, point2):
        """
        計算兩點之間的歐式距離
        
        參數:
            point1, point2: 座標點 (x, y) 或 (x, y, z)
            
        回傳:
            distance: 距離
        """
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def close(self):
        """關閉 MediaPipe 模型"""
        self.holistic.close()
