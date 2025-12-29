"""
學生坐姿與壓力即時偵測系統
使用 Webcam 進行即時姿勢與壓力行為偵測
"""

import cv2
import time
import sys
from pathlib import Path

# 加入 utils 模組路徑
sys.path.append(str(Path(__file__).parent))

from utils.landmark_utils import LandmarkExtractor
from utils.posture_detector import PostureDetector
from utils.stress_detector import StressDetector


class RealtimeDetectionSystem:
    """即時偵測系統主類別"""
    
    def __init__(self):
        """初始化系統元件"""
        print("🔧 初始化系統...")
        
        # 初始化各個偵測器
        self.landmark_extractor = LandmarkExtractor()
        self.posture_detector = PostureDetector()
        self.stress_detector = StressDetector()
        
        # 專注度統計
        self.continuous_focus_time = 0.0  # 連續專注秒數
        self.total_focus_time = 0.0       # 累積專注秒數
        self.stress_event_count = 0       # 壓力事件次數
        
        # 時間記錄
        self.last_time = time.time()
        self.last_focused = False
        
        # 視窗設定
        self.window_name = "學生坐姿與壓力偵測系統"
        
        print("✅ 系統初始化完成！")
    
    def run(self):
        """執行即時偵測"""
        # 開啟 Webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ 無法開啟 Webcam！")
            return
        
        print("📹 Webcam 已開啟")
        print("💡 按 'q' 或 'ESC' 離開")
        print("-" * 50)
        
        # 設定視窗
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # 讀取影格
                ret, frame = cap.read()
                if not ret:
                    print("❌ 無法讀取影像")
                    break
                
                # 翻轉影像（鏡像效果）
                frame = cv2.flip(frame, 1)
                
                # 處理影格
                frame = self._process_frame(frame)
                
                # 顯示影像
                cv2.imshow(self.window_name, frame)
                
                # 檢查按鍵
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    break
        
        finally:
            # 清理資源
            cap.release()
            cv2.destroyAllWindows()
            self.landmark_extractor.close()
            
            # 顯示最終統計
            self._print_final_stats()
    
    def _process_frame(self, frame):
        """
        處理單一影格
        
        參數:
            frame: 影像
            
        回傳:
            frame: 處理後的影像
        """
        # 提取關鍵點
        results = self.landmark_extractor.process_frame(frame)
        
        # 取得關鍵點資料
        pose_landmarks = self.landmark_extractor.get_pose_landmarks(results)
        left_hand = self.landmark_extractor.get_hand_landmarks(results, 'left')
        right_hand = self.landmark_extractor.get_hand_landmarks(results, 'right')
        face_landmarks = self.landmark_extractor.get_face_landmarks(results)
        
        # 偵測姿勢
        posture, posture_conf = self.posture_detector.detect_posture(pose_landmarks)
        
        # 偵測壓力行為
        stress_behavior, stress_conf = self.stress_detector.detect_stress_behavior(
            pose_landmarks, left_hand, right_hand, face_landmarks
        )
        
        # 更新統計資料
        self._update_statistics(posture, stress_behavior)
        
        # 繪製關鍵點
        frame = self.landmark_extractor.draw_landmarks(frame, results)
        
        # 繪製資訊介面
        frame = self._draw_hud(frame, posture, posture_conf, stress_behavior, stress_conf)
        
        return frame
    
    def _update_statistics(self, posture, stress_behavior):
        """
        更新統計資料
        
        參數:
            posture: 當前姿勢
            stress_behavior: 當前壓力行為
        """
        # 計算時間差
        current_time = time.time()
        delta_time = current_time - self.last_time
        self.last_time = current_time
        
        # 判斷是否專注
        is_focused = self.posture_detector.is_focused(posture)
        
        if is_focused:
            # 專注狀態：累加時間
            self.continuous_focus_time += delta_time
            self.total_focus_time += delta_time
            self.last_focused = True
        else:
            # 分心狀態：重置連續專注時間
            if self.last_focused:
                self.continuous_focus_time = 0.0
                self.last_focused = False
        
        # 檢查壓力行為是否改變（計算事件次數）
        if self.stress_detector.is_behavior_changed(stress_behavior):
            self.stress_event_count += 1
    
    def _draw_hud(self, frame, posture, posture_conf, stress_behavior, stress_conf):
        """
        繪製抬頭顯示器（HUD）
        
        參數:
            frame: 影像
            posture: 姿勢類別
            posture_conf: 姿勢信心度
            stress_behavior: 壓力行為類別
            stress_conf: 壓力行為信心度
            
        回傳:
            frame: 繪製後的影像
        """
        h, w = frame.shape[:2]
        
        # 設定字體
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 判斷專注狀態
        is_focused = self.posture_detector.is_focused(posture)
        focus_status = "FOCUSED 🎯" if is_focused else "DISTRACTED ⚠️"
        focus_color = (0, 255, 0) if is_focused else (0, 165, 255)
        
        # === 左上角：姿勢資訊 ===
        y_offset = 40
        
        # 姿勢標籤
        posture_label = self.posture_detector.get_posture_label(posture)
        cv2.putText(frame, f"Posture: {posture_label}", (20, y_offset), 
                    font, 0.7, (255, 255, 255), 2)
        y_offset += 35
        
        # 信心度
        cv2.putText(frame, f"Confidence: {posture_conf:.2f}", (20, y_offset), 
                    font, 0.6, (200, 200, 200), 2)
        y_offset += 50
        
        # 壓力行為標籤
        behavior_label = self.stress_detector.get_behavior_label(stress_behavior)
        stress_color = (0, 255, 255) if stress_behavior != "neutral" else (200, 200, 200)
        cv2.putText(frame, f"Stress: {behavior_label}", (20, y_offset), 
                    font, 0.7, stress_color, 2)
        y_offset += 35
        
        # 壓力行為信心度
        cv2.putText(frame, f"Confidence: {stress_conf:.2f}", (20, y_offset), 
                    font, 0.6, (200, 200, 200), 2)
        
        # === 右上角：專注狀態 ===
        focus_text = focus_status
        text_size = cv2.getTextSize(focus_text, font, 1.0, 2)[0]
        cv2.putText(frame, focus_text, (w - text_size[0] - 20, 50), 
                    font, 1.0, focus_color, 3)
        
        # === 右下角：統計資訊 ===
        stats_x = w - 350
        stats_y = h - 130
        
        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (stats_x - 10, stats_y - 40), 
                     (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # 統計文字
        cv2.putText(frame, "=== Statistics ===", (stats_x, stats_y), 
                    font, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Continuous focus: {int(self.continuous_focus_time)} s", 
                    (stats_x, stats_y + 30), font, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Total focus: {int(self.total_focus_time)} s", 
                    (stats_x, stats_y + 60), font, 0.6, (0, 200, 255), 2)
        
        cv2.putText(frame, f"Stress events: {self.stress_event_count}", 
                    (stats_x, stats_y + 90), font, 0.6, (0, 165, 255), 2)
        
        # === 底部：提示訊息 ===
        tip_text = "Press 'Q' or 'ESC' to exit"
        tip_size = cv2.getTextSize(tip_text, font, 0.5, 1)[0]
        cv2.putText(frame, tip_text, ((w - tip_size[0]) // 2, h - 15), 
                    font, 0.5, (180, 180, 180), 1)
        
        return frame
    
    def _print_final_stats(self):
        """顯示最終統計資訊"""
        print("\n" + "=" * 50)
        print("📊 最終統計")
        print("=" * 50)
        print(f"累積專注時間: {int(self.total_focus_time)} 秒")
        print(f"壓力事件次數: {self.stress_event_count} 次")
        print("=" * 50)
        print("👋 系統已關閉")


def main():
    """主程式進入點"""
    print("=" * 50)
    print("🎓 學生坐姿與壓力偵測系統")
    print("=" * 50)
    
    # 建立並執行系統
    system = RealtimeDetectionSystem()
    system.run()


if __name__ == "__main__":
    main()
