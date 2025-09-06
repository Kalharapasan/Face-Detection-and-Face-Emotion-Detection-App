# data_collector.py
import cv2
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import threading

class DataCollector:
    
    def __init__(self, emotion_label):
        self.emotion_label = emotion_label
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.collecting = False
        self.image_count = 0
        self.session_count = 0
        
        # Create directories
        self.base_dir = "training_data"
        self.emotion_dir = os.path.join(self.base_dir, emotion_label.lower())
        os.makedirs(self.emotion_dir, exist_ok=True)
    
    def auto_collect(self):
        """Automatically collect multiple images"""
        self.collecting = True
        auto_thread = threading.Thread(target=self.auto_collection_loop)
        auto_thread.daemon = True
        auto_thread.start()
    
    def update_progress_display(self):
        """Update progress display"""
        if hasattr(self, 'progress_text'):
            # Get current stats
            total_images = len([f for f in os.listdir(self.emotion_dir) if f.endswith('.jpg')])
            
            progress_info = f"📊 Collection Progress for {self.emotion_label}\n"
            progress_info += "=" * 50 + "\n"
            progress_info += f"Total images collected: {total_images}\n"
            progress_info += f"Images this session: {self.image_count}\n"
            progress_info += f"Target: 100+ images\n"
            progress_info += f"Progress: {min(100, (total_images/100)*100):.1f}%\n\n"
            
            # Tips
            progress_info += "💡 Collection Tips:\n"
            progress_info += "• Vary your facial expressions\n"
            progress_info += "• Change lighting conditions\n"
            progress_info += "• Try different angles\n"
            progress_info += "• Ensure clear, focused images\n"
            progress_info += "• Keep expressions natural\n\n"
            
            # Recent files
            recent_files = sorted([f for f in os.listdir(self.emotion_dir) if f.endswith('.jpg')])[-10:]
            if recent_files:
                progress_info += "📁 Recent captures:\n"
                for file in recent_files:
                    progress_info += f"  • {file}\n"
            
            self.progress_text.delete(1.0, tk.END)
            self.progress_text.insert(1.0, progress_info)
    


def main():
    """Test data collector independently"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python data_collector.py <emotion>")
        print("Available emotions: Happy, Sad, Angry, Surprised, Fear, Disgust, Neutral")
        return
    
    emotion = sys.argv[1]
    collector = DataCollector(emotion)
    
    # Simple command-line interface
    cap = cv2.VideoCapture(0)
    image_count = 0
    
    print(f"Starting data collection for {emotion}")
    print("Press SPACEBAR to capture, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = collector.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Emotion: {emotion} | Count: {image_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and len(faces) > 0:
            collector.capture_face_data(frame, faces)
            image_count += 1
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()