# data_collector.py
import cv2
import os
import numpy as np
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
import threading

class DataCollector:
    

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