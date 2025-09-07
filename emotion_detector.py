# emotion_detector.py
import cv2
import numpy as np


class EmotionDetector:
    
     def __init__(self):
        # Load all necessary cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Emotion history for smoothing
        self.emotion_history = []
        self.history_size = 5
        
        # Emotion colors for display
        self.emotion_colors = {
            'Happy': (0, 255, 0),
            'Sad': (255, 0, 0),
            'Angry': (0, 0, 255),
            'Surprised': (0, 255, 255),
            'Fear': (255, 0, 255),
            'Disgust': (128, 0, 128),
            'Neutral': (255, 255, 255),
            'Calm': (200, 200, 200),
            'Serious': (128, 128, 128),
            'Unknown': (64, 64, 64)
        }
    
    
    
