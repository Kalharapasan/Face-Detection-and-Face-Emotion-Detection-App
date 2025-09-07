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
    
    def detect_facial_features(self, face_gray):
        """Detect various facial features"""
        features = {}
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            face_gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(10, 10)
        )
        features['eyes'] = len(eyes)
        features['eye_positions'] = eyes
        
        # Detect smile
        smiles = self.smile_cascade.detectMultiScale(
            face_gray, 
            scaleFactor=1.8, 
            minNeighbors=15, 
            minSize=(25, 25)
        )
        features['smiles'] = len(smiles)
        features['smile_positions'] = smiles
        
        return features
    
    def analyze_face_geometry(self, face_gray):
        """Analyze face geometry for emotion clues"""
        height, width = face_gray.shape
        
        # Divide face into regions
        upper_face = face_gray[0:height//3, :]  # Forehead and eyes
        middle_face = face_gray[height//3:2*height//3, :]  # Nose and cheeks
        lower_face = face_gray[2*height//3:height, :]  # Mouth and chin
        
        # Calculate brightness in each region
        upper_brightness = np.mean(upper_face)
        middle_brightness = np.mean(middle_face)
        lower_brightness = np.mean(lower_face)
        
        # Calculate contrast and texture
        overall_contrast = np.std(face_gray)
        
        # Calculate additional metrics
        brightness_variance = np.var([upper_brightness, middle_brightness, lower_brightness])
        
        return {
            'upper_brightness': upper_brightness,
            'middle_brightness': middle_brightness,
            'lower_brightness': lower_brightness,
            'overall_contrast': overall_contrast,
            'brightness_ratio': lower_brightness / upper_brightness if upper_brightness > 0 else 1,
            'brightness_variance': brightness_variance
        }
    
    def classify_emotion(self, features, geometry):
        """Classify emotion based on features and geometry"""
        
        # Happy: Smile detected, normal eye opening
        if features['smiles'] > 0:
            confidence = min(0.95, 0.7 + features['smiles'] * 0.1)
            return "Happy", confidence
        
        # Surprised: Wide eyes (more than 2 detected, high contrast)
        if features['eyes'] > 2 and geometry['overall_contrast'] > 45:
            return "Surprised", 0.80
        
        # Sad: Lower face darker, no smile, low contrast
        if (geometry['brightness_ratio'] < 0.90 and 
            features['smiles'] == 0 and 
            geometry['lower_brightness'] < 100):
            return "Sad", 0.75
        
        # Angry: Low overall brightness, no smile, high contrast
        if (geometry['overall_contrast'] > 40 and 
            features['smiles'] == 0 and 
            geometry['upper_brightness'] < 90):
            return "Angry", 0.70
        
        # Fear: High brightness variance, wide eyes
        if (geometry['brightness_variance'] > 200 and 
            features['eyes'] >= 2 and 
            geometry['overall_contrast'] > 35):
            return "Fear", 0.65
        
        # Disgust: Specific brightness pattern
        if (geometry['middle_brightness'] < geometry['upper_brightness'] and 
            geometry['middle_brightness'] < geometry['lower_brightness'] and
            features['smiles'] == 0):
            return "Disgust", 0.60
        
        # Neutral/Calm: Balanced features
        if features['eyes'] >= 2 and features['smiles'] == 0:
            if geometry['overall_contrast'] < 35:
                return "Calm", 0.65
            else:
                return "Neutral", 0.60
        
        # Serious: Low brightness, no smile
        if (geometry['upper_brightness'] < 95 and 
            features['smiles'] == 0):
            return "Serious", 0.55
        
        # Default
        return "Unknown", 0.30
