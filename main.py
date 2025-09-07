#main.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import os
from PIL import Image, ImageTk
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.cap = None
        self.detection_running = False
        self.current_frame = None
        self.emotion_history = []
        self.detection_results = []
        
        # Load emotion detector
        from emotion_detector import EmotionDetector
        self.detector = EmotionDetector()
        
        self.setup_ui()
        self.load_saved_results()


def main():
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()