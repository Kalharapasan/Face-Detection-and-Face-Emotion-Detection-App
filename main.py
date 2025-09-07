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
    
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main container
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_container, 
            text="🎭 Emotion Detection System", 
            font=('Arial', 24, 'bold'),
            fg='#ecf0f1', 
            bg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Live Detection
        self.setup_detection_tab()
        
        # Tab 2: Train Data
        self.setup_training_tab()
        
        # Tab 3: Results & Analytics
        self.setup_results_tab()
        
        # Tab 4: Settings
        self.setup_settings_tab()
    
    
    


def main():
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()