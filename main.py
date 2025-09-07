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
    
    def setup_detection_tab(self):
        """Setup live detection tab"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="📹 Live Detection")
        
        # Left panel for video
        left_panel = tk.Frame(detection_frame, bg='#34495e')
        left_panel.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        
        # Video display
        self.video_label = tk.Label(left_panel, text="Camera Feed", bg='#34495e', fg='white')
        self.video_label.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(left_panel, bg='#34495e')
        button_frame.pack(pady=10)
        
        self.start_btn = tk.Button(
            button_frame, 
            text="▶️ Start Detection", 
            command=self.start_detection,
            bg='#27ae60', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            padx=20
        )
        self.start_btn.pack(side='left', padx=5)
        
        self.stop_btn = tk.Button(
            button_frame, 
            text="⏹️ Stop Detection", 
            command=self.stop_detection,
            bg='#e74c3c', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            padx=20,
            state='disabled'
        )
        self.stop_btn.pack(side='left', padx=5)
        
        self.save_btn = tk.Button(
            button_frame, 
            text="💾 Save Result", 
            command=self.save_current_result,
            bg='#3498db', 
            fg='white', 
            font=('Arial', 12, 'bold'),
            padx=20
        )
        self.save_btn.pack(side='left', padx=5)
        
        # Right panel for information
        right_panel = tk.Frame(detection_frame, bg='#2c3e50', width=300)
        right_panel.pack(side='right', fill='y', padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        # Current emotion display
        info_label = tk.Label(
            right_panel, 
            text="📊 Detection Info", 
            font=('Arial', 16, 'bold'),
            fg='#ecf0f1', 
            bg='#2c3e50'
        )
        info_label.pack(pady=10)
        
        # Emotion display
        self.emotion_var = tk.StringVar(value="No emotion detected")
        emotion_label = tk.Label(
            right_panel, 
            textvariable=self.emotion_var,
            font=('Arial', 14),
            fg='#e74c3c', 
            bg='#2c3e50',
            wraplength=250
        )
        emotion_label.pack(pady=5)
        
        # Confidence display
        self.confidence_var = tk.StringVar(value="Confidence: 0%")
        confidence_label = tk.Label(
            right_panel, 
            textvariable=self.confidence_var,
            font=('Arial', 12),
            fg='#f39c12', 
            bg='#2c3e50'
        )
        confidence_label.pack(pady=5)
        
        # Statistics
        stats_frame = tk.LabelFrame(
            right_panel, 
            text="📈 Session Statistics", 
            fg='#ecf0f1', 
            bg='#2c3e50',
            font=('Arial', 12, 'bold')
        )
        stats_frame.pack(fill='x', pady=10, padx=10)
        
        self.faces_detected_var = tk.StringVar(value="Faces Detected: 0")
        tk.Label(stats_frame, textvariable=self.faces_detected_var, fg='#ecf0f1', bg='#2c3e50').pack(anchor='w')
        
        self.session_time_var = tk.StringVar(value="Session Time: 00:00")
        tk.Label(stats_frame, textvariable=self.session_time_var, fg='#ecf0f1', bg='#2c3e50').pack(anchor='w')
        
        # Recent emotions list
        recent_frame = tk.LabelFrame(
            right_panel, 
            text="🕐 Recent Emotions", 
            fg='#ecf0f1', 
            bg='#2c3e50',
            font=('Arial', 12, 'bold')
        )
        recent_frame.pack(fill='both', expand=True, pady=10, padx=10)
        
        self.recent_listbox = tk.Listbox(
            recent_frame, 
            height=8,
            bg='#34495e', 
            fg='#ecf0f1',
            selectbackground='#3498db'
        )
        self.recent_listbox.pack(fill='both', expand=True)
    
    def setup_training_tab(self):
        """Setup training/data collection tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🎓 Train Data")
        
        # Training instructions
        instruction_text = """
        📚 Training Data Collection
        
        This tab allows you to collect and manage training data for emotion detection.
        
        Steps:
        1. Select emotion category
        2. Start data collection
        3. Capture facial expressions
        4. Review and manage collected data
        """
        
        instruction_label = tk.Label(
            training_frame,
            text=instruction_text,
            font=('Arial', 12),
            justify='left',
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        instruction_label.pack(fill='x', padx=20, pady=10)
        
        # Training controls
        controls_frame = tk.Frame(training_frame, bg='#34495e')
        controls_frame.pack(fill='x', padx=20, pady=10)
        
        # Emotion selection
        tk.Label(
            controls_frame, 
            text="Select Emotion:", 
            font=('Arial', 12, 'bold'),
            bg='#34495e', 
            fg='white'
        ).pack(side='left', padx=10)
        
        self.emotion_combo = ttk.Combobox(
            controls_frame,
            values=['Happy', 'Sad', 'Angry', 'Surprised', 'Fear', 'Disgust', 'Neutral'],
            state='readonly'
        )
        self.emotion_combo.pack(side='left', padx=10)
        self.emotion_combo.set('Happy')
        
        # Training buttons
        tk.Button(
            controls_frame,
            text="🎯 Start Data Collection",
            command=self.start_data_collection,
            bg='#27ae60',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='left', padx=10)
        
        tk.Button(
            controls_frame,
            text="📁 Open Training Data Folder",
            command=self.open_training_folder,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='left', padx=10)
        
        # Training data preview
        preview_frame = tk.LabelFrame(
            training_frame,
            text="📂 Training Data Overview",
            font=('Arial', 12, 'bold')
        )
        preview_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.training_text = tk.Text(
            preview_frame,
            height=15,
            bg='#2c3e50',
            fg='#ecf0f1',
            font=('Consolas', 10)
        )
        self.training_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.update_training_overview()
    
    def setup_results_tab(self):
        """Setup results and analytics tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Results & Analytics")
        
        # Results controls
        controls_frame = tk.Frame(results_frame, bg='#34495e')
        controls_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            controls_frame,
            text="🔄 Refresh Results",
            command=self.load_saved_results,
            bg='#3498db',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='left', padx=10)
        
        tk.Button(
            controls_frame,
            text="📈 Generate Report",
            command=self.generate_report,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='left', padx=10)
        
        tk.Button(
            controls_frame,
            text="🗑️ Clear All Results",
            command=self.clear_results,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side='left', padx=10)
        
        # Results display
        self.results_text = tk.Text(
            results_frame,
            bg='#2c3e50',
            fg='#ecf0f1',
            font=('Consolas', 10)
        )
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
    

def main():
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()