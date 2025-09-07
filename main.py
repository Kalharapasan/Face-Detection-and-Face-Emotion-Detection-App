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
    
    def setup_settings_tab(self):
        """Setup settings tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Settings content
        settings_content = tk.Frame(settings_frame, bg='#ecf0f1')
        settings_content.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Camera settings
        camera_frame = tk.LabelFrame(
            settings_content,
            text="📷 Camera Settings",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1'
        )
        camera_frame.pack(fill='x', pady=10)
        
        tk.Label(camera_frame, text="Camera Index:", bg='#ecf0f1').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.camera_var = tk.StringVar(value="0")
        tk.Entry(camera_frame, textvariable=self.camera_var, width=10).grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(camera_frame, text="Resolution:", bg='#ecf0f1').grid(row=1, column=0, sticky='w', padx=10, pady=5)
        self.resolution_var = tk.StringVar(value="640x480")
        resolution_combo = ttk.Combobox(
            camera_frame, 
            textvariable=self.resolution_var,
            values=['640x480', '800x600', '1024x768', '1280x720'],
            state='readonly'
        )
        resolution_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Detection settings
        detection_frame = tk.LabelFrame(
            settings_content,
            text="🔍 Detection Settings",
            font=('Arial', 12, 'bold'),
            bg='#ecf0f1'
        )
        detection_frame.pack(fill='x', pady=10)
        
        tk.Label(detection_frame, text="Confidence Threshold:", bg='#ecf0f1').grid(row=0, column=0, sticky='w', padx=10, pady=5)
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        tk.Scale(
            detection_frame,
            from_=0.0,
            to=1.0,
            resolution=0.1,
            orient='horizontal',
            variable=self.confidence_threshold,
            bg='#ecf0f1'
        ).grid(row=0, column=1, padx=10, pady=5)
        
        # Save settings button
        tk.Button(
            settings_content,
            text="💾 Save Settings",
            command=self.save_settings,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20
        ).pack(pady=20)
    
    def start_detection(self):
        """Start live emotion detection"""
        try:
            camera_index = int(self.camera_var.get())
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.detection_running = True
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            
            # Start detection thread
            self.detection_thread = threading.Thread(target=self.detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            # Start timer
            self.session_start_time = datetime.now()
            self.update_session_time()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def stop_detection(self):
        """Stop live emotion detection"""
        self.detection_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.video_label.config(image='', text="Camera Feed Stopped")
    
    def detection_loop(self):
        """Main detection loop"""
        face_count = 0
        
        while self.detection_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            results = self.detector.process_frame(frame)
            
            if results:
                for result in results:
                    emotion = result['emotion']
                    confidence = result['confidence']
                    
                    # Update UI
                    self.root.after(0, self.update_detection_ui, emotion, confidence)
                    
                    # Save to history
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.emotion_history.append({
                        'timestamp': timestamp,
                        'emotion': emotion,
                        'confidence': confidence
                    })
                    
                    face_count += 1
            
            # Convert frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Update video display
            self.root.after(0, self.update_video_display, photo)
            
            # Update face count
            self.root.after(0, lambda: self.faces_detected_var.set(f"Faces Detected: {face_count}"))
    
    def update_detection_ui(self, emotion, confidence):
        """Update detection UI elements"""
        self.emotion_var.set(f"Current Emotion: {emotion}")
        self.confidence_var.set(f"Confidence: {confidence:.1%}")
        
        # Add to recent emotions list
        timestamp = datetime.now().strftime("%H:%M:%S")
        recent_entry = f"{timestamp} - {emotion} ({confidence:.1%})"
        self.recent_listbox.insert(0, recent_entry)
        
        # Keep only last 20 entries
        if self.recent_listbox.size() > 20:
            self.recent_listbox.delete(20, tk.END)
    
    def update_video_display(self, photo):
        """Update video display"""
        self.video_label.config(image=photo)
        self.video_label.image = photo  # Keep a reference
    
    def update_session_time(self):
        """Update session timer"""
        if self.detection_running and hasattr(self, 'session_start_time'):
            elapsed = datetime.now() - self.session_start_time
            minutes, seconds = divmod(elapsed.seconds, 60)
            time_str = f"Session Time: {minutes:02d}:{seconds:02d}"
            self.session_time_var.set(time_str)
            
            # Schedule next update
            self.root.after(1000, self.update_session_time)
    
    def save_current_result(self):
        """Save current detection results"""
        if not self.emotion_history:
            messagebox.showwarning("Warning", "No detection results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_results_{timestamp}.json"
        
        try:
            os.makedirs("results", exist_ok=True)
            filepath = os.path.join("results", filename)
            
            with open(filepath, 'w') as f:
                json.dump(self.emotion_history, f, indent=2)
            
            messagebox.showinfo("Success", f"Results saved to {filepath}")
            self.load_saved_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save results: {str(e)}")
    
    def start_data_collection(self):
        """Start data collection for training"""
        selected_emotion = self.emotion_combo.get()
        if not selected_emotion:
            messagebox.showwarning("Warning", "Please select an emotion first")
            return
        
        # Import and run data collection
        try:
            from data_collector import DataCollector
            collector = DataCollector(selected_emotion)
            collector.start_collection()
            self.update_training_overview()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start data collection: {str(e)}")
    

def main():
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()