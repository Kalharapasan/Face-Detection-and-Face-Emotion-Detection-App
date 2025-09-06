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
    
    
    def start_collection(self):
        """Start data collection in a separate window"""
        self.collection_window = tk.Toplevel()
        self.collection_window.title(f"Data Collection - {self.emotion_label}")
        self.collection_window.geometry("800x600")
        self.collection_window.configure(bg='#2c3e50')
        
        # Instructions
        instructions = f"""
        🎭 Collecting Data for: {self.emotion_label.upper()}
        
        Instructions:
        1. Position your face in the camera view
        2. Show clear {self.emotion_label.lower()} expression
        3. Press SPACEBAR to capture images
        4. Press 'q' to quit
        5. Try different angles and lighting
        
        Target: 100+ images for good training
        """
        
        instruction_label = tk.Label(
            self.collection_window,
            text=instructions,
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1',
            justify='left'
        )
        instruction_label.pack(pady=10)
        
        # Status display
        self.status_var = tk.StringVar(value=f"Images collected: 0")
        status_label = tk.Label(
            self.collection_window,
            textvariable=self.status_var,
            font=('Arial', 14, 'bold'),
            bg='#2c3e50',
            fg='#27ae60'
        )
        status_label.pack(pady=5)
        
        # Control buttons
        button_frame = tk.Frame(self.collection_window, bg='#2c3e50')
        button_frame.pack(pady=10)
        
        start_btn = tk.Button(
            button_frame,
            text="🎥 Start Collection",
            command=self.start_camera_collection,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20
        )
        start_btn.pack(side='left', padx=10)
        
        stop_btn = tk.Button(
            button_frame,
            text="⏹️ Stop Collection",
            command=self.stop_camera_collection,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20
        )
        stop_btn.pack(side='left', padx=10)
        
        # Auto-collect button
        auto_btn = tk.Button(
            button_frame,
            text="🤖 Auto Collect (10 images)",
            command=self.auto_collect,
            bg='#3498db',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=20
        )
        auto_btn.pack(side='left', padx=10)
        
        # Progress info
        progress_text = tk.Text(
            self.collection_window,
            height=10,
            bg='#34495e',
            fg='#ecf0f1',
            font=('Consolas', 10)
        )
        progress_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.progress_text = progress_text
        self.update_progress_display()
    
    
    def start_camera_collection(self):
        """Start camera for manual collection"""
        self.collecting = True
        collection_thread = threading.Thread(target=self.camera_collection_loop)
        collection_thread.daemon = True
        collection_thread.start()
    
    def stop_camera_collection(self):
        """Stop camera collection"""
        self.collecting = False
    
    
    def camera_collection_loop(self):
        """Main camera collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        print(f"Starting data collection for {self.emotion_label}")
        print("Press SPACEBAR to capture, 'q' to quit")
        
        while self.collecting:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Add instruction text
                cv2.putText(frame, f"Show {self.emotion_label} expression", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add collection info
            cv2.putText(frame, f"Emotion: {self.emotion_label}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Images: {self.image_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "SPACEBAR: Capture | Q: Quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f'Data Collection - {self.emotion_label}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Spacebar to capture
                self.capture_face_data(frame, faces)
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def capture_face_data(self, frame, faces):
        """Capture face data from current frame"""
        if len(faces) == 0:
            print("No face detected!")
            return
        
        # Use the largest face
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        # Extract face region with some padding
        padding = 20
        y1 = max(0, y - padding)
        y2 = min(frame.shape[0], y + h + padding)
        x1 = max(0, x - padding)
        x2 = min(frame.shape[1], x + w + padding)
        
        face_img = frame[y1:y2, x1:x2]
        
        # Resize to standard size
        face_img = cv2.resize(face_img, (128, 128))
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.emotion_label.lower()}_{timestamp}_{self.image_count:04d}.jpg"
        filepath = os.path.join(self.emotion_dir, filename)
        
        cv2.imwrite(filepath, face_img)
        self.image_count += 1
        
        print(f"Captured image {self.image_count}: {filename}")
        
        # Update UI
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Images collected: {self.image_count}")
            self.update_progress_display()
    
    def auto_collect(self):
        """Automatically collect multiple images"""
        self.collecting = True
        auto_thread = threading.Thread(target=self.auto_collection_loop)
        auto_thread.daemon = True
        auto_thread.start()
    
    def auto_collection_loop(self):
        """Auto collection loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        target_images = 10
        collected = 0
        frame_skip = 15  # Collect every 15 frames
        frame_count = 0
        
        print(f"Auto-collecting {target_images} images for {self.emotion_label}")
        
        while self.collecting and collected < target_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(100, 100)
            )
            
            # Draw progress
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add collection info
            cv2.putText(frame, f"Auto-collecting: {collected}/{target_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, f"Show {self.emotion_label} expression", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Countdown
            if len(faces) > 0:
                countdown = frame_skip - (frame_count % frame_skip)
                cv2.putText(frame, f"Next capture in: {countdown}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.imshow(f'Auto Collection - {self.emotion_label}', frame)
            
            # Capture at intervals
            if len(faces) > 0 and frame_count % frame_skip == 0:
                self.capture_face_data(frame, faces)
                collected += 1
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.collecting = False
        
        messagebox.showinfo("Complete", f"Auto-collection finished! Collected {collected} images.")
    
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