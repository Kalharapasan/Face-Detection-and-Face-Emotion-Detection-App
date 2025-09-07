# 🎭 Emotion Detection System
<img width="1366" height="768" alt="Screenshot (166)" src="https://github.com/user-attachments/assets/d52f9451-8427-4ecd-9cf4-322440585713" />
<img width="1366" height="768" alt="Screenshot (167)" src="https://github.com/user-attachments/assets/6656d19f-940a-4d08-8436-26b922421e5f" />
<img width="1366" height="768" alt="Screenshot (168)" src="https://github.com/user-attachments/assets/3e31e797-94e4-481d-aa3d-2da6546b552a" />



A comprehensive emotion detection application with GUI interface, training data collection, and detailed analytics reporting.

## 📋 Features

### 🖥️ Main Application (`main.py`)
- **User-friendly GUI** with tabbed interface
- **Live emotion detection** from webcam
- **Real-time statistics** and emotion history
- **Session management** with save/load capabilities
- **Settings configuration** for camera and detection parameters

### 📹 Live Detection Tab
- Real-time face and emotion detection
- Confidence scoring
- Session statistics
- Recent emotions tracking
- Screenshot capture functionality

### 🎓 Training Data Collection Tab
- Manual data collection with webcam
- Automatic batch collection
- Progress tracking
- Multiple emotion categories
- Quality guidelines and tips

### 📊 Results & Analytics Tab
- Comprehensive result analysis
- Session comparison
- Confidence analysis
- Export capabilities

### ⚙️ Settings Tab
- Camera configuration
- Detection parameters
- Confidence thresholds
- Custom settings save/load

## 🛠️ Individual Modules

### `emotion_detector.py`
Core emotion detection engine that can run independently:
```bash
python emotion_detector.py
```

### `data_collector.py`
Standalone training data collection tool:
```bash
python data_collector.py Happy
python data_collector.py Sad
# ... other emotions
```

### `report_generator.py`
Advanced analytics and reporting:
```bash
python report_generator.py
```

## 📦 Installation

1. **Clone or download** all files to a directory
2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the main application:**
   ```bash
   python main.py
   ```

## 🚀 Quick Start

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Begin detection:**
   - Go to "Live Detection" tab
   - Click "Start Detection"
   - Position your face in the camera
   - Watch emotions being detected in real-time

3. **Collect training data:**
   - Go to "Train Data" tab
   - Select emotion type
   - Click "Start Data Collection"
   - Follow on-screen instructions

4. **View analytics:**
   - Go to "Results & Analytics" tab
   - Click "Generate Report"
   - Check the `reports/` folder for detailed analysis

## 📁 Directory Structure

```
emotion_detection_system/
├── main.py                 # Main GUI application
├── emotion_detector.py     # Core detection module
├── data_collector.py       # Training data collection
├── report_generator.py     # Analytics and reporting
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── settings.json          # App settings (auto-generated)
├── results/               # Detection results (auto-generated)
├── training_data/         # Collected training images
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── surprised/
│   ├── fear/
│   ├── disgust/
│   └── neutral/
└── reports/               # Generated analytics reports
```

## 🎯 Supported Emotions

- 😊 **Happy** - Smiles, joy, contentment
- 😢 **Sad** - Sadness, melancholy, sorrow
- 😠 **Angry** - Anger, frustration, irritation
- 😲 **Surprised** - Surprise, shock, amazement
- 😨 **Fear** - Fear, anxiety, worry
- 🤢 **Disgust** - Disgust, distaste, revulsion
- 😐 **Neutral** - Calm, composed, no strong emotion

## 🔧 Configuration

### Camera Settings
- **Camera Index:** Change camera source (0 for default)
- **Resolution:** Set video resolution
- **Confidence Threshold:** Adjust detection sensitivity

### Detection Parameters
- **Face Detection Scale:** Adjust face detection sensitivity
- **Minimum Face Size:** Set minimum detectable face size
- **History Smoothing:** Configure emotion smoothing over time

## 📈 Analytics Features

### Emotion Distribution
- Pie charts and bar graphs of emotion frequencies
- Percentage breakdown by emotion type
- Visual comparison across sessions

### Timeline Analysis
- Emotion patterns over time
- Hourly distribution analysis
- Session-based trends

### Confidence Analysis
- Detection confidence statistics
- Quality indicators
- Threshold analysis

### Training Data Reports
- Collection progress tracking
- Data completeness assessment
- Quality recommendations

## 🎮 Usage Examples

### Example 1: Basic Detection
```bash
# Start main application
python main.py

# Or run detector independently
python emotion_detector.py
```

### Example 2: Collect Training Data
```bash
# Collect happy emotion data
python data_collector.py Happy

# Collect multiple emotions
python data_collector.py Sad
python data_collector.py Angry
```

### Example 3: Generate Reports
```bash
# Generate comprehensive analytics
python report_generator.py
```

## 🔍 Troubleshooting

### Camera Issues
- **Camera not opening:** Check camera index in settings
- **Poor detection:** Ensure good lighting
- **Slow performance:** Lower resolution in settings

### Detection Issues
- **No emotions detected:** Check confidence threshold
- **Incorrect emotions:** Collect more training data
- **Unstable results:** Increase smoothing in settings

### Installation Issues
- **OpenCV errors:** Try `pip install opencv-python-headless`
- **Tkinter missing:** Install `python3-tk` (Linux) or update Python
- **Import errors:** Ensure all files are in the same directory

## 🛡️ Privacy & Security

- **Local Processing:** All detection runs locally on your machine
- **No Data Transmission:** No images or data sent to external servers
- **User Control:** Full control over data collection and storage
- **Optional Saving:** Choose what to save and when

## 🤝 Contributing

This system is designed to be modular and extensible:

1. **Add new emotions:** Extend the emotion categories
2. **Improve detection:** Enhance the classification algorithms
3. **Add features:** Implement new analysis methods
4. **Optimize performance:** Improve processing speed

## 📝 License

This project is provided as educational software. Feel free to modify and distribute according to your needs.

## 🔗 Dependencies

- **OpenCV:** Computer vision and image processing
- **NumPy:** Numerical computations
- **Matplotlib/Seaborn:** Data visualization
- **Pandas:** Data analysis and manipulation
- **Tkinter:** GUI interface (included with Python)
- **Pillow:** Image processing

## 💡 Tips for Best Results

### Detection Quality
- **Good Lighting:** Use well-lit environment
- **Clear View:** Position face clearly in camera
- **Stable Position:** Minimize head movement
- **Natural Expressions:** Use genuine facial expressions

### Training Data Quality
- **Variety:** Collect images with different lighting/angles
- **Quantity:** Aim for 100+ images per emotion
- **Quality:** Ensure clear, focused facial expressions
- **Consistency:** Maintain similar image quality across emotions

### Performance Optimization
- **Resolution:** Use appropriate camera resolution
- **Background:** Use simple, non-distracting backgrounds
- **Hardware:** Ensure adequate processing power
- **Settings:** Adjust detection parameters based on use case

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all requirements are installed
3. Ensure camera permissions are granted
4. Test individual modules separately

**Happy Emotion Detecting! 🎭**

