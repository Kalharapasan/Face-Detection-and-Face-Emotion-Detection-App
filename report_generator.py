# report_generator.py
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from collections import Counter
import seaborn as sns

class ReportGenerator:
    def __init__(self):
        self.results_dir = "results"
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_all_results(self):
        """Load all result files"""
        all_results = []
        
        if not os.path.exists(self.results_dir):
            return all_results
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Add filename info
                    for item in data:
                        item['session'] = filename
                        item['session_date'] = filename.split('_')[2] if len(filename.split('_')) > 2 else 'unknown'
                    
                    all_results.extend(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return all_results
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analytics report"""
        results = self.load_all_results()
        
        if not results:
            print("No results found to generate report")
            return
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate different types of analysis
        self.generate_emotion_distribution(results, timestamp)
        self.generate_timeline_analysis(results, timestamp)
        self.generate_confidence_analysis(results, timestamp)
        self.generate_session_comparison(results, timestamp)
        self.generate_summary_report(results, timestamp)
        
        print(f"Comprehensive report generated in {self.reports_dir}/")
    
    def generate_emotion_distribution(self, results, timestamp):
        """Generate emotion distribution charts"""
        emotions = [r['emotion'] for r in results]
        emotion_counts = Counter(emotions)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_counts)))
        ax1.pie(emotion_counts.values(), labels=emotion_counts.keys(), autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax1.set_title('Emotion Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(emotion_counts.keys(), emotion_counts.values(), color=colors)
        ax2.set_title('Emotion Frequency', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Emotions')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.reports_dir}/emotion_distribution_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_timeline_analysis(self, results, timestamp):
        """Generate timeline analysis"""
        # Convert timestamps and group by time periods
        timeline_data = {}
        for result in results:
            time_str = result.get('timestamp', '00:00:00')
            try:
                # Extract hour from timestamp
                hour = int(time_str.split(':')[0])
                if hour not in timeline_data:
                    timeline_data[hour] = []
                timeline_data[hour].append(result['emotion'])
            except:
                continue
        
        if not timeline_data:
            return
        
        # Create hourly emotion distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        
        emotions = ['Happy', 'Sad', 'Angry', 'Surprised', 'Fear', 'Disgust', 'Neutral']
        hours = sorted(timeline_data.keys())
        
        emotion_by_hour = {}
        for emotion in emotions:
            emotion_by_hour[emotion] = []
            for hour in hours:
                count = timeline_data[hour].count(emotion)
                emotion_by_hour[emotion].append(count)
        
        # Create stacked bar chart
        bottom = np.zeros(len(hours))
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
        
        for i, emotion in enumerate(emotions):
            ax.bar(hours, emotion_by_hour[emotion], bottom=bottom, 
                  label=emotion, color=colors[i])
            bottom += emotion_by_hour[emotion]
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Emotion Count')
        ax.set_title('Emotion Distribution by Time of Day', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{self.reports_dir}/timeline_analysis_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_confidence_analysis(self, results, timestamp):
        """Generate confidence analysis charts"""
        confidences = [float(r['confidence']) for r in results if 'confidence' in r]
        emotions = [r['emotion'] for r in results if 'confidence' in r]
        
        if not confidences:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall confidence distribution
        ax1.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Confidence Distribution')
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.2f}')
        ax1.legend()
        
        # Confidence by emotion (box plot)
        emotion_confidence = {}
        for emotion, confidence in zip(emotions, confidences):
            if emotion not in emotion_confidence:
                emotion_confidence[emotion] = []
            emotion_confidence[emotion].append(confidence)
        
        emotions_list = list(emotion_confidence.keys())
        confidence_lists = [emotion_confidence[emotion] for emotion in emotions_list]
        
        ax2.boxplot(confidence_lists, labels=emotions_list)
        ax2.set_xlabel('Emotions')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Distribution by Emotion')
        ax2.tick_params(axis='x', rotation=45)
        
        # Average confidence by emotion
        avg_confidences = [np.mean(emotion_confidence[emotion]) for emotion in emotions_list]
        bars = ax3.bar(emotions_list, avg_confidences, color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Emotions')
        ax3.set_ylabel('Average Confidence')
        ax3.set_title('Average Confidence by Emotion')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, conf in zip(bars, avg_confidences):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{conf:.2f}', ha='center', va='bottom')
        
        # Confidence threshold analysis
        thresholds = np.arange(0.1, 1.0, 0.1)
        detection_counts = []
        for threshold in thresholds:
            count = sum(1 for conf in confidences if conf >= threshold)
            detection_counts.append(count)
        
        ax4.plot(thresholds, detection_counts, marker='o', linewidth=2, markersize=6)
        ax4.set_xlabel('Confidence Threshold')
        ax4.set_ylabel('Number of Detections')
        ax4.set_title('Detections vs Confidence Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.reports_dir}/confidence_analysis_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_session_comparison(self, results, timestamp):
        """Generate session comparison analysis"""
        session_data = {}
        for result in results:
            session = result.get('session', 'unknown')
            if session not in session_data:
                session_data[session] = []
            session_data[session].append(result)
        
        if len(session_data) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Session duration comparison (approximate)
        session_lengths = []
        session_names = []
        for session, data in session_data.items():
            session_lengths.append(len(data))
            session_names.append(session[:15] + '...' if len(session) > 15 else session)
        
        bars = ax1.bar(range(len(session_names)), session_lengths, color='lightcoral')
        ax1.set_xlabel('Sessions')
        ax1.set_ylabel('Number of Detections')
        ax1.set_title('Detections per Session')
        ax1.set_xticks(range(len(session_names)))
        ax1.set_xticklabels(session_names, rotation=45)
        
        # Add value labels
        for bar, length in zip(bars, session_lengths):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{length}', ha='center', va='bottom')
        
        # Dominant emotion per session
        session_emotions = {}
        for session, data in session_data.items():
            emotions = [d['emotion'] for d in data]
            most_common = Counter(emotions).most_common(1)[0][0]
            session_emotions[session] = most_common
        
        # Count dominant emotions
        dominant_counts = Counter(session_emotions.values())
        ax2.pie(dominant_counts.values(), labels=dominant_counts.keys(), autopct='%1.1f%%',
                startangle=90)
        ax2.set_title('Dominant Emotions Across Sessions')
        
        plt.tight_layout()
        plt.savefig(f"{self.reports_dir}/session_comparison_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self, results, timestamp):
        """Generate text summary report"""
        total_detections = len(results)
        emotion_counts = Counter([r['emotion'] for r in results])
        confidences = [float(r['confidence']) for r in results if 'confidence' in r]
        
        # Calculate statistics
        most_common_emotion = emotion_counts.most_common(1)[0] if emotion_counts else ('None', 0)
        avg_confidence = np.mean(confidences) if confidences else 0
        high_confidence_count = sum(1 for c in confidences if c >= 0.7)
        
        # Get unique sessions
        sessions = set([r.get('session', 'unknown') for r in results])
        
        # Generate report text
        report_text = f"""
EMOTION DETECTION ANALYSIS REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
==================================================

SUMMARY STATISTICS:
------------------
• Total Detections: {total_detections}
• Unique Sessions: {len(sessions)}
• Most Common Emotion: {most_common_emotion[0]} ({most_common_emotion[1]} occurrences)
• Average Confidence: {avg_confidence:.2%}
• High Confidence Detections (≥70%): {high_confidence_count} ({high_confidence_count/len(confidences)*100:.1f}%)

EMOTION BREAKDOWN:
-----------------
"""
        
        for emotion, count in emotion_counts.most_common():
            percentage = (count / total_detections) * 100
            report_text += f"• {emotion}: {count} detections ({percentage:.1f}%)\n"
        
        report_text += f"""

CONFIDENCE ANALYSIS:
------------------
• Minimum Confidence: {min(confidences):.2%}
• Maximum Confidence: {max(confidences):.2%}
• Standard Deviation: {np.std(confidences):.2%}

QUALITY INDICATORS:
------------------
• High Quality Detections (≥80%): {sum(1 for c in confidences if c >= 0.8)}
• Medium Quality Detections (50-80%): {sum(1 for c in confidences if 0.5 <= c < 0.8)}
• Low Quality Detections (<50%): {sum(1 for c in confidences if c < 0.5)}

RECOMMENDATIONS:
---------------
"""
        
        # Add recommendations based on analysis
        if avg_confidence < 0.6:
            report_text += "• Consider improving lighting conditions for better detection accuracy\n"
        if emotion_counts.get('Unknown', 0) > total_detections * 0.1:
            report_text += "• High number of 'Unknown' emotions detected - consider recalibrating detection parameters\n"
        if len(emotion_counts) < 3:
            report_text += "• Limited emotional range detected - try varying expressions during detection\n"
        if high_confidence_count / len(confidences) > 0.8:
            report_text += "• Excellent detection quality! System is performing well\n"
        
        report_text += f"""

TECHNICAL DETAILS:
-----------------
• Report Generated: {timestamp}
• Data Sources: {len(sessions)} session file(s)
• Analysis Period: Various sessions
• Chart Files Generated: 4 visualization files

END OF REPORT
==================================================
        """
        
        # Save text report
        with open(f"{self.reports_dir}/summary_report_{timestamp}.txt", 'w') as f:
            f.write(report_text)
        
        print("Summary report saved as text file")
    
    def generate_training_data_report(self):
        """Generate report on training data collection"""
        training_dir = "training_data"
        if not os.path.exists(training_dir):
            print("No training data found")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Analyze training data
        emotions = ['happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'neutral']
        training_stats = {}
        
        for emotion in emotions:
            emotion_dir = os.path.join(training_dir, emotion)
            if os.path.exists(emotion_dir):
                images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))]
                training_stats[emotion] = len(images)
            else:
                training_stats[emotion] = 0
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart of training data
        emotions_cap = [e.capitalize() for e in emotions]
        counts = [training_stats[e] for e in emotions]
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
        
        bars = ax1.bar(emotions_cap, counts, color=colors)
        ax1.set_title('Training Data Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotions')
        ax1.set_ylabel('Number of Images')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}', ha='center', va='bottom')
        
        # Training completeness pie chart
        target = 100
        complete_emotions = sum(1 for count in counts if count >= target)
        incomplete_emotions = len(emotions) - complete_emotions
        
        ax2.pie([complete_emotions, incomplete_emotions], 
               labels=['Complete (≥100 images)', 'Incomplete (<100 images)'],
               autopct='%1.1f%%', startangle=90,
               colors=['lightgreen', 'lightcoral'])
        ax2.set_title('Training Data Completeness')
        
        plt.tight_layout()
        plt.savefig(f"{self.reports_dir}/training_data_report_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        total_images = sum(counts)
        report_text = f"""
TRAINING DATA COLLECTION REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
============================================

OVERVIEW:
---------
• Total Training Images: {total_images}
• Target per Emotion: 100 images
• Emotions with Sufficient Data: {complete_emotions}/{len(emotions)}
• Collection Completeness: {(complete_emotions/len(emotions))*100:.1f}%

DETAILED BREAKDOWN:
------------------
"""
        
        for emotion, count in zip(emotions, counts):
            status = "✓ Complete" if count >= target else "⚠ Incomplete"
            percentage = (count / target) * 100 if target > 0 else 0
            report_text += f"• {emotion.capitalize()}: {count} images ({percentage:.1f}% of target) - {status}\n"
        
        report_text += """

RECOMMENDATIONS:
---------------
"""
        
        for emotion, count in zip(emotions, counts):
            if count < target:
                needed = target - count
                report_text += f"• Collect {needed} more images for {emotion.capitalize()}\n"
        
        if complete_emotions == len(emotions):
            report_text += "• All emotions have sufficient training data! Ready for model training.\n"
        
        report_text += """

QUALITY GUIDELINES:
------------------
• Ensure images have clear facial expressions
• Include variety in lighting conditions
• Capture different angles and distances
• Remove blurry or poorly lit images
• Maintain consistent image quality across emotions

END OF TRAINING REPORT
============================================
"""
        
        with open(f"{self.reports_dir}/training_data_report_{timestamp}.txt", 'w') as f:
            f.write(report_text)
        
        print("Training data report generated")

def main():
    """Generate reports independently"""
    generator = ReportGenerator()
    
    print("Generating comprehensive emotion detection report...")
    generator.generate_comprehensive_report()
    
    print("Generating training data report...")
    generator.generate_training_data_report()
    
    print("All reports generated successfully!")

if __name__ == "__main__":
    main()