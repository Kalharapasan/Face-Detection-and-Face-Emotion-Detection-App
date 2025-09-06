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