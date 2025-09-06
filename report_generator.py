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