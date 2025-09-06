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