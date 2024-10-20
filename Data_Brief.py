import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

def load_file(file_path):
    """
    Loads a file based on its extension (CSV, Excel, JSON).
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file format! Please provide a CSV, Excel, or JSON file.")
    
    return data

def data_overview(data):
    """
    Generates an overview of the data:
    - Shape
    - Data Types
    - Missing Values
    """
    print("Data Overview:")
    print(f"Number of rows: {data.shape[0]}")
    print(f"Number of columns: {data.shape[1]}")
    print("\nData Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum() / len(data) * 100)

def data_statistics(data):
    """
    Generates statistics on the numeric columns.
    """
    print("\nNumerical Column Statistics:")
    print(data.describe())

def data_correlation(data):
    """
    Generates a correlation heatmap for numerical columns.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()
    else:
        print("No numeric columns available for correlation analysis.")

def imbalance_detection(data):
    """
    Detects imbalance in classification target columns.
    """
    for col in data.select_dtypes(include=['object', 'category']):
        print(f"\nColumn: {col}")
        print(data[col].value_counts(normalize=True))

def generate_report(file_path):
    """
    Combines the overview, statistics, and correlation into a final report.
    """
    data = load_file(file_path)
    
    # Overview
    data_overview(data)
    
    # Numerical statistics
    data_statistics(data)
    
    # Correlation analysis
    data_correlation(data)
    
    # Imbalance detection
    imbalance_detection(data)
    
    print("\nReport Generated Successfully!")

if __name__ == "__main__":
    file_path = input("Enter the path of your file (CSV, Excel, or JSON): ")
    generate_report(file_path)
