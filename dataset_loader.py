import pandas as pd
import numpy as np
import requests
from io import StringIO
import chardet

class DataLoader:
    """Handles loading data from local files and URLs."""
    
    @staticmethod
    def load_from_url(url):
        response = requests.get(url)
        if response.status_code == 200:
            if url.endswith(".csv"):
                return pd.read_csv(StringIO(response.text))
            elif url.endswith(".json"):
                return pd.read_json(StringIO(response.text))
            else:
                print("Unsupported file format from URL.")
                return None
        else:
            print("Error fetching data from URL.")
            return None
    
    @staticmethod
    def load_from_file(filepath):
        try:
            # Detect the file encoding first
            with open(filepath, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                print(f"Detected encoding: {encoding}")

            # Load file with detected encoding
            if filepath.endswith(".csv"):
                return pd.read_csv(filepath, encoding=encoding)
            elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
                return pd.read_excel(filepath)
            elif filepath.endswith(".json"):
                return pd.read_json(filepath)
            else:
                print("Unsupported file format.")
                return None
        except Exception as e:
            print(f"Failed to load file: {e}")
            return None

class DataCategorizer:
    """Handles categorization of data."""
    
    CATEGORIES = {
        1: "Financial Analysis",
        2: "Business Analytics",
        3: "Scientific/Statistical Analysis"
    }
    
    @staticmethod
    def categorize(user_choice):
        return DataCategorizer.CATEGORIES.get(user_choice, "Unknown Category")
