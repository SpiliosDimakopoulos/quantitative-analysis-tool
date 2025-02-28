#main.py

import pandas as pd
import matplotlib.pyplot as plt
from dataset_loader import DataLoader, DataCategorizer
from data_cleaner import DataCleaner
from machine_learning_analysis import MachineLearningAnalysis
from matplotlib.backends.backend_pdf import PdfPages
from statistical_analysis import StatisticsAnalysis
from report_generation import ReportGeneration

class Main:
    """Main class to handle user input, data processing, and machine learning analysis."""

    def run(self):
        print("Select a category:")
        for key, value in DataCategorizer.CATEGORIES.items():
            print(f"{key}: {value}")

        try:
            user_choice = int(input("Enter your choice (1-3): "))
        except ValueError:
            print("Invalid input. Defaulting to Unknown Category.")
            user_choice = 0

        category = DataCategorizer.categorize(user_choice)
        print(f"Selected Category: {category}")

        source = input("Enter the dataset file path or URL: ")
        if source.startswith("http"):
            df = DataLoader.load_from_url(source)
        else:
            df = DataLoader.load_from_file(source)

        if df is not None:
            print("Dataset Loaded Successfully.")
            print("Cleaning data...\n")

            # Use DataCleaner to clean the dataset
            df = DataCleaner.clean_data(df)
            print("Data Cleaning Complete!")

            # Generate report
            pdf_path = "report.pdf"  # Default report filename
            ReportGeneration.create_report(df)

            # Create an instance of StatisticsAnalysis with the loaded DataFrame
            stats_analysis = StatisticsAnalysis(df)

            # Call run_analysis with the selected category
            stats_analysis.run_analysis(user_choice)
            
        else:
            print("Failed to load dataset.") 

if __name__ == "__main__":
    processor = Main()
    processor.run()
