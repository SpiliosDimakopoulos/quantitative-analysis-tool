import pandas as pd
import matplotlib.pyplot as plt
from dataset_loader import DataLoader, DataCategorizer
from data_cleaner import DataCleaner
from machine_learning_analysis import MachineLearningAnalysis
from data_visualization import DataVisualization  # Import DataVisualization class
from matplotlib.backends.backend_pdf import PdfPages

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
            df = DataCleaner.clean_data(df)
            print("Data Cleaning Complete!")
            print(df.head())
            
            pdf_path = "report.pdf"  # Default report filename
            DataVisualization.create_report(df, pdf_path)
            
            '''
            # Prompt user for algorithm choices
            target_column = input("Enter the target column name: ")
            selected_algorithms = MachineLearningAnalysis.prompt_user_algorithm_choice()

            # Apply the selected algorithms
            results = MachineLearningAnalysis.apply_algorithm(df, selected_algorithms, target_column)

            print("\nModel Results:")
            for algo, result in results.items():
                print(f"Algorithm {algo}: {result}")

            # Visualize after analysis
            with PdfPages(pdf_path) as pdf:
                DataVisualization.visualize_after_analysis(results, pdf)  # Call to visualize after machine learning analysis
            '''
            print(f"Report saved to {pdf_path}") 
                
        else:
            print("Failed to load dataset.")

if __name__ == "__main__":
    processor = Main()
    processor.run()
