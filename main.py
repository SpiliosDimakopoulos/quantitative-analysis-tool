import pandas as pd
from dataset_loader import DataLoader, DataCategorizer
from data_cleaner import DataCleaner

class Main:
    """Main class to handle user input and data processing."""
    
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
        else:
            print("Failed to load dataset.")

if __name__ == "__main__":
    processor = Main()
    processor.run()
