# data_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error

class DataVisualization:
    """Handles the visualization of the dataset before and after machine learning analysis."""
    
    @staticmethod
    def visualize_before_analysis(df):
        """Visualize the dataset before analysis."""
        print("Visualizing dataset before analysis...")
        
        # Overview of the dataset
        print("\nDataset Head:")
        print(df.head())
        
        print("\nDataset Info:")
        print(df.info())
        
        # Checking for null values
        print("\nNull Values in the Dataset:")
        print(df.isnull().sum())
        
        # Visualize the distribution of the target variable (assuming it's categorical)
        if df.select_dtypes(include=['object']).shape[1] > 0:  # Categorical columns
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df, x=df.select_dtypes(include=['object']).columns[0])
            plt.title(f'Distribution of {df.select_dtypes(include=["object"]).columns[0]}')
            plt.show()
        
        # Visualize correlations
        plt.figure(figsize=(10, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

        # Visualize distributions of numeric features
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

    @staticmethod
    def visualize_after_analysis(model_results):
        """Visualize the model results after analysis."""
        print("\nVisualizing model performance after analysis...")
        
        # Create a bar plot to visualize accuracy (for classification models)
        classification_results = {key: result for key, result in model_results.items() if 'Accuracy' in result}
        regression_results = {key: result for key, result in model_results.items() if 'MSE' in result}

        if classification_results:
            model_names = [f"Model {key}" for key in classification_results]
            accuracies = [float(result.split(": ")[1].strip('%')) for result in classification_results.values()]

            plt.figure(figsize=(10, 6))
            plt.barh(model_names, accuracies, color='royalblue')
            plt.xlabel('Accuracy (%)')
            plt.title('Classification Model Performance')
            plt.show()

        if regression_results:
            model_names = [f"Model {key}" for key in regression_results]
            mse_values = [float(result.split(": ")[1]) for result in regression_results.values()]

            plt.figure(figsize=(10, 6))
            plt.barh(model_names, mse_values, color='tomato')
            plt.xlabel('Mean Squared Error (MSE)')
            plt.title('Regression Model Performance')
            plt.show()

        print("\nModel performance visualization generated.")

