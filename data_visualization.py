import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import textwrap

class DataVisualization:
    """Handles the visualization of the dataset before and after machine learning analysis."""

    @staticmethod
    def visualize_before_analysis(df, pdf):
        """Visualize the dataset before machine learning analysis."""
        print("\nVisualizing dataset before analysis...")
        
        # Dataset overview
        DataVisualization.show_dataset_overview(df, pdf)

        # Correlation heatmap
        DataVisualization.plot_correlation_heatmap(df, pdf)

        # Distributions of numeric features
        DataVisualization.plot_numeric_distributions(df, pdf)

    @staticmethod
    def visualize_after_analysis(model_results, pdf):
        """Visualize the model results after analysis."""
        print("\nVisualizing model performance after analysis...")

        # Classification results
        DataVisualization.plot_classification_results(model_results, pdf)

        # Regression results
        DataVisualization.plot_regression_results(model_results, pdf)

    @staticmethod
    def create_report(df, pdf_filename="report.pdf"):
        """Create a PDF report and save it as 'report.pdf'."""
        print(f"Generating report...")

        # Use the 'visualize' method to generate the PDF report without user input
        DataVisualization.visualize(df, None, pdf_filename)
        print(f"Report saved to {pdf_filename}")

    @staticmethod
    def visualize(df, model_results, pdf_filename="visualizations.pdf"):
        """Combine both before and after analysis visualizations into a single PDF."""
        with PdfPages(pdf_filename) as pdf:
            # Visualize before analysis
            DataVisualization.visualize_before_analysis(df, pdf)

            # Visualize after analysis (if model results are provided)
            if model_results is not None:
                DataVisualization.visualize_after_analysis(model_results, pdf)

        print(f"Visualizations saved to {pdf_filename}")

    @staticmethod
    def show_dataset_overview(df, pdf):
        """Show dataset overview including head and info."""
        print("\nDataset Head:")
        print(df.head())

        print("\nDataset Info:")
        print(df.info())

        print("\nNull Values in the Dataset:")
        print(df.isnull().sum())

    @staticmethod
    def plot_correlation_heatmap(df, pdf):
        """Plot correlation heatmap for the numerical features."""
        df_encoded = DataVisualization.encode_categorical(df)
        plt.figure(figsize=(12, 10))  # Increased figure size for better clarity
        
        # Calculate the correlation matrix
        corr_matrix = df_encoded.corr()

        # Generate a mask to hide the upper triangle (reduces redundancy)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Create the heatmap
        sns.heatmap(
            corr_matrix,
            annot=False,  # Set to True if you want numerical values
            cmap='coolwarm',  # Improved color palette
            fmt=".2f",
            linewidths=0.5,  # Reduce grid thickness for clarity
            mask=mask,  # Apply mask to hide the upper triangle
            cbar_kws={"shrink": 0.8}  # Shrink color bar for aesthetics
        )

        plt.title("Correlation Heatmap (Before Analysis)")
        pdf.savefig()  # Save the current figure to PDF
        plt.close()

    @staticmethod
    def plot_numeric_distributions(df, pdf):
        """Plot the distribution of numerical features in a grid of subplots with 3 histograms per row."""
        df_encoded = DataVisualization.encode_categorical(df)
        numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
        
        # Create a grid of subplots with 3 histograms per row
        num_columns = 3  # Number of columns for the subplot grid (3 histograms per row)
        num_rows = (len(numeric_columns) // num_columns) + (len(numeric_columns) % num_columns > 0)  # Calculate number of rows

        # Create subplots with specified figure size
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))  # Adjusted to fit 3 per row
        axes = axes.flatten()  # Ensure axes is a 1D array for easy iteration

        # Loop through numeric columns and plot each histogram
        for i, col in enumerate(numeric_columns):
            sns.histplot(df_encoded[col], kde=True, ax=axes[i])
            
            # Wrap title to prevent overlap and break it into two lines
            wrapped_title = "\n".join(textwrap.wrap(f'Distribution of {col} (Before Analysis)', width=20))
            axes[i].set_title(wrapped_title, fontsize=10)
            axes[i].set_xlabel(col, fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)

        # Remove any empty subplots if the number of numeric columns is less than grid size
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout to prevent overlap and add space between plots
        plt.tight_layout(pad=3.0)  # Increased padding between subplots to prevent overlap

        # Save the current figure to PDF
        pdf.savefig(fig, bbox_inches='tight')  # Save the current figure to PDF
        plt.close()

    @staticmethod
    def plot_classification_results(model_results, pdf):
        """Plot classification model results (accuracy)."""
        classification_results = {key: result for key, result in model_results.items() if 'Accuracy' in result}
        if classification_results:
            model_names = [f"Model {key}" for key in classification_results]
            accuracies = [float(result.split(": ")[1].strip('%')) for result in classification_results.values()]
            plt.figure(figsize=(10, 6))
            plt.barh(model_names, accuracies, color='royalblue')
            plt.xlabel('Accuracy (%)')
            plt.title('Classification Model Performance (After Analysis)')
            pdf.savefig()  # Save the current figure to PDF
            plt.close()

    @staticmethod
    def plot_regression_results(model_results, pdf):
        """Plot regression model results (Mean Squared Error)."""
        regression_results = {key: result for key, result in model_results.items() if 'MSE' in result}
        if regression_results:
            model_names = [f"Model {key}" for key in regression_results]
            mse_values = [float(result.split(": ")[1]) for result in regression_results.values()]
            plt.figure(figsize=(10, 6))
            plt.barh(model_names, mse_values, color='tomato')
            plt.xlabel('Mean Squared Error (MSE)')
            plt.title('Regression Model Performance (After Analysis)')
            pdf.savefig()  # Save the current figure to PDF
            plt.close()

    @staticmethod
    def encode_categorical(df):
        """Convert categorical columns to numerical codes."""
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        df_encoded = df.copy()
        for col in categorical_columns:
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
        return df_encoded
