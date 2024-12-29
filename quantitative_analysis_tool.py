import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

class QuantitativeAnalysisTool:
    def __init__(self, data):
        """Initialize the Quantitative Analysis Tool."""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self.data = data
        self.scaled_data = None

    def preprocess_data(self):
        """Scale the data for consistent analysis."""
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data)
        print("Data successfully scaled.")

    def perform_pca(self, n_components=2):
        """Perform Principal Component Analysis (PCA)."""
        if self.scaled_data is None:
            raise RuntimeError("Data must be scaled before performing PCA. Call preprocess_data().")
        
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(self.scaled_data)
        explained_variance = pca.explained_variance_ratio_

        print(f"Explained variance by components: {explained_variance}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=components[:, 0], y=components[:, 1], alpha=0.7)
        plt.title("PCA Scatter Plot")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

    def correlation_analysis(self):
        """Perform correlation analysis and visualize it."""
        corr_matrix = self.data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Matrix")
        plt.show()

    def regression_analysis(self, x_col, y_col):
        """Perform linear regression between two columns."""
        x = self.data[x_col]
        y = self.data[y_col]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        print(f"Slope: {slope}, Intercept: {intercept}")
        print(f"R-squared: {r_value**2}")

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x, y=y, alpha=0.7)
        plt.plot(x, slope * x + intercept, color='red', label=f'Regression Line')
        plt.legend()
        plt.title(f"Regression Analysis: {x_col} vs {y_col}")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()

    def time_series_analysis(self, column, order=(1, 1, 0)):
        """Perform time series analysis using ARIMA."""
        ts_data = self.data[column].dropna()

        model = ARIMA(ts_data, order=order)
        model_fit = model.fit()

        print(model_fit.summary())

        plt.figure(figsize=(10, 6))
        plt.plot(ts_data, label='Original')
        plt.plot(model_fit.fittedvalues, color='red', label='Fitted')
        plt.legend()
        plt.title("Time Series Analysis")
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Load your data from a CSV file
    df = pd.read_csv('your_data.csv')  # Replace 'your_data.csv' with the path to your CSV file
    tool = QuantitativeAnalysisTool(df)

    # Perform analysis
    tool.preprocess_data()
    tool.perform_pca()  # Modify parameters as needed
    tool.correlation_analysis()
    tool.regression_analysis('column_x', 'column_y')  # Replace with your actual column names
    tool.time_series_analysis('time_series_column')  # Replace with your time series column name
