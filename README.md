# Quantitative Analysis Tool

## Overview
The Quantitative Analysis Tool is a Python application designed for performing advanced data analysis tasks, including:

1. **Data Preprocessing** - Scales input data for consistent analysis.
2. **Principal Component Analysis (PCA)** - Reduces dimensionality and visualizes key components.
3. **Correlation Analysis** - Visualizes the correlation matrix to identify relationships between variables.
4. **Regression Analysis** - Performs linear regression and visualizes results.
5. **Time Series Analysis** - Models time series data using ARIMA.

## Features
- Modular and scalable design.
- Integration with pandas, NumPy, Matplotlib, Seaborn, and SciPy for robust data analysis and visualization.
- ARIMA modeling for forecasting and time series evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SpiliosDimakopoulos/quantitative-analysis-tool.git
   cd quantitative-analysis-tool
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your data in a CSV file, e.g., `data.csv`.
2. Update the `data.csv` placeholder in the `if __name__ == "__main__":` section of the code with your file's path.
3. Specify the relevant column names for analysis in methods like `regression_analysis` and `time_series_analysis`.
4. Run the script:
   ```bash
   python quantitative_analysis_tool.py
   ```

## Example

Here is an example of how to load your data and perform analysis:

```python
# Example usage:
if __name__ == "__main__":
    # Load your data from a CSV file
    df = pd.read_csv('data.csv')
    tool = QuantitativeAnalysisTool(df)

    # Perform analysis
    tool.preprocess_data()
    tool.perform_pca()
    tool.correlation_analysis()
    tool.regression_analysis('column_x', 'column_y')
    tool.time_series_analysis('time_series_column')
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributions
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/quantitative-analysis-tool/issues).

