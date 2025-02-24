import pandas as pd
import numpy as np
import scipy.stats as stats
from io import StringIO
import chardet
from sklearn.linear_model import LinearRegression

class StatisticsAnalysis:
    """Performs classical statistical analysis on a dataset for different categories."""

    def __init__(self, df):
        self.df = df

    def run_analysis(self, category):
        """Run analysis based on the selected category."""
        if category == 1:
            print("\nPerforming Financial Analysis:")
            self.financial_analysis()
        elif category == 2:
            print("\nPerforming Business Analytics Analysis:")
            self.business_analytics_analysis()
        elif category == 3:
            print("\nPerforming Scientific/Statistical Analysis:")
            self.scientific_statistical_analysis()
        else:
            print("Unknown category.")

        # Perform common analysis for all categories
        self.common_analysis()

    def common_analysis(self):
        """Common statistical analysis for all categories."""
        print("\nCommon Statistical Analysis:")

        # Descriptive statistics
        print("\nDescriptive Statistics (mean, std, min, max):")
        print(self.df.describe())

        # Correlation matrix for numerical features
        print("\nCorrelation Matrix:")
        print(self.df.corr())

        # Normality test for numerical columns
        print("\nNormality Test (Shapiro-Wilk):")
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stat, p_value = stats.shapiro(self.df[col].dropna())
            print(f"Column: {col} | Stat: {stat:.4f} | p-value: {p_value:.4f}")

    def financial_analysis(self):
        """Perform financial analysis on the dataset."""
        print("\nFinancial Analysis Metrics:")
        
        # Define possible financial columns
        possible_revenue_keywords = ['revenue', 'income', 'turnover', 'sales']
        possible_expenses_keywords = ['expenses', 'costs', 'spending']
        
        # Search for columns matching any of the financial keywords (case-insensitive)
        revenue_columns = [col for col in self.df.columns if any(keyword in col.lower() for keyword in possible_revenue_keywords)]
        expenses_columns = [col for col in self.df.columns if any(keyword in col.lower() for keyword in possible_expenses_keywords)]
        
        if revenue_columns and expenses_columns:
            # Use the first match from each list as the column names
            revenue_column = revenue_columns[0]
            expenses_column = expenses_columns[0]
            
            # Calculate profit margin
            self.df['profit_margin'] = (self.df[revenue_column] - self.df[expenses_column]) / self.df[revenue_column]
            
            # Show results
            print(self.df[[revenue_column, expenses_column, 'profit_margin']].head())
        else:
            print("Unable to find suitable columns for financial analysis.")

    def business_analytics_analysis(self):
        """Perform business analytics on the dataset."""
        print("\nBusiness Analytics Metrics:")
        
        # Define possible business-related columns
        possible_sales_keywords = ['sales', 'revenue', 'income', 'turnover']
        
        # Search for columns matching any of the business keywords (case-insensitive)
        sales_columns = [col for col in self.df.columns if any(keyword in col.lower() for keyword in possible_sales_keywords)]
        
        if sales_columns:
            # Use the first match as the column name
            sales_column = sales_columns[0]
            
            # Perform business analytics metrics
            print(f"Total {sales_column.capitalize()}: {self.df[sales_column].sum()}")
            print(f"Average {sales_column.capitalize()}: {self.df[sales_column].mean()}")
        else:
            print("Unable to find suitable columns for business analytics.")


    def scientific_statistical_analysis(self):
        """Perform scientific/statistical analysis on the dataset."""
        print("\nScientific/Statistical Analysis Metrics:")
        
        # Find numerical columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) >= 2:
            print("\nAvailable numerical columns for regression:")
            for i, col in enumerate(numeric_columns):
                print(f"{i}: {col}")
            
            # Ask the user to select two columns for the regression analysis
            try:
                col_x_index = int(input("\nSelect the index of the independent variable column: "))
                col_y_index = int(input("Select the index of the dependent variable column: "))
                
                if col_x_index != col_y_index:
                    X = self.df[[numeric_columns[col_x_index]]]
                    y = self.df[numeric_columns[col_y_index]]
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    
                    print(f"\nLinear Regression Results:")
                    print(f"Regression Coefficients: {model.coef_}")
                    print(f"Intercept: {model.intercept_}")
                else:
                    print("Error: Independent and dependent variables cannot be the same.")
            except (ValueError, IndexError):
                print("Invalid input, please select valid column indexes.")
        else:
            print("Not enough numerical columns for regression analysis.")
