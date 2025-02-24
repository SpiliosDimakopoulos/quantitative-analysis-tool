import pandas as pd
import numpy as np

class DataCleaner:
    """Handles advanced cleaning and preprocessing of datasets."""

    @staticmethod
    def clean_data(df, missing_strategy='median', outlier_strategy='cap', normalization_method='minmax'):
        """
        Cleans the dataset by handling missing values, removing duplicates, 
        handling outliers, converting data types, and normalizing numerical columns.
        
        Parameters:
            df (pd.DataFrame): The dataset.
            missing_strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', or 'drop').
            outlier_strategy (str): Strategy for handling outliers ('cap' or 'remove').
            normalization_method (str): Method for normalizing numerical columns ('minmax' or 'zscore').

        Returns:
            pd.DataFrame: Cleaned dataset.
        """
        df = df.copy()
        df = DataCleaner.handle_missing_values(df, strategy=missing_strategy)
        df = df.drop_duplicates()
        df = DataCleaner.handle_outliers(df, strategy=outlier_strategy)
        df = DataCleaner.convert_data_types(df)
        df = DataCleaner.clean_string_columns(df)
        df = DataCleaner.normalize_numerical_columns(df, method=normalization_method)
        df = DataCleaner.validate_data(df)  # Add data validation step
        return df

    @staticmethod
    def handle_missing_values(df, strategy='median'):
        """Handle missing values using a specified strategy."""
        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            if df[col].dtype == 'object':  # Categorical data
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:  # Numerical data
                if strategy == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif strategy == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif strategy == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[col], inplace=True)
        return df

    @staticmethod
    def handle_outliers(df, strategy='cap'):
        """Detect and handle outliers using the IQR method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if strategy == 'cap':
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            elif strategy == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    @staticmethod
    def convert_data_types(df):
        """Convert columns to appropriate data types."""
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].nunique() / len(df) < 0.05:  # Low cardinality categorical
                df[col] = df[col].astype('category')
            elif df[col].dropna().astype(str).str.lower().isin(['true', 'false']).all():
                df[col] = df[col].astype('bool')
        return df

    @staticmethod
    def clean_string_columns(df):
        """Clean string columns (e.g., remove extra spaces, fix inconsistent casing)."""
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.title()
        return df

    @staticmethod
    def normalize_numerical_columns(df, method='minmax'):
        """Normalize numerical columns using Min-Max scaling or Z-score standardization."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if method == 'minmax':
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            elif method == 'zscore':
                df[col] = (df[col] - df[col].mean()) / df[col].std()
        return df

    @staticmethod
    def validate_data(df):
        """Validate numerical and categorical columns."""
        # Example: Ensure numerical columns like 'age' are within a reasonable range
        if 'age' in df.columns.str.lower():
            print("Validating Age Range:")
            print(df[(df['age'] < 0) | (df['age'] > 120)])

        # Check for inconsistent categorical values in 'gender' column (case-insensitive)
        if 'gender' in df.columns.str.lower():
            print("\nInconsistent Gender Values:")
            df['gender'] = df['gender'].str.lower()  # Convert all gender values to lowercase
            print(df['gender'].value_counts())
        
        return df

    @staticmethod
    def preview_data(df, num_rows=5):
        """Preview the first few rows of the dataset."""
        return df.head(num_rows)
