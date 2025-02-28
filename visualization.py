#visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
from data_cleaner import DataCleaner

class CorrelationVisualization:
    """Handles the visualization of correlation heatmaps."""

    @staticmethod
    def plot_correlation_heatmap(df, pdf):
        """Plot correlation heatmap for the numerical features."""
        df_encoded = DataCleaner.encode_categorical(df)
        plt.figure(figsize=(12, 10))
        
        corr_matrix = df_encoded.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=0.5,
            mask=mask,
            cbar_kws={"shrink": 0.8}
        )

        plt.title("Correlation Heatmap (Before Analysis)")
        pdf.savefig()
        plt.close()


class NumericDistributionVisualization:
    """Handles the visualization of numeric feature distributions."""

    @staticmethod
    def plot_numeric_distributions(df, pdf):
        """Plot the distribution of numerical features."""
        df_encoded = DataCleaner.encode_categorical(df)
        numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
        
        num_columns = 3
        num_rows = (len(numeric_columns) // num_columns) + (len(numeric_columns) % num_columns > 0)
        
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(numeric_columns):
            sns.histplot(df_encoded[col], kde=True, ax=axes[i])
            wrapped_title = "\n".join(textwrap.wrap(f'Distribution of {col} (Before Analysis)', width=20))
            axes[i].set_title(wrapped_title, fontsize=10)
            axes[i].set_xlabel(col, fontsize=8)
            axes[i].set_ylabel('Frequency', fontsize=8)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=3.0)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
