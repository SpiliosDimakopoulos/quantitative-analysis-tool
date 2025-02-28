#data_overview.py

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import io

class DatasetOverview:
    """Handles dataset exploration and provides an overview."""

    @staticmethod
    def show_head(df, pdf):
        """Show the first few rows of the dataset."""
        print("\nDataset Head:")
        head = df.head()
        head_str = head.to_string()
        DatasetOverview.add_text_to_pdf(head_str, pdf)

    @staticmethod
    def show_info(df, pdf):
        """Show the dataset info (data types, memory usage)."""
        print("\nDataset Info:")
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()  # Get string from StringIO buffer
        DatasetOverview.add_text_to_pdf(info_str, pdf)

    @staticmethod
    def show_null_values(df, pdf):
        """Show the count of missing values in the dataset."""
        print("\nNull Values in the Dataset:")
        null_values_str = df.isnull().sum().to_string()
        DatasetOverview.add_text_to_pdf(null_values_str, pdf)

    @staticmethod
    def add_text_to_pdf(text, pdf):
        """Adds text to the PDF."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0, 1, text, wrap=True, ha='left', va='top', fontsize=10)
        ax.axis('off')  # Turn off axis
        pdf.savefig(fig)
        plt.close(fig)
