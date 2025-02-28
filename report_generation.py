import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from visualization import CorrelationVisualization, NumericDistributionVisualization
from model_results import ClassificationResultsVisualization, RegressionResultsVisualization

class ReportGeneration:
    """Handles the creation of a PDF report."""

    @staticmethod
    def create_report(df, model_results=None, pdf_filename="report.pdf"):
        """Create a PDF report combining visualizations before and after analysis."""
        from data_overview import DatasetOverview  # Import here to avoid circular import

        print(f"Generating report...")

        with PdfPages(pdf_filename) as pdf:
            # Add dataset overview to PDF
            DatasetOverview.show_head(df, pdf)
            DatasetOverview.show_info(df, pdf)
            DatasetOverview.show_null_values(df, pdf)

            # Add visualizations to PDF
            CorrelationVisualization.plot_correlation_heatmap(df, pdf)
            NumericDistributionVisualization.plot_numeric_distributions(df, pdf)

            if model_results is not None:
                ClassificationResultsVisualization.plot_classification_results(model_results, pdf)
                RegressionResultsVisualization.plot_regression_results(model_results, pdf)

        print(f"Report saved to {pdf_filename}")
