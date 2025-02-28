import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class ClassificationResultsVisualization:
    """Handles the visualization of classification results."""

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
            pdf.savefig()
            plt.close()


class RegressionResultsVisualization:
    """Handles the visualization of regression results."""

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
            pdf.savefig()
            plt.close()
