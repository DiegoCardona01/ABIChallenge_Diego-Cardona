"""
    train
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains auxiliary functions used for saving the model and generating a 
    report of metrics and charts.
"""
from sklearn.pipeline import Pipeline
import logging
from joblib import dump
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dvc import api
from io import StringIO

def update_model(model: Pipeline) -> None:
    dump(model, 'model/model_movies.pkl')

class DataLoader:

    """
    Class for loading data from DVC.

    Parameters:
    - file_path (str): The path to the file to be loaded.

    Attributes:
    - file_path (str): The path to the specified file.
    - logger (Logger): Logging instance for recording events and messages.

    Methods:
    - load_data() -> pd.DataFrame or None:
      Attempts to load data from the specified file.
      Returns the loaded data if successful, or None in case of errors.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            data = api.read(self.file_path, remote='dataset-tracker', encoding='utf-8')
            data = pd.read_csv(StringIO(data))
            self.logger.info(f"Data loaded successfully from {self.file_path}")
            return data
        except FileNotFoundError:
            self.logger.error(f"Error: File '{self.file_path}' not found.")
            return None
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            return None

class ModelEvaluator:
    """
    Utility class for evaluating and saving simple metrics of a machine learning model.

    Methods:
    - save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
      Saves model pipeline information and evaluation scores to a text file.

    - get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) -> None:
      Generates and saves a regression plot of predicted vs. real values for model performance on the test set.
    """
    def __init__(self):
        """Initializes a ModelEvaluator instance."""
        pass

    @staticmethod
    def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
        with open('report.txt', 'w') as report_file:
            report_file.write('# Model Pipeline Description'+'\n')

            for key, value in model.named_steps.items():
                report_file.write(f'### {key}:{value.__repr__()}'+'\n')
            report_file.write(f'## Train Score: {train_score}'+'\n')
            report_file.write(f'## Test Score: {test_score}'+'\n')
            report_file.write(f'## Validation Score: {validation_score}'+'\n')

    @staticmethod
    def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) -> None:
        fig, ax = plt.subplots()
        fig.set_figheight(8)
        fig.set_figwidth(8)

        sns.regplot(x=y_pred, y=y_real, ax=ax)

        ax.set_xlabel('Predicted worldwide gross')
        ax.set_ylabel('Real worldwide gross')
        ax.set_title('Behavior of model prediction')
        fig.savefig('prediction_behavior.png')