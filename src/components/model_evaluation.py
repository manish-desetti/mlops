import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.utils import load_object
from src.logger.logging import logging
from src.exception.exception import customexception

class ModelEvaluation:
    def __init__(self):
        logging.info("Evaluation started")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))  # RMSE
        mae = mean_absolute_error(actual, pred)  # MAE
        r2 = r2_score(actual, pred)  # R2 value
        logging.info("Evaluation metrics captured")
        return rmse, mae, r2

    def initiate_model_evaluation(self, train_array, test_array):
        try:
            X_test, y_test = (test_array[:, :-1], test_array[:, -1])

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            # Explicitly set the tracking URI to the local file store
            mlflow.set_tracking_uri("file:///C:/Users/desettma/Desktop/mlops/mlruns")

            tracking_uri = mlflow.get_tracking_uri()
            logging.info(f"Tracking URI: {tracking_uri}")

            tracking_url_type_store = urlparse(tracking_uri).scheme
            logging.info(f"Tracking URL type store: {tracking_url_type_store}")

            with mlflow.start_run():
                prediction = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")

        except Exception as e:
            raise customexception(e, sys)

# Example usage
if __name__ == "__main__":
    try:
        # Replace these with your actual data arrays
        train_array = np.array([[1, 2, 3], [4, 5, 6]])  # Dummy training data
        test_array = np.array([[1, 2, 3], [4, 5, 6]])  # Dummy test data

        evaluator = ModelEvaluation()
        evaluator.initiate_model_evaluation(train_array, test_array)
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
