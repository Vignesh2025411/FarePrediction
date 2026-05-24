import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Configuration class for model training"""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Class to handle model training and evaluation"""
    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train multiple models and return the best performing one
        
        Args:
            train_array: Training features and target
            test_array: Testing features and target
            
        Returns:
            Dictionary with model performance metrics
        """
        try:
            logging.info("Split training and test input data")
            
            # Split arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Model training initiated")

            # Define models to train
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "SVM": SVR()
            }

            # Define hyperparameters for GridSearchCV
            params = {
                "Linear Regression": {},
                "Ridge": {
                    "alpha": [0.1, 1, 10, 100]
                },
                "Lasso": {
                    "alpha": [0.1, 1, 10, 100]
                },
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.5]
                },
                "SVM": {
                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1, 10]
                }
            }

            # Train and evaluate models
            logging.info("Evaluating models...")
            model_report, trained_models = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Log results
            logging.info("Model evaluation completed")
            logging.info(f"Model Report: {model_report}")

            # Find best model
            best_model_name = max(model_report, key=lambda x: model_report[x]["test"])
            best_model_score = model_report[best_model_name]["test"]
            best_model = trained_models[best_model_name]

            logging.info(f"Best model: {best_model_name} with test R² score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable performance", sys)

            # Save the best model
            logging.info("Saving best model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            

            return {
                "model_report": model_report,
                "best_model": best_model_name,
                "best_model_score": best_model_score,
                "model_path": self.model_trainer_config.trained_model_file_path
            }

        except Exception as e:
            raise CustomException(str(e), sys)