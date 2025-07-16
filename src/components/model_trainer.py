import sys
import os
from dataclasses import dataclass

from sklearn.ensemble import (RandomForestRegressor, 
                              GradientBoostingRegressor, 
                              AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and testing arrays into features and target")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor()
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 score above 0.6", sys)

            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_model_score}")

            save_object(
                file_path=self.trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, predicted)
            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)