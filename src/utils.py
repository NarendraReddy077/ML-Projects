import os
import sys
import dill

import pandas as pd
import numpy as np
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    """
    Saves an object to a file using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates the model using R2 score.
    """
    try:
        model_report = {}

        for model_name, model in models.items():

            # Hyperparameter tuning
            para = param.get(model_name, {})
            
            grid_search = GridSearchCV(estimator=model, param_grid=para, cv=3)
            grid_search.fit(X_train, y_train)
            # model = grid_search.best_estimator_
            # print(f"Best parameters for {model_name}: {grid_search.best_params_}")

            model.set_params(**grid_search.best_params_)                
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = round(r2_score(y_test, y_test_pred), 3)
            model_report[model_name] = test_model_score
        
        return model_report
    except Exception as e:
        CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)