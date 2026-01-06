import pickle
import os
from src.logger import logging
from src.exception import CustomExpection
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def save_object(file_path , obj):
    try:
        with open(file_path ,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomExpection(e,sys)
    

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        reports = {}
        logging.info("Enter Traning_model method at Utils.py")

        for i in range(len(list(models.keys()))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params[list(models.keys())[i]]
            grid = GridSearchCV(model , param , n_jobs=-1 , cv=3)
            grid.fit(x_train , y_train)

            model.set_params(**grid.best_params_)
            model.fit(x_train , y_train)

            y_pred = model.predict(x_test)

            r_square = r2_score(y_test , y_pred)

            reports[model_name] = r_square

        return reports

    except Exception as e:
        raise CustomExpection(e,sys)