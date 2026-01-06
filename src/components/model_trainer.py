import os
import sys

from src.logger import logging
from src.exception import CustomExpection
from src.components.model_list import models , params
from src.utils import evaluate_model , save_object

from dataclasses import dataclass

@dataclass
class ModelTraniningConfig:
    model_path = os.path.join("artifacts","model.pkl")
    
class ModelTraning:
    def __init__(self):
        self.model_traning_config = ModelTraniningConfig()

    def initiate_model_training(self,train_arr , test_arr):
        try:
            x_train_arr = train_arr[:,:-1]
            y_train_arr = train_arr[:,-1]

            x_test_arr = test_arr[:,:-1]
            y_test_arr = test_arr[:,-1]
            logging.info("spliting the X and Y....")

            report = evaluate_model(x_train_arr , y_train_arr , x_test_arr , y_test_arr , models , params)
            max_r2 = max(list(report.values()))
            best_model = list(models.values())[list(report.values()).index(max_r2)]
            logging.info(f"Best Model [{best_model}] R2 Score [{max_r2}]")
            save_object(self.model_traning_config.model_path , best_model)
            logging.info("Model Saved Successfully ...................")
            
            return best_model
        
        except Exception as e:
            raise CustomExpection(e,sys)
    
        


        
        