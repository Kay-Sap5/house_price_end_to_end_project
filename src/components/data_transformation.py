from src.logger import logging
from src.exception import CustomExpection
from src.utils import save_object

import numpy as np
import pandas as pd
import os
import sys

from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        

    def get_data_transform(self,train,test):
        try:
            logging.info("Entered get_data_transform....")
            num_features = list(train.select_dtypes(exclude="O").columns) #removing the target feature       
            
            cat_features = list(test.select_dtypes(include='O').columns) 
            
            num_pipeline = Pipeline(steps=[
                ("simpleimputer",SimpleImputer(strategy='mean')),
                ("minmaxscaler",MinMaxScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("simpleimputer",SimpleImputer(strategy='most_frequent',)),
                ("Ohe",OneHotEncoder())
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_features),
                ('cat_pipeline',cat_pipeline,cat_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomExpection(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df  = pd.read_csv(test_path)
            logging.info("Read the train and test data")
            

            target_feature = 'price'

            x_train_df = self.train_df.drop(columns = [target_feature])
            y_train_df = self.train_df[target_feature]

            
            x_test_df = self.test_df.drop(columns = [target_feature])
            y_test_df = self.test_df[target_feature]
            

            preprocessor =self.get_data_transform(train=x_train_df , test=x_test_df)

            x_train_arr = preprocessor.fit_transform(x_train_df)
            x_test_arr  = preprocessor.transform(x_test_df)
            logging.info(f"Scaling the dataframe Completed......[train.shape({x_train_arr.shape}) , test_shape({x_test_arr.shape})]")
            train_arr = np.c_[x_train_arr,y_train_df]
            test_arr  = np.c_[x_test_arr,y_test_df]
            logging.info(f"Concating the dataframe Completed......[train.shape({train_arr.shape}) , test_shape({test_arr.shape})]")

            save_object(self.data_transformation_config.preprocessor_path , preprocessor)
            
            return (
                train_arr,test_arr
            )
        except Exception as e:
            logging.info(f"Error {e}")
            raise CustomExpection(e,sys)





            
