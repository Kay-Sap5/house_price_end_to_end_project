import os
import pandas as pd
import sys

from src.logger import logging
from src.exception import CustomExpection

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str  = os.path.join("artifacts","test.csv")
    raw_data_path:str   = os.path.join("artifacts",'raw.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Entered initiate_data_ingestion method")

            df = pd.read_csv(r'C:/Users/M S I/Desktop/Houe_Price_End_to_End_project/notebook/data.csv')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            train_set , test_set = train_test_split(df , test_size=0.2 , random_state=32)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info("Saved DataSet as dataframes ")

            return (
                self.data_ingestion_config.train_data_path ,
                self.data_ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomExpection(e,sys)

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise CustomExpection(e,sys)

