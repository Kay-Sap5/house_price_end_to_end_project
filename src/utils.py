import pickle
import os
from src.logger import logging
from src.exception import CustomExpection
import sys

def save_object(file_path , obj):
    try:
        with open(file_path ,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomExpection(e,sys)
    