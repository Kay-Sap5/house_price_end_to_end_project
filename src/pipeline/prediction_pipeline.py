import os
import sys
from src.logger import logging
from src.exception import CustomExpection
from src.utils import load_object
import pandas as pd


class PredictPipleine:
    def __init__(self):
        pass

    def predict(self , features):
        try:
            model_path = os.path.join("artifacts",'model.pkl')
            preprocessor_path = os.path.join("artifacts",'preprocessor.pkl')

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            logging.info("Both Model and Preprocessor files are loaded.......")

            scaled_data = preprocessor.transform(features)
            y_pred = model.predict(scaled_data)
            logging.info("Prediction Done........")

            return y_pred
        except Exception as e:
            raise CustomExpection(e,sys)

class CustomData:

    def __init__(self, crime_rate, resid_area, air_qual, room_num, age,
       teachers, poor_prop, airport, n_hos_beds, n_hot_rooms,
       waterbody, rainfall, parks):
        
        self.crime_rate :float= crime_rate
        self.resid_area :float= resid_area
        self.air_qual :float = air_qual
        self.room_num :float= room_num
        self.age :float = age
        self.teachers :float = teachers
        self.poor_prop :float= poor_prop
        self.airport :str= airport
        self.n_hos_beds :float = n_hos_beds
        self.n_hot_rooms :float = n_hot_rooms
        self.waterbody :str = waterbody
        self.rainfall :int= rainfall
        self.parks :float = parks



    def get_data_as_data_frame(self):
        try:
            data_frame = pd.DataFrame([{
                'crime_rate':self.crime_rate , "resid_area":self.resid_area , "air_qual":self.air_qual, 'room_num':self.room_num,
                "age":self.age ,                "teachers":self.teachers ,     "poor_prop":self.poor_prop ,
                'airport':self.airport,          "n_hos_beds":self.n_hos_beds,  "n_hot_rooms":self.n_hot_rooms,
                'waterbody':self.waterbody,       "rainfall":self.rainfall ,     "parks":self.parks


            }])
            logging.info("DataFrame Created Successfully ......... ")
            return data_frame
        
        except Exception as e:
            raise CustomExpection(e,sys)

