from flask import Flask , render_template , request
from src.pipeline.prediction_pipeline import CustomData , PredictPipleine
from src.logger import logging
import numpy as np

application = Flask(__name__)

@application.route("/")
def welcome():
    return render_template('form.html')

@application.route("/predictdata",methods = ['GET','POST'])
def predict_data_point():
    if request.method == "GET":
        return "<h2>This is the Get request page</h2>"
    else:
        data = CustomData(
            crime_rate=request.form.get('crime_rate'),
            resid_area=request.form.get("resid_area"),
            air_qual=request.form.get('air_qual'),
            room_num=request.form.get('room_num'),
            age = request.form.get('age'),
            teachers=request.form.get('teachers'),
            poor_prop=request.form.get('poor_prop'),
            airport=request.form.get('airport'),
            n_hos_beds=request.form.get('n_hos_beds'),
            n_hot_rooms=request.form.get('n_hot_rooms'),
            waterbody=request.form.get('waterbody'),
            rainfall=request.form.get('rainfall'),
            parks=request.form.get('parks') )
        
        data_frame = data.get_data_as_data_frame()
        logging.info(f"{data_frame}")
        pipe = PredictPipleine()
        y_pred = pipe.predict(data_frame)
        
        return render_template('form.html',result = np.round( y_pred[0],2))



if __name__ == "__main__":
    application.run(host='0.0.0.0')