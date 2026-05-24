from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            is_weekend=int(request.form.get('is_weekend')),
            is_peak_hour=int(request.form.get('is_peak_hour')),

            trip_distance_km=float(request.form.get('trip_distance_km')),   # ✔ FIX
            passenger_count=int(request.form.get('passenger_count')),

            ride_type=request.form.get('ride_type'),
            vehicle_type=request.form.get('vehicle_type'),
            weather=request.form.get('weather'),
            traffic_level=request.form.get('traffic_level'),

            surge_multiplier=float(request.form.get('surge_multiplier'))   # ✔ FIX
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        


