import sys
import pandas as pd
import os

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading Model & Preprocessor")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 is_weekend: int,
                 is_peak_hour: int,
                 trip_distance_km: float,
                 passenger_count: int,
                 ride_type: str,
                 vehicle_type: str,
                 weather: str,
                 traffic_level: str,
                 surge_multiplier: float):

        self.is_weekend = is_weekend
        self.is_peak_hour = is_peak_hour
        self.trip_distance_km = trip_distance_km
        self.passenger_count = passenger_count
        self.ride_type = ride_type
        self.vehicle_type = vehicle_type
        self.weather = weather
        self.traffic_level = traffic_level
        self.surge_multiplier = surge_multiplier

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "is_weekend": [self.is_weekend],
                "is_peak_hour": [self.is_peak_hour],
                "trip_distance_km": [self.trip_distance_km],
                "passenger_count": [self.passenger_count],
                "ride_type": [self.ride_type],
                "vehicle_type": [self.vehicle_type],
                "weather": [self.weather],
                "traffic_level": [self.traffic_level],
                "surge_multiplier": [self.surge_multiplier],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(str(e), sys)