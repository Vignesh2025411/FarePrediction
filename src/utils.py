import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
#from sklearn.metrics import r2_score
from src.exception import CustomException
#from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """Save object to file using pickle"""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(str(e), sys)