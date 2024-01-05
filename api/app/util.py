"""
    util
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    In this script, there is the function that calls the model to be used for making predictions, 
    and the function that converts the user request (JSON) into a dataframe format so that it can 
    be processed by the model.
"""
from joblib import load
from sklearn.pipeline import Pipeline
from pydantic import BaseModel
from pandas import DataFrame
import os
from io import BytesIO

import pickle

def get_model() -> Pipeline:
    # Variable que recibe de donde viene el modelo, por defecto usamos el model.pkl
    model_path = os.environ.get('MODEL_PATH', 'model/model_movies.pkl')
    with open(model_path, 'rb') as model_file:
        # Transformamos el model_path en bytes y lo cargamos
        model = load(BytesIO(model_file.read()))
    # with open(model_path, 'rb') as model_file:
    #     model = pickle.load(model_file)
    return model

def transform_to_dataframe(class_model: BaseModel) -> DataFrame:
    transition_dictionary = {key:[value] for key, value in class_model.model_dump().items()}
    data_frame = DataFrame(transition_dictionary)
    return data_frame