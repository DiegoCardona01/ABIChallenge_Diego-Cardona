"""
    views
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains the function that performs the prediction on the data entered by the user.
"""
from .models import PredictionRequest
from .util import get_model, transform_to_dataframe
from google.cloud import storage
import pandas as pd
import os

model = get_model()
client = storage.Client()

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    prediction = model.predict(data_to_predict)[0] # dado que es un array queremos su valor [0]
    # damos un max(0, prediction) porque no es bueno darle a un usuario final
    # una predicción cruda, en este caso si la predicción nos da negativa ponemos un 0
    # así no confundimos al usuario final en caso de predicciones negativas cuyo análisis
    # le corresponde al científico de datos o al ingeniero de machine learning.
    return max(0, prediction)

def get_data(request: PredictionRequest) -> pd.DataFrame:
    data_to_predict = transform_to_dataframe(request)
    return data_to_predict

def save_new_data(data_user: pd.DataFrame, prediction: float) -> None:
    
    new_data = data_user.copy()

    new_data['prediction'] = prediction

    # csv_filename = './dataset/data_storage.csv'
    gcs_bucket_name = 'model-dataset-tracker-abi'
    gcs_filename = 'data_storage.csv'

    # csv_data = new_data.to_csv(index=False)

    bucket = client.get_bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_filename)
    blob.upload_from_string(new_data, content_type='text/csv')

    # if not os.path.exists(csv_filename):
    #     new_data.to_csv(csv_filename, index=False, header=True, mode='w')
    #     blob.upload_from_string(new_data, content_type='text/csv')
    # else:
    #     new_data.to_csv(csv_filename, index=False, header=False, mode='a')
    #     blob.upload_from_string(new_data, content_type='text/csv')