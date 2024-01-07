"""
    views
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains the function that performs the prediction on the data entered by the user.
"""
from .models import PredictionRequest
from .util import get_model, transform_to_dataframe
from google.cloud import storage
from io import StringIO
import pandas as pd
import os

client = storage.Client()
model = get_model()

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    prediction = model.predict(data_to_predict)[0] 
    return max(0, prediction)

def get_data(request: PredictionRequest) -> pd.DataFrame:
    data_to_predict = transform_to_dataframe(request)
    return data_to_predict


def save_new_data(data_user: pd.DataFrame, prediction: float) -> None:

    new_data = data_user.copy()
    new_data['prediction'] = prediction

    gcs_bucket_name = 'model-dataset-tracker-abi'
    gcs_filename = 'storage/data_storage.csv'

    try:
        bucket = client.get_bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_filename)
        existing_data = blob.download_as_text()
        existing_df = pd.read_csv(StringIO(existing_data))
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        existing_df = pd.DataFrame()

    combined_data = pd.concat([existing_df, new_data], ignore_index=True)

    combined_csv_data = combined_data.to_csv(index=False)
    combined_bytes_data = combined_csv_data.encode('utf-8')

    blob.upload_from_string(combined_bytes_data, content_type='text/csv')