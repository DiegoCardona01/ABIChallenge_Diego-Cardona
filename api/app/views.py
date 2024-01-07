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
    prediction = model.predict(data_to_predict)[0] # dado que es un array queremos su valor [0]
    # damos un max(0, prediction) porque no es bueno darle a un usuario final
    # una predicción cruda, en este caso si la predicción nos da negativa ponemos un 0
    # así no confundimos al usuario final en caso de predicciones negativas cuyo análisis
    # le corresponde al científico de datos o al ingeniero de machine learning
    return max(0, prediction)

def get_data(request: PredictionRequest) -> pd.DataFrame:
    data_to_predict = transform_to_dataframe(request)
    return data_to_predict

def save_new_data(data_user: pd.DataFrame, prediction: float) -> None:
    
    new_data = data_user.copy()

    new_data['prediction'] = prediction


def save_new_data(data_user: pd.DataFrame, prediction: float) -> None:

    new_data = data_user.copy()
    new_data['prediction'] = prediction

    # GCS configuration
    gcs_bucket_name = 'model-dataset-tracker-abi'
    gcs_filename = 'storage/data_storage.csv'

    # Download existing data from GCS if it exists
    try:
        bucket = client.get_bucket(gcs_bucket_name)
        blob = bucket.blob(gcs_filename)
        existing_data = blob.download_as_text()
        existing_df = pd.read_csv(StringIO(existing_data))
    except:
        # If the file doesn't exist yet, start with an empty DataFrame
        existing_df = pd.DataFrame()

    # Concatenate existing data with new data
    combined_data = pd.concat([existing_df, new_data], ignore_index=True)

    # Upload the combined data back to GCS
    combined_csv_data = combined_data.to_csv(index=False)
    combined_bytes_data = combined_csv_data.encode('utf-8')

    blob.upload_from_string(combined_bytes_data, content_type='text/csv')   


#### past data #####
    # csv_filename = './dataset/data_storage.csv'
    # gcs_bucket_name = 'model-dataset-tracker-abi'
    # gcs_filename = 'storage/data_storage.csv'

    # csv_data = new_data.to_csv(index=False)
    # bytes_data = csv_data.encode('utf-8')

    # bucket = client.get_bucket(gcs_bucket_name)
    # blob = bucket.blob(gcs_filename)
    # blob.upload_from_string(bytes_data, content_type='text/csv')

    # if not os.path.exists(csv_filename):
    #     new_data.to_csv(csv_filename, index=False, header=True, mode='w')
    #     blob.upload_from_string(new_data, content_type='text/csv')
    # else:
    #     new_data.to_csv(csv_filename, index=False, header=False, mode='a')
    #     blob.upload_from_string(new_data, content_type='text/csv')