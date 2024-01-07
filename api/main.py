"""
    main
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    In this script, we instantiate the API service with FastAPI, create the endpoint, 
    and have the function that executes the model prediction.
"""
from fastapi import FastAPI
from .app.models import PredictionResponse, PredictionRequest
from .app.views import get_prediction, get_data, save_new_data
from google.cloud import storage

bucket_name = 'model-dataset-tracker-abi'

app = FastAPI(docs_url='/')

@app.post('/v1/prediction')
def make_model_prediction(request: PredictionRequest):
    data_user = get_data(request)
    predict = get_prediction(request)
    save_new_data(data_user, predict)
    return PredictionResponse(worldwide_gross=get_prediction(request))