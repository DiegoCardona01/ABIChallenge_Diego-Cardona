"""
    views
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains the function that performs the prediction on the data entered by the user.
"""
from .models import PredictionRequest
from .util import get_model, transform_to_dataframe

model = get_model()

def get_prediction(request: PredictionRequest) -> float:
    data_to_predict = transform_to_dataframe(request)
    prediction = model.predict(data_to_predict)[0] # dado que es un array queremos su valor [0]
    # damos un max(0, prediction) porque no es bueno darle a un usuario final
    # una predicción cruda, en este caso si la predicción nos da negativa ponemos un 0
    # así no confundimos al usuario final en caso de predicciones negativas cuyo análisis
    # le corresponde al científico de datos o al ingeniero de machine learning. 
    return max(0, prediction)