"""
    models
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    Contains the features for making predictions and the target to be predicted.
    For serializing the incoming and outgoing JSONs in the requests
"""
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    opening_gross : float
    screens : float
    production_budget : float
    title_year : int
    aspect_ratio : float
    duration : int
    cast_total_facebook_likes : float
    budget : float
    imdb_score : float

class PredictionResponse(BaseModel):
    worldwide_gross: float