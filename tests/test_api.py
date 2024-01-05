"""
    test_api
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    In this script, we implement a basic test for the API to ensure that it is functioning
    as intended. To do this, we call the instantiated API from the 'app' folder and use the test 
    client. The test involves checking that the API is running with a status_code of 200 and 
    with some test data that should yield a zero and non-zero value. 
"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

# Test de cuando se le ingresan valores nulos a la api para predecir
def test_null_prediction():
    response = client.post('/v1/prediction', json={ 
                                                    'opening_gross' : 0,
                                                    'screens' : 0,
                                                    'production_budget' : 0,
                                                    'title_year' : 0,
                                                    'aspect_ratio' : 0,
                                                    'duration' : 0,
                                                    'cast_total_facebook_likes' : 0,
                                                    'budget' : 0,
                                                    'imdb_score' : 0
                                                    })
    # Primero nos aseguramos que se ejecute, por lo tanto el status_code debe ser igual a 200.
    assert response.status_code == 200
    # Como sabemos la respuesta debe ser cero y nos aseguramos de eso
    assert response.json()['worldwide_gross'] == 0

# Creamos una función referencia que sabemos que su valor no dará cero, la tomamos de forma arbitraria
def test_random_prediction():
    response = client.post('/v1/prediction', json={ 
                                                    'opening_gross' : 8330681,
                                                    'screens' : 2271,
                                                    'production_budget' : 13000000,
                                                    'title_year' : 1999,
                                                    'aspect_ratio' : 1.85,
                                                    'duration' : 97,
                                                    'cast_total_facebook_likes' : 37907,
                                                    'budget' : 16000000,
                                                    'imdb_score' : 7.2
                                                    })
    assert response.status_code == 200
    assert response.json()['worldwide_gross'] != 0