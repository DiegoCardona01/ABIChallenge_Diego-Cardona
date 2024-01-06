import requests

url = 'https://deployment-abichallenge-ml-service-ejp2ragddq-uc.a.run.app/v1/prediction'

json = {
        "opening_gross": 8330681,
        "screens": 2271,
        "production_budget": 14500000,
        "title_year": 1999,
        "aspect_ratio": 1.85,
        "duration": 97,
        "cast_total_facebook_likes": 37907,
        "budget": 16000000,
        "imdb_score": 7.2
        }

response = requests.post(url, json=json)
result = response.json()['worldwide_gross']

print('\n')
print('**'*40)
print(f'The prediction for the features in the script is {result}')
print('Now we can conected from a frontend, backend or other services ')
print('**'*40)