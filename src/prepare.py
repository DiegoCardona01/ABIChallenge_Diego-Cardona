"""
    prepare
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains the logic to prepare the data that our ML model will receive.

    First, the paths containing the DVC data files are read with api from dvc library. Next, 
    the data is structured into CSV format, and the corresponding columns are organized. 
    Finally, the final file is created in a format suitable for training the model.
"""
from dvc import api
import pandas as pd
from io import StringIO
import sys
import logging

# Logger to track the data processing progress.
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('Fetching data..')

movie_data_path = api.read('dataset/movies.csv', remote='dataset-tracker', encoding='utf-8')
finantial_data_path = api.read('dataset/finantials.csv', remote='dataset-tracker', encoding='utf-8')
opening_data_path = api.read('dataset/opening_gross.csv', remote='dataset-tracker', encoding='utf-8')

movie_data = pd.read_csv(StringIO(movie_data_path))
fin_data = pd.read_csv(StringIO(finantial_data_path))
opening_data = pd.read_csv(StringIO(opening_data_path))

numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns+['movie_title']]

fin_data = fin_data[['movie_title', 'production_budget', 'worldwide_gross']]

fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left', validate='1:1')
full_movie_data = pd.merge(opening_data, fin_movie_data, on='movie_title', how='left', validate='1:1')

full_movie_data = full_movie_data.drop(['gross', 'movie_title'], axis=1)

full_movie_data.to_csv('dataset/full_data.csv', index=False)

logger.info('Data Fetch and prepared...')