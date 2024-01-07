"""
    prepare
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This Python script fetches data related to movies, including information about movies, 
    financials, and opening gross. It then processes and merges the datasets, focusing on 
    numeric columns and relevant information. The final dataset is saved as 
    path_full_data. The script uses logging for information and error tracking 
    throughout its execution.
"""

import logging
import sys
from io import StringIO

import pandas as pd
from dvc import api

from config import path_finantials, path_movies, path_opening_gross, path_full_data
from util import DataLoader

# Logger to track the data processing progress.
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(name)s: %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)

logging.info('Fetching data..')

movie_data_load = DataLoader(path_movies)
fin_data_load = DataLoader(path_finantials)
opening_data_load = DataLoader(path_opening_gross)

movie_data = movie_data_load.load_data()
fin_data = fin_data_load.load_data()
opening_data = opening_data_load.load_data()

numeric_columns_mask = (movie_data.dtypes == float) | (movie_data.dtypes == int)
numeric_columns = [column for column in numeric_columns_mask.index if numeric_columns_mask[column]]
movie_data = movie_data[numeric_columns+['movie_title']]

fin_data = fin_data[['movie_title', 'production_budget', 'worldwide_gross']]

fin_movie_data = pd.merge(fin_data, movie_data, on='movie_title', how='left')
full_movie_data = pd.merge(opening_data, fin_movie_data, on='movie_title', how='left')

full_movie_data = full_movie_data.drop(['gross', 'movie_title'], axis=1)

full_movie_data.to_csv(path_full_data, index=False)

logger.info('Data Fetch and prepared...')