"""
    train
    create by: Diego Fernando Cardona Pineda
    Date: 05/01/2024

    This script contains the model training logic. After preparing the data in the prepare.py 
    script, the data is received, processed accordingly, and then split into training and test 
    sets. Hyperparameter tuning is performed, and the model is executed with the best results.
"""
from util import update_model, ModelEvaluator
from config import path_full_data
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PowerTransformer

import logging
import sys
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s: %(message)s', 
                              datefmt='%H:%M:%S')

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)

logger.addHandler(handler)

logger.info('Loading Data...')
data = pd.read_csv(path_full_data)

logger.info('Loading model...')
model = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', missing_values=np.nan)),
    ('core_model', GradientBoostingRegressor())
])

logger.info('Scaling the data')
data_scaled = data.copy()
scaler = PowerTransformer(method='box-cox')
data_scaled = scaler.fit_transform(data_scaled)
data = pd.DataFrame(data_scaled, columns = data.columns)

logger.info('Separating dataset into train and test')
X = data.drop('worldwide_gross', axis=1)
y = data['worldwide_gross']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

logging.info('Seatting Hyperparameters to tune')
param_tuning = {'core_model__n_estimators': range(20,301,20)}


grid_search = GridSearchCV(model, param_grid=param_tuning, scoring='r2', cv=5)


logger.info('Starting grid search...')
grid_search.fit(X_train, y_train)

logging.info('Cross validating with best model...')
final_result = cross_validate(grid_search.best_estimator_, X_train, y_train, return_train_score=True, cv=5)

train_score = np.mean(final_result['train_score'])
test_score = np.mean(final_result['test_score'])

assert train_score > 0.7
assert test_score > 0.65

logger.info(f'Train Score: {train_score}')
logger.info(f'Test Score: {test_score}')

logger.info('Updating model...')
update_model(grid_search.best_estimator_)

logger.info('Generating model report...')
validation_score = grid_search.best_estimator_.score(X_test, y_test)
best_model = grid_search.best_estimator_
model_evaluator = ModelEvaluator()

model_evaluator.save_simple_metrics_report(train_score, test_score, validation_score, best_model)
# save_simple_metrics_report(train_score, test_score, validation_score, grid_search.best_estimator_)

y_test_pred = grid_search.best_estimator_.predict(X_test)
model_evaluator.get_model_performance_test_set(y_test, y_test_pred)
# get_model_performance_test_set(y_test, y_test_pred)

logger.info('Training Finished')