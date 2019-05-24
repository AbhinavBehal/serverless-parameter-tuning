from timeit import timeit

import pandas as pd
import xgboost as xgb
from pprint import pprint
from tuning import random_search, grid_search, sha

target_column = 'RainTomorrow'

df = pd.read_csv('./preprocessed.csv')
data = {
    'X': df.loc[:, df.columns != target_column],
    'y': df[[target_column]].values.ravel()
}

estimator = xgb.XGBClassifier(n_jobs=-1, random_state=33)

# score, params = grid_search.run(data=data, estimator=estimator)

elapsed = timeit(stmt='sha.run(data=data, n_configurations=16, min_r=1, max_r=20, reduction_factor=2)',
                 number=1,
                 globals=globals())

print(f'{elapsed} seconds')

# print(f'Best Score: {score}')
# print('Best Params:')
# pprint(params)
