import pandas as pd
import xgboost as xgb
from pprint import pprint
from tuning import random_search, grid_search

target_column = 'RainTomorrow'

df = pd.read_csv('./preprocessed.csv')
data = {
    'X': df.loc[:, df.columns != target_column],
    'y': df[[target_column]].values.ravel()
}

estimator = xgb.XGBClassifier(n_jobs=-1, random_state=33)

score, params = random_search.run(data=data, estimator=estimator, n_iter=2)

print(f'Best Score: {score}')
print('Best Params:')
pprint(params)
