import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import json
import logging

def run(event, context):

    input_body = json.loads(event['body'])
    if 'params' not in input_body:
        logging.error("Validation Failed")
        raise Exception("Couldn't create the todo item.")
        return

    xgb_params = input_body['params']

    df = pd.read_csv("../preprocessed.csv")

    X = df.loc[:, df.columns != 'RainTomorrow']
    y = df[['RainTomorrow']]

    model = XGBClassifier(xgb_params)
    kfold = KFold(n_splits=5, random_state=7)

    results = cross_val_score(model, X, y, cv=kfold, verbose=3)

    mean_score = results.mean()

    response = {
        "statusCode": 200,
        "body": {
            "evaluation_score":mean_score
        }
    }

    return response
