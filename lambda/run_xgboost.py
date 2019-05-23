import json
import logging
import pandas as pd
import xgboost as xgb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(message)s')

try:
    import unzip_requirements
except ImportError:
    logger.error("failed unzipping reqs")
    pass


def run(event, context):
    logger.info("Called run function")
    input_body = json.loads(event['body'])
    if 'params' not in input_body:
        logger.error("Validation Failed")
        raise Exception("Couldn't create the todo item.")

    xgb_params = input_body['params']
    logger.info("Reading CSV")
    df = pd.read_csv("./preprocessed.csv")

    X = df.loc[:, df.columns != 'RainTomorrow']
    y = df[['RainTomorrow']]

    data_dmatrix = xgb.DMatrix(data=X, label=y)

    logger.info("Performing Cross Validation")
    # this returns a dataframe of mean error values, each row adds another boosting tree.
    results = xgb.cv(dtrain=data_dmatrix, params=xgb_params, nfold=3, seed=123, metrics="error")

    logger.info("Done")
    # select the iteration with the lowest error (with the optimal number of trees ensembled)
    best_mean_score = results['test-error-mean'].min()

    response = {
        "statusCode": 200,
        "body": json.dumps({"status": "OK", "evaluation_score": best_mean_score}),
    }

    logger.info("Sending response", response)

    return response
