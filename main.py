import argparse
import json
import time
from pprint import pprint

import numpy as np
import pandas as pd

from tuning import asha, sha, grid_search, random_search

if __name__ == '__main__':
    # reproducible results
    np.random.seed(33)

    parser = argparse.ArgumentParser()
    parser.add_argument('-a',
                        '--algorithm',
                        type=str,
                        choices=['random', 'grid', 'sha', 'asha'],
                        required=True,
                        help='The hyperparameter tuning algorithm to use.')
    parser.add_argument('-p',
                        '--parameters',
                        type=json.loads,
                        required=True,
                        help='The parameters for the particular algorithm.')

    args = parser.parse_args()

    target_column = 'RainTomorrow'

    df = pd.read_csv('./preprocessed.csv')
    data = {
        'X': df.loc[:, df.columns != target_column],
        'y': df[[target_column]].values.ravel()
    }

    results = []
    t1 = time.perf_counter()

    if args.algorithm == 'random':
        results = random_search.run(
            data,
            **args.parameters)

    elif args.algorithm == 'grid':
        results = grid_search.run(
            data,
            **args.parameters)

    elif args.algorithm == 'sha':
        results = sha.run(
            data,
            **args.parameters)

    elif args.algorithm == 'asha':
        results = asha.run(**args.parameters)

    elapsed = time.perf_counter() - t1

    score, params = results

    print(f'Took: {elapsed} seconds')
    print(f'Best score: {score}')
    print(f'Best params: ')
    pprint(params)
