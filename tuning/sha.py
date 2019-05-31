from concurrent.futures import ProcessPoolExecutor

import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from tuning import util


def run(data, n_configs, min_r, max_r, reduction_factor, cv):
    configurations = []
    for i in range(n_configs):
        configurations.append({
            'params': util.get_random_params(),
            'score': 0
        })

    s_max = int(math.ceil(math.log(max_r / min_r, reduction_factor))) + 1

    with ProcessPoolExecutor(max_workers=None) as executor:
        for i in range(s_max):
            n = max(1, math.floor(n_configs * math.pow(reduction_factor, -i)))
            r = int(min_r * math.pow(reduction_factor, i))

            configurations.sort(key=lambda c: c['score'], reverse=True)
            configurations = configurations[:n]

            futures = []
            for config in configurations:
                futures.append({
                    'future': executor.submit(
                        cross_val_score,
                        XGBClassifier(**config['params'], n_estimators=r),
                        data['X'],
                        data['y'],
                        cv=StratifiedKFold(n_splits=cv, shuffle=True),
                    ),
                    'config': config
                })

            for f in futures:
                f['config']['score'] = f['future'].result().mean()
                print(f'Rung {i} - score: {f["config"]["score"]}')

    best_config = max(configurations, key=lambda c: c['score'])

    return [best_config['score'], best_config['params']]
