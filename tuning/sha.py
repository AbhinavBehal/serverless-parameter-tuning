import math
import xgboost as xgb
from pprint import pprint
from scipy.stats import uniform, randint


def run(data, n_configurations, min_r, max_r, reduction_factor):
    configurations = []
    for i in range(n_configurations):
        configurations.append({
            'params': _get_random_config(),
            'error': 0
        })

    s_max = math.floor(math.log(max_r / min_r, reduction_factor))
    d_matrix = xgb.DMatrix(data=data['X'], label=data['y'])

    assert(n_configurations >= math.pow(reduction_factor, s_max))

    for i in range(s_max + 1):
        print(i)
        n = max(1, math.floor(n_configurations * math.pow(reduction_factor, -i)))
        r = int(min_r * math.pow(reduction_factor, i))
        for config in configurations:
            config['error'] = xgb.cv(
                params=config['params'],
                dtrain=d_matrix,
                num_boost_round=r,
                nfold=3,
                metrics='error',
                seed=33
            )['test-error-mean'].min()

        configurations.sort(key=lambda c: c['error'])
        configurations = configurations[:n]

    pprint(min(configurations, key=lambda c: c['error']))


def _get_random_config():
    param_distributions = {
        'eta': uniform(),
        'gamma': randint(0, 10),
        'max_depth': randint(1, 12),
        'min_child_weight': randint(0, 12),
        'max_delta_step': randint(0, 12),
        'subsample': uniform(),
        'colsample_bytree': uniform(),
        # 'tree_method': ['auto']
    }

    generated_config = {}

    for param, dist in param_distributions.items():
        generated_config[param] = dist.rvs()

    return generated_config
