import math
import xgboost as xgb
from scipy.stats import uniform, randint


def run(data, n_configs, min_r, max_r, reduction_factor, cv):
    configurations = []
    for i in range(n_configs):
        configurations.append({
            'params': _get_random_config(),
            'error': 0
        })

    s_max = math.floor(math.log(max_r / min_r, reduction_factor))
    d_matrix = xgb.DMatrix(data=data['X'], label=data['y'])

    assert(n_configs >= math.pow(reduction_factor, s_max))

    for i in range(s_max + 1):
        n = max(1, math.floor(n_configs * math.pow(reduction_factor, -i)))
        r = int(min_r * math.pow(reduction_factor, i))
        for config in configurations:
            config['error'] = xgb.cv(
                params=config['params'],
                dtrain=d_matrix,
                num_boost_round=r,
                nfold=cv,
                metrics='error',
                verbose_eval=True
            )['test-error-mean'].min()

        configurations.sort(key=lambda c: c['error'])
        configurations = configurations[:n]

    best_config = min(configurations, key=lambda c: c['error'])

    return [1 - best_config['error'], best_config['params']]


def _get_random_config():
    param_distributions = {
        'eta': uniform(),
        'gamma': randint(0, 10),
        'max_depth': randint(1, 12),
        'min_child_weight': randint(0, 12),
        'max_delta_step': randint(0, 12),
        'subsample': uniform(),
        'colsample_bytree': uniform(),
    }

    generated_config = {}

    for param, dist in param_distributions.items():
        generated_config[param] = dist.rvs()

    return generated_config
