import random

from scipy.stats import randint, uniform

param_distributions = {
    'eta': uniform(),
    'gamma': randint(0, 10),
    'max_depth': randint(1, 12),
    'min_child_weight': randint(0, 12),
    'max_delta_step': randint(0, 12),
    'subsample': uniform(),
    'colsample_bytree': uniform(),
}

param_grid = {
    'eta': [0.3, 0.9],
    'gamma': [0, 2],
    'max_depth': [6],
    'min_child_weight': [0, 2],
    'max_delta_step': [0, 2],
    'subsample': [0.5, 1.0],
    'colsample_bytree': [0.0, 0.5, 1.0],
}


def get_random_params():
    generated_config = {}

    for param, dist in param_distributions.items():
        if isinstance(dist, list):
            generated_config[param] = random.choice(dist)
        else:
            generated_config[param] = dist.rvs()

    return generated_config
