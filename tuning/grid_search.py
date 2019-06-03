import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from tuning import util


def run(data, n_workers, max_samples, cv):
    search = GridSearchCV(
        estimator=xgb.XGBClassifier(n_jobs=-1, n_estimators=10),
        param_grid=_generate_grid(max_samples),
        scoring='roc_auc',
        n_jobs=n_workers,
        verbose=1,
        cv=cv
    )

    search.fit(data['X'], data['y'])

    return [search.best_score_, search.best_params_]


def _generate_grid(max_samples):
    generated_grid = {}

    for param, dist in util.param_distributions.items():
        generated_grid[param] = []
        num_samples = np.random.randint(1, max_samples + 1)

        if isinstance(dist, list):
            generated_grid[param].extend(np.random.choice(dist, size=num_samples))
        else:
            for i in range(num_samples):
                generated_grid[param].append(dist.rvs())

    return generated_grid
