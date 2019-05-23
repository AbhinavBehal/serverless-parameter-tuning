from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV


def run(data, estimator, n_iter=5, random_state=33):
    param_distributions = {
        'eta': uniform(),
        'gamma': randint(0, 10),
        'max_depth': randint(1, 12),
        'min_child_weight': randint(0, 12),
        'max_delta_step': randint(0, 12),
        'subsample': uniform(),
        'colsample_bytree': uniform(),
        'tree_method': ['auto']
    }

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        scoring='roc_auc',
        n_iter=n_iter,
        n_jobs=-1,
        cv=3,
        verbose=2,
        random_state=random_state
    )

    search.fit(data['X'], data['y'])

    return [search.best_score_, search.best_params_]
