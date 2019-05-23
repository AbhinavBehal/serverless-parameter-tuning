from sklearn.model_selection import GridSearchCV


def run(data, estimator):
    param_grid = {
        'eta': [0.3, 0.9],
        'gamma': [0, 2],
        'max_depth': [6],
        'min_child_weight': [0, 2],
        'max_delta_step': [0, 2],
        'subsample': [0.5, 1.0],
        'colsample_bytree': [0.0, 0.5, 1.0],
        'tree_method': ['auto']
    }

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
    )

    search.fit(data['X'], data['y'])

    return [search.best_score_, search.best_params_]
