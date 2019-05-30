import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

from tuning import util


def run(data, n_iter, cv):
    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(n_jobs=-1),
        param_distributions=util.param_distributions,
        scoring='roc_auc',
        n_iter=n_iter,
        n_jobs=-1,
        cv=cv,
        verbose=2,
    )

    search.fit(data['X'], data['y'])

    return [search.best_score_, search.best_params_]
