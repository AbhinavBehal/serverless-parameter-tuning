import xgboost as xgb
from sklearn.model_selection import GridSearchCV

from tuning import util


def run(data, cv):
    search = GridSearchCV(
        estimator=xgb.XGBClassifier(n_jobs=-1),
        param_grid=util.param_grid,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        cv=cv
    )

    search.fit(data['X'], data['y'])

    return [search.best_score_, search.best_params_]
