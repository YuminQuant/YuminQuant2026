"""Shared native XGBoost fitting with reusable DMatrix inputs."""


def make_matrix(xgb, features, labels=None, n_jobs=-1):
    return xgb.DMatrix(features, label=labels, nthread=n_jobs)


def fit_matrix(xgb, params, matrix):
    # Use the installed sklearn adapter's parameter mapping (including seed
    # and thread aliases), preserving its defaults across XGBoost versions.
    estimator = xgb.XGBRegressor(**params)
    if callable(estimator.objective):
        raise ValueError("matrix fitting requires a named XGBoost objective")
    estimator._Booster = xgb.train(
        estimator.get_xgb_params(),
        matrix,
        num_boost_round=estimator.get_num_boosting_rounds(),
        verbose_eval=False,
    )
    return estimator
