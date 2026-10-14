import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class SeasonalNaive(RegressorMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return np.asarray(X["lag_7"], dtype=float)


def candidates():
    return {"seasonal_naive": SeasonalNaive(),
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                max_iter=120, max_leaf_nodes=15, learning_rate=.08,
                l2_regularization=2, random_state=42)}
