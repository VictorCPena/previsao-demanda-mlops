import numpy as np
import os
import platform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ColumnBaseline(RegressorMixin, BaseEstimator):
    def __init__(self, column="lag_week"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return X[self.column].to_numpy()


class TorchMLP(RegressorMixin, BaseEstimator):
    """Small tabular neural baseline, CPU-only; fitted normalization from training only."""

    def __init__(self, epochs=80, seed=42):
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        self.scale = StandardScaler().fit(X)
        self.center = float(np.mean(y))
        self.spread = max(float(np.std(y)), 1.0)
        xx = torch.tensor(self.scale.transform(X), dtype=torch.float32)
        yy = torch.tensor((np.asarray(y) - self.center) / self.spread, dtype=torch.float32).view(-1, 1)
        net = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=0.001)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xx), yy)
            loss.backward()
            opt.step()
        # Store plain arrays, reconstruct network only at inference; no arbitrary remote model loading.
        self.weights = [(k, v.detach().numpy()) for k, v in net.state_dict().items()]
        return self

    def predict(self, X):
        Z = self.scale.transform(X).astype("float32")
        params = dict(self.weights)
        for i in [0, 2, 4]:
            Z = Z @ params[f"{i}.weight"].T + params[f"{i}.bias"]
            if i != 4:
                Z = np.maximum(Z, 0)
        return np.maximum(0, Z.ravel() * self.spread + self.center)


class SklearnMLP(RegressorMixin, BaseEstimator):
    """Portable two-hidden-layer neural baseline; never relabeled as PyTorch."""

    def __init__(self, epochs=300, seed=42):
        self.epochs = epochs
        self.seed = seed

    def fit(self, X, y):
        self.scale = StandardScaler().fit(X)
        self.center = float(np.mean(y))
        self.spread = max(float(np.std(y)), 1.0)
        self.network = MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            learning_rate_init=0.01, alpha=0.001, max_iter=self.epochs,
            early_stopping=False, random_state=self.seed,
        ).fit(self.scale.transform(X), (np.asarray(y) - self.center) / self.spread)
        return self

    def predict(self, X):
        return np.maximum(0, self.network.predict(self.scale.transform(X)) * self.spread + self.center)


def neural_backend() -> str:
    """Choose explicitly, or select the supported native Intel Mac backend.

    Missing/broken PyTorch on other platforms raises its original error rather
    than silently changing the experiment. FORECAST_NEURAL_BACKEND allows the
    portable profile to be reproduced on Linux for validation.
    """
    requested = os.environ.get("FORECAST_NEURAL_BACKEND", "auto")
    if requested not in {"auto", "torch", "sklearn"}:
        raise ValueError("FORECAST_NEURAL_BACKEND must be auto, torch or sklearn")
    if requested != "auto":
        return requested
    return "sklearn" if platform.system() == "Darwin" and platform.machine() == "x86_64" else "torch"
