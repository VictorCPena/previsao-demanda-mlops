import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from forecasting.models import SklearnMLP, neural_backend
from forecasting.train import candidates
from scripts.load_api import rss_mebibytes


def test_rss_units_are_platform_specific():
    assert rss_mebibytes(2 * 1024 ** 2, "darwin") == 2
    assert rss_mebibytes(2 * 1024, "linux") == 2


def test_portable_network_learns_and_roundtrips(tmp_path):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(350, 3)), columns=["x", "y", "z"])
    y = 15 + 3 * X.x + 2 * np.maximum(0, X.y)
    model = clone(SklearnMLP()).fit(X.iloc[:250], y.iloc[:250])
    pred = model.predict(X.iloc[250:])
    assert np.mean(np.abs(pred - y.iloc[250:])) < 0.5
    assert (pred >= 0).all()
    np.testing.assert_allclose(model.scale.mean_, X.iloc[:250].mean())
    joblib.dump(model, tmp_path / "network.joblib")
    np.testing.assert_allclose(joblib.load(tmp_path / "network.joblib").predict(X.iloc[250:]), pred)


def test_backend_selection_is_explicit_and_preserves_torch(monkeypatch):
    monkeypatch.setenv("FORECAST_NEURAL_BACKEND", "auto")
    monkeypatch.setattr("forecasting.models.platform.system", lambda: "Darwin")
    monkeypatch.setattr("forecasting.models.platform.machine", lambda: "x86_64")
    assert neural_backend() == "sklearn"
    assert "sklearn_mlp" in candidates() and "torch_mlp" not in candidates()
    monkeypatch.setenv("FORECAST_NEURAL_BACKEND", "torch")
    assert "torch_mlp" in candidates() and "sklearn_mlp" not in candidates()
    monkeypatch.setenv("FORECAST_NEURAL_BACKEND", "unknown")
    with pytest.raises(ValueError):
        candidates()
