import json
import hashlib
import numpy as np
import pandas as pd
import pytest
import joblib
from fastapi.testclient import TestClient
from forecasting.data import features, FEATURES
from forecasting.api import create_app
from forecasting.models import ColumnBaseline, TorchMLP, neural_backend
from forecasting.operations import deploy, rollback, drift, matured_performance


def grid():
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC"),
            "zone": 74,
            "count": np.arange(400, dtype=float),
        }
    )


@pytest.mark.parametrize("h", [1, 6, 24])
def test_no_future_leakage(h):
    raw = grid()
    target = raw.time.iloc[300]
    origin = target - pd.Timedelta(hours=h)
    before = features(raw, h).set_index("time").loc[target, FEATURES]
    raw.loc[raw.time > origin, "count"] += 100000
    after = features(raw, h).set_index("time").loc[target, FEATURES]
    pd.testing.assert_series_equal(before, after)


def test_gaps_and_duplicates_rejected():
    with pytest.raises(ValueError):
        features(grid().drop(index=10), 1)
    with pytest.raises(ValueError):
        features(pd.concat([grid(), grid().iloc[:1]]), 1)


def setup(root):
    frame = features(grid(), 1)
    model = ColumnBaseline().fit(frame[FEATURES], frame["count"])
    file = root / "model.joblib"
    joblib.dump({"model": model, "zones": [74], "winner": "test", "q90": 2.0, "data_sha256": "fixture"}, file)
    info = {"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "cv_mae": 1.0}
    (root / "deployment.json").write_text(json.dumps({"active": {"1": info}, "previous": None}))
    return info


def payload():
    return {
        "zone": 74,
        "horizon": 1,
        "history": [{"time": str(r.time), "count": float(r.count)} for r in grid().tail(200).itertuples()],
    }


def test_api_contract_integrity_and_unexpected_input(tmp_path):
    setup(tmp_path)
    client = TestClient(create_app(tmp_path))
    p = payload()
    assert client.post("/predict", json=p).status_code == 200
    p["zone"] = 999
    assert client.post("/predict", json=p).status_code == 422
    p = payload()
    p["history"][0]["count"] = -1
    assert client.post("/predict", json=p).status_code == 422
    p = payload()
    p["history"][0]["time"] = p["history"][1]["time"]
    assert client.post("/predict", json=p).status_code == 422
    # A new process or app must refuse a corrupted artifact.
    (tmp_path / "model.joblib").write_bytes(b"corrupt")
    assert TestClient(create_app(tmp_path)).post("/predict", json=payload()).status_code == 503


def test_promotion_gate_and_rollback(tmp_path):
    info = setup(tmp_path)
    with pytest.raises(ValueError, match="regression"):
        deploy(tmp_path, {"1": {**info, "cv_mae": 5}})
    deploy(tmp_path, {"1": {**info, "cv_mae": 0.9}})
    rollback(tmp_path)
    assert json.loads((tmp_path / "deployment.json").read_text())["active"]["1"]["cv_mae"] == 1


def test_drift_and_matured_error_trigger_review():
    ref = pd.DataFrame({"x": np.arange(100)})
    assert drift(ref, ref + 1000)["features"]["x"]["review"]
    assert matured_performance([10, 10], [0, 0], 1)["review"]


@pytest.mark.skipif(neural_backend() != "torch", reason="Explicit portable profile; PyTorch not installed")
def test_torch_roundtrip(tmp_path):
    frame = features(grid(), 1)
    X = frame[FEATURES]
    y = frame["count"]
    m = TorchMLP(epochs=5).fit(X, y)
    joblib.dump(m, tmp_path / "net.joblib")
    assert np.isfinite(joblib.load(tmp_path / "net.joblib").predict(X)).all()


def test_promotion_rejects_corrupted_candidate_before_manifest_change(tmp_path):
    info = setup(tmp_path)
    before = (tmp_path / "deployment.json").read_text()
    with pytest.raises(ValueError, match="integrity"):
        deploy(tmp_path, {"1": {**info, "sha256": "0" * 64}})
    assert (tmp_path / "deployment.json").read_text() == before
