import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import features, generate, temporal_folds, next_features, FEATURES
from models import SeasonalNaive
from monitor import drift
from serve import create_app
from local_test_client import LocalTestClient


def test_features_do_not_read_current_or_future_target():
    frame = generate(days=400, products=1)
    original = features(frame)
    target_day = frame.day.iloc[100]
    altered = frame.copy()
    altered.loc[altered.day >= target_day, 'units'] += 10000
    rebuilt = features(altered)
    pd.testing.assert_frame_equal(original.loc[original.day <= target_day, FEATURES].reset_index(drop=True),
                                  rebuilt.loc[rebuilt.day <= target_day, FEATURES].reset_index(drop=True))


def test_temporal_split_groups_all_products_by_date():
    frame = features(generate())
    folds, train, test = temporal_folds(frame)
    assert train.day.max() < test.day.min()
    for past, future in folds:
        assert past.day.max() < future.day.min()
        assert set(past.day).isdisjoint(future.day)
        assert len(future.product_id.unique()) == 6


def test_missing_day_rejected():
    frame = generate(days=100, products=1).drop(index=40)
    with pytest.raises(ValueError, match='lacunas'):
        features(frame)


def test_serving_features_match_training_features():
    raw = generate(days=100, products=1)
    frame = features(raw)
    row = frame.iloc[-1]
    history = [{'day': str(r.day.date()), 'units': r.units} for r in raw.iloc[-29:-1].itertuples()]
    X = next_features('P00', history, row.day, row.promo)
    np.testing.assert_allclose(X.iloc[0].astype(float), row[FEATURES].astype(float))


@pytest.fixture
def client(tmp_path):
    X = features(generate(days=100, products=1))[FEATURES]
    bundle = {'model': SeasonalNaive().fit(X), 'name': 'seasonal_naive', 'products': ['P00'],
              'q90': 5., 'version': '1', 'synthetic': True}
    joblib.dump(bundle, tmp_path / 'model.joblib')
    (tmp_path / 'deployment.json').write_text(json.dumps({'active': {'file': 'model.joblib'}}))
    return LocalTestClient(create_app(tmp_path))


def request_body():
    return {'product_id': 'P00', 'day': '2026-01-29', 'promo': 0,
            'history': [{'day': str(d.date()), 'units': 12.} for d in pd.date_range('2026-01-01', periods=28)]}


def test_prediction_and_recursive_api(client):
    response = client.post('/predict', json=request_body())
    assert response.status_code == 200
    assert response.json()['prediction'] == 12
    response = client.post('/forecast', json={**request_body(), 'horizon': 3, 'planned_promotions': [0, 0, 0]})
    assert response.status_code == 200
    assert len(response.json()['predictions']) == 3


def test_api_rejects_unknown_product_and_leakage(client):
    payload = request_body()
    payload['product_id'] = 'P99'
    assert client.post('/predict', json=payload).status_code == 422
    payload = request_body()
    payload['history'][-1]['day'] = payload['day']
    assert client.post('/predict', json=payload).status_code == 422


def test_api_without_model_returns_503(tmp_path):
    client = LocalTestClient(create_app(tmp_path))
    assert not client.get('/health').json()['model_ready']
    assert client.post('/predict', json=request_body()).status_code == 503


def test_monitor_detects_simulated_shift():
    reference = pd.DataFrame({c: np.arange(1, 101) for c in ['lag_1', 'lag_7', 'mean_7', 'mean_28']})
    assert not drift(reference, reference)['review_needed']
    assert drift(reference, reference*10)['review_needed']


def test_health_requires_real_artifact(tmp_path):
    workspace = tmp_path
    (workspace / "deployment.json").write_text(json.dumps({'active': {'file': 'missing.joblib'}}))
    client = LocalTestClient(create_app(workspace))
    assert client.get('/health').json() == {'status': 'degraded', 'model_ready': False}
    assert client.post('/predict', json=request_body()).status_code == 503


def test_root_documents_available_endpoints(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['docs'] == '/docs'
    assert '/predict' in response.json()['endpoints']


def test_consecutive_days_rejects_duplicate_and_gap():
    from data import consecutive_days
    assert consecutive_days(pd.date_range('2026-01-01', periods=28))
    assert not consecutive_days(pd.to_datetime(['2026-01-01', '2026-01-01']))
    assert not consecutive_days(pd.to_datetime(['2026-01-01', '2026-01-03']))
