import json
from datetime import timedelta
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from backtest import backtest, rollout, seasonal_predictor
from data import generate
from load_study import describe, local_base, measure_level


def predictor(_product, history, _day, _promo):
    return history[-1]['units'] + 1


def test_rollout_is_recursive_and_does_not_mutate_input():
    history=[{'day':str(day.date()), 'units':10.0} for day in pd.date_range('2026-01-01',periods=28)]
    result=rollout('P00',history,'2026-01-29',[0,0,0],predictor)
    assert [r['prediction'] for r in result]==[11,12,13]
    assert len(history)==28
    assert [r['day'] for r in result]==['2026-01-29','2026-01-30','2026-01-31']


def test_recursive_seasonal_baseline_reuses_predicted_day_seven():
    history=[{'day':str(day.date()),'units':float(i)} for i, day in enumerate(pd.date_range('2026-01-01',periods=28))]
    r=rollout('P00',history,'2026-01-29',[0]*14,seasonal_predictor)
    assert r[0]['prediction']==21 and r[7]['prediction']==21


def test_backtest_future_target_mutation_cannot_change_first_origin_predictions():
    raw=generate(days=100,products=1)
    original,report=backtest(raw,predictor,'2024-03-01',horizons=(1,7),stride=7)
    assert '7' in report['at_horizon']
    assert len(original)>0
    altered=raw.copy()
    altered.loc[altered.day>=pd.Timestamp('2024-03-01'),'units'] += 9999
    changed,_=backtest(altered,predictor,'2024-03-01',horizons=(1,7),stride=7)
    first=str(original.origin.min())
    left=original[original.origin==first].prediction.to_numpy()
    right=changed[changed.origin==first].prediction.to_numpy()
    np.testing.assert_allclose(left,right)
    assert not original[original.origin==first].actual.equals(changed[changed.origin==first].actual)


def test_backtest_counts_and_per_product():
    raw=generate(days=105,products=2)
    rows, report=backtest(raw,predictor,'2024-03-01',horizons=(1,7,14),stride=7)
    assert report['origin_count']==rows[['product_id','origin']].drop_duplicates().shape[0]
    assert report['at_horizon']['14']['model']['n']==report['origin_count']
    assert set(report['by_product_at_horizon']['7'])=={'P00','P01'}
    assert report['through_horizon']['7']['model']['n']==report['origin_count']*7


def test_backtest_rejects_short_or_unknown():
    with pytest.raises(ValueError):backtest(generate(days=32,products=1),predictor,'2024-02-01',horizons=(14,))
    with pytest.raises(ValueError):backtest(generate(days=100,products=1),predictor,'2024-03-01',products=['P99'])


def test_local_load_safety_and_percentile_calibration():
    assert describe([1,2,3])['p50_ms']==2
    assert local_base('http://127.0.0.1:8000/')=='http://127.0.0.1:8000'
    for url in ('https://127.0.0.1:8000','http://example.com:8000','http://127.0.0.1','http://localhost:8000@evil.com'):
        with pytest.raises(ValueError):local_base(url)
    with pytest.raises(ValueError):measure_level('http://127.0.0.1:8000',{},17,1)


def test_model_cache_reuses_bundle_and_changes_on_promotion(monkeypatch, tmp_path):
    import joblib
    from data import features, FEATURES
    from models import SeasonalNaive
    from serve import create_app
    from local_test_client import LocalTestClient
    from sklearn.dummy import DummyRegressor
    source=generate(days=100,products=1)
    X=features(source)[FEATURES]
    seasonal=SeasonalNaive().fit(X)
    constant=DummyRegressor(strategy='constant',constant=99).fit(X, np.ones(len(X)))
    for name,model in [('one',seasonal),('two',constant)]:
        joblib.dump({'model':model,'name':name,'products':['P00'],'q90':1.,
                     'version':name,'synthetic':True},tmp_path/f'{name}.joblib')
    registry=tmp_path/'deployment.json'
    registry.write_text(json.dumps({'active':{'file':'one.joblib'}}))
    app=LocalTestClient(create_app(tmp_path))
    hist=[{'day':str(row.day.date()),'units':float(row.units)} for row in source.tail(28).itertuples()]
    body={'product_id':'P00','day':(pd.Timestamp(source.day.max()).date() + timedelta(days=1)).isoformat(),
          'promo':0,'history':hist}
    original=app.post('/predict',json=body)
    repeated=app.post('/predict',json=body)
    assert original.json()['model']==repeated.json()['model']=='one'
    registry.write_text(json.dumps({'active':{'file':'two.joblib'}}))
    promoted=app.post('/predict',json=body)
    assert promoted.json()['model']=='two' and promoted.json()['prediction']==99
    registry.write_text(json.dumps({'active':{'file':'one.joblib'}}))
    assert app.post('/predict',json=body).json()['model']=='one'


def test_model_path_traversal_denied(tmp_path):
    from serve import create_app
    from local_test_client import LocalTestClient
    outside=tmp_path.parent/'not_an_allowed_model.joblib'
    outside.write_bytes(b'not a model')
    (tmp_path/'deployment.json').write_text(json.dumps({'active':{'file':'../not_an_allowed_model.joblib'}}))
    client=LocalTestClient(create_app(tmp_path))
    source=generate(days=100,products=1)
    hist=[{'day':str(row.day.date()),'units':float(row.units)} for row in source.tail(28).itertuples()]
    body={'product_id':'P00','day':(pd.Timestamp(source.day.max()).date() + timedelta(days=1)).isoformat(),'promo':0,'history':hist}
    assert client.post('/predict',json=body).status_code==503
    outside.unlink()


def test_first_holdout_day_without_numpy_timedelta_deprecation():
    from backtest import first_holdout_day
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        assert first_holdout_day(np.datetime64('2026-04-01')).isoformat() == '2026-04-02'
    assert first_holdout_day('2026-12-31').isoformat() == '2027-01-01'
