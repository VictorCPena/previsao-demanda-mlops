import json
from datetime import timedelta
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from uncertainty import scenario_plans, forecast_scenarios, calibration_report
from data import generate, features, FEATURES
from models import SeasonalNaive
from serve import create_app
from local_test_client import LocalTestClient
import joblib


def simple(_id, history, day, promo):
    return history[-1]['units']+5*promo


def test_scenarios_horizon_and_nonmutation():
    history=[{'day':str(x.date()), 'units':10.} for x in pd.date_range('2026-01-01',periods=28)]
    result=forecast_scenarios('P00',history,'2026-01-29',simple,horizon=3)
    assert len(result['by_day'])==3
    assert result['by_day'][0]['min_scenario']==10
    assert result['by_day'][0]['max_scenario']==15
    assert len(history)==28
    assert result['by_day'][0]['weighted_prediction']==12.5


def test_scenario_rejects_invalid_weights_and_promotions():
    with pytest.raises(ValueError):scenario_plans(2,[{'name':'A','promotions':[0,0],'weight':.7}])
    with pytest.raises(ValueError):scenario_plans(2,[{'name':'A','promotions':[0,2],'weight':1}])
    with pytest.raises(ValueError):scenario_plans(15)


def fake_detail(origins=15, products=2, horizon=7):
    rows=[]
    for i, date in enumerate(pd.date_range('2026-01-01',periods=origins,freq='7D')):
        for p in range(products):
            for h in range(1,horizon+1):
                rows.append({'origin':str(date.date()),'product_id':f'P{p:02}',
                   'horizon_step':h,'actual':30+(i%3)+p+h*.4,
                   'prediction':30+p+h*.2})
    return pd.DataFrame(rows)


def test_calibration_nonoverlap_and_coverage():
    detail=fake_detail()
    cal=calibration_report(detail,max_horizon=7,embargo_days=7)
    assert set(cal['per_step'])==set(str(i) for i in range(1,8))
    assert pd.Timestamp(cal['evaluation_start_origin'])>pd.Timestamp(cal['calibration_cutoff_origin'])
    assert all(0<=r['empirical_coverage']<=1 for r in cal['per_step'].values())
    assert cal['per_step']['7']['calibration_n']>0


def test_calibration_rejects_leakage_and_short():
    with pytest.raises(ValueError):calibration_report(fake_detail(),max_horizon=7,embargo_days=6)
    with pytest.raises(ValueError):calibration_report(fake_detail(origins=3),max_horizon=3,embargo_days=3)


def test_api_scenario_route(monkeypatch,tmp_path):
    from sklearn.dummy import DummyRegressor
    frame=generate(days=80,products=1)
    x=features(frame)[FEATURES]
    model=DummyRegressor(strategy='constant',constant=15).fit(x,np.ones(len(x)))
    joblib.dump({'model':model,'name':'dummy','products':['P00'],'q90':1,
       'version':'test','synthetic':True},tmp_path/'bundle.joblib')
    (tmp_path/'deployment.json').write_text(json.dumps({'active':{'file':'bundle.joblib'}}))
    c=LocalTestClient(create_app(tmp_path))
    hist=[{'day':str(row.day.date()),'units':float(row.units)} for row in frame.tail(28).itertuples()]
    body={'product_id':'P00','day':(pd.Timestamp(frame.day.max()).date() + timedelta(days=1)).isoformat(),
          'promo':0,'history':hist,'horizon':3}
    response=c.post('/forecast/scenarios',json=body)
    assert response.status_code==200
    assert response.json()['by_day'][0]['weighted_prediction']==15
    assert c.get('/live').status_code==200
    bad={**body,'scenarios':[{'name':'wrong','weight':.9,'promotions':[0,0,0]}]}
    assert c.post('/forecast/scenarios',json=bad).status_code==422
