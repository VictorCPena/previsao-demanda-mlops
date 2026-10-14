"""Verifica os modelos realmente treinados, API e promoção/rollback local."""
import json

from local_test_client import LocalTestClient
from mlflow.tracking import MlflowClient

from data import ROOT
from promote import promote
from serve import create_app


def main():
    workspace = ROOT/'work'
    deployment = json.loads((workspace/'deployment.json').read_text())
    champion = deployment['winner']
    request = json.loads((ROOT/'work/evaluation/exemplo_requisicao.json').read_text())
    client = LocalTestClient(create_app())
    first = client.post('/predict',json=request)
    assert first.status_code == 200, first.text
    recursive = client.post('/forecast',json={**request,'horizon':7,'planned_promotions':[0]*7})
    assert recursive.status_code == 200, recursive.text
    assert len(recursive.json()['predictions']) == 7
    try:
        promote('seasonal_naive')
        fallback = client.post('/predict',json=request)
        assert fallback.status_code == 200
        assert fallback.json()['model'] == 'seasonal_naive'
    finally:
        promote(champion)
    restored = client.post('/predict',json=request)
    assert restored.json()['version'] == first.json()['version']
    # Testa também o modelo serializado no MLflow, além do joblib da API.
    import mlflow.pyfunc
    from data import next_features
    X = next_features(request['product_id'],request['history'],request['day'],request['promo'])
    mlflow_model = mlflow.pyfunc.load_model(f"models:/demanda-diaria@champion")
    import numpy as np
    np.testing.assert_allclose(mlflow_model.predict(X),[first.json()['prediction']])
    result = {'actual_trained_model_api':True,'recursive_endpoint':True,'promotion_and_rollback':True,
              'mlflow_registered_model_loaded':True,'first_prediction':first.json(),
              'recursive_demo':recursive.json()}
    (ROOT/'work/evaluation/api_smoke.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('API com modelo treinado, registry, promoção e rollback: OK')


if __name__=='__main__':
    main()
