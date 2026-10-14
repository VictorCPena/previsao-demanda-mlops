"""Backtesting por origem móvel com rollout realmente recursivo (sem ground truth futuro).

O modelo é selecionado e treinado ANTES do holdout. Nenhum dado futuro entra nas
features; apenas a agenda de promoções, assumida conhecida no instante da origem.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from data import ROOT, next_features, validate_daily


def rollout(product_id, history, start_day, promotions, predictor):
    """Mesma regra de recursão da API; predictor(product_id, history, day, promo)."""
    history = [dict(p) for p in history]  # isolar estado do chamador
    if not promotions or any(p not in (0, 1) for p in promotions):
        raise ValueError('Preveja ao menos um dia, com promoções binárias conhecidas')
    predictions = []
    for index, promo in enumerate(promotions):
        day = pd.Timestamp(start_day).date() + timedelta(days=index)
        value = float(predictor(product_id, history, day, promo))
        if not np.isfinite(value):
            raise ValueError('Modelo produziu previsão não finita')
        value = max(0.0, value)
        predictions.append({'day': day.isoformat(), 'prediction': value})
        history.append({'day': day.isoformat(), 'units': value})
    return predictions


def seasonal_predictor(_product_id, history, _day, _promo):
    return float(history[-7]['units'])


def score(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    estimated = np.asarray(y_pred, dtype=float)
    diff = estimated - actual
    total = float(actual.sum())
    return {'n': int(len(actual)), 'mae': float(np.abs(diff).mean()),
            'rmse': float(np.sqrt(np.mean(diff**2))),
            'bias': float(diff.mean()),
            'wape': float(np.abs(diff).sum()/total) if total else None}


def backtest(raw, predictor, holdout_start, horizons=(1, 7, 14), stride=7, products=None):
    """Rolling origins em holdout, com janela de 28 dias reais ANTERIORES a cada origem.

    Atualizar histórico real ENTRE origens é permitido (simula implantação diária);
    dentro de uma origem, D+2...D+H recebem exclusivamente previsões anteriores.
    """
    raw = raw.copy()
    raw['day'] = pd.to_datetime(raw.day)
    validate_daily(raw)
    if stride < 1 or not horizons or min(horizons) < 1 or len(set(horizons)) != len(horizons):
        raise ValueError('Stride/horizontes inválidos')
    horizons = sorted(int(h) for h in horizons)
    if max(horizons) > 30:
        raise ValueError('Este protocolo suporta até 30 dias')
    holdout_start = pd.Timestamp(holdout_start)
    rows = []
    available = set(products) if products is not None else set(raw.product_id.unique())
    if not available.issubset(set(raw.product_id.unique())):
        raise ValueError('Produto não encontrado')
    for product_id, group in raw.groupby('product_id'):
        if product_id not in available:
            continue
        group = group.sort_values('day').reset_index(drop=True)
        days = group['day']
        first = int(days.searchsorted(holdout_start))
        for origin_index in range(first, len(group)-max(horizons)+1, stride):
            if origin_index < 28:
                continue
            origin = pd.Timestamp(group.iloc[origin_index].day)
            history = [{'day': pd.Timestamp(row.day).date().isoformat(), 'units': float(row.units)}
                       for row in group.iloc[origin_index-28:origin_index].itertuples()]
            future = group.iloc[origin_index:origin_index+max(horizons)]
            promotions = [int(x) for x in future.promo]
            predictions = rollout(product_id, history, origin, promotions, predictor)
            naive = rollout(product_id, history, origin, promotions, seasonal_predictor)
            for step, (predicted, baseline, actual) in enumerate(zip(predictions, naive, future.itertuples()), start=1):
                rows.append({'origin': origin.date().isoformat(), 'day': predicted['day'],
                             'product_id': product_id, 'horizon_step': step,
                             'actual': float(actual.units), 'prediction': predicted['prediction'],
                             'baseline': baseline['prediction']})
    if not rows:
        raise ValueError('Sem origens válidas: revise holdout/horizonte/histórico')
    frame = pd.DataFrame(rows)
    report = {'protocol': 'rolling-origin recursive; observed history only before origin; known planned promotions',
              'holdout_start': holdout_start.date().isoformat(), 'stride_days': stride,
              'horizons': horizons, 'origin_count': int(frame[['product_id','origin']].drop_duplicates().shape[0]),
              'per_step': {}, 'at_horizon': {}, 'through_horizon': {}, 'by_product_at_horizon': {},
              'limitations': ['Modelo selecionado por CV antes de avaliar holdout; métricas não são CV.',
                              'Entre origens os valores reais tornam-se observáveis; dentro de cada origem, rollout só usa suas próprias previsões.',
                              'Promoções futuras simuladas como conhecidas; incerteza das promoções não está incluída.',
                              'Origens se sobrepõem: amostras não são independentes e IC ingênuo seria incorreto.',
                              'Nenhum retreino ocorre durante este backtest.',
                              'Intervalos residuais de 1 dia não são válidos como intervalos de H dias.']}
    for step, part in frame.groupby('horizon_step'):
        report['per_step'][str(step)] = {'model': score(part.actual,part.prediction),
                                        'seasonal_naive': score(part.actual,part.baseline)}
    for horizon in horizons:
        target = frame[frame.horizon_step == horizon]
        through = frame[frame.horizon_step <= horizon]
        report['at_horizon'][str(horizon)] = {'model': score(target.actual,target.prediction),
                                              'seasonal_naive': score(target.actual,target.baseline)}
        report['through_horizon'][str(horizon)] = {'model': score(through.actual,through.prediction),
                                                   'seasonal_naive': score(through.actual,through.baseline)}
        report['by_product_at_horizon'][str(horizon)] = {
            str(p): {'model': score(g.actual,g.prediction), 'seasonal_naive': score(g.actual,g.baseline)}
            for p,g in target.groupby('product_id')}
    return frame, report


def first_holdout_day(train_end):
    """Dia seguinte ao corte de treino, sem timedelta genérico NumPy/pandas."""
    return pd.Timestamp(train_end).date() + timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'data/demanda_sintetica.csv')
    parser.add_argument('--workspace', type=Path, default=ROOT/'work')
    parser.add_argument('--output', type=Path, default=ROOT/'work/evaluation')
    parser.add_argument('--stride', type=int, default=7)
    parser.add_argument('--horizons', type=int, nargs='+', default=[1,7,14])
    args = parser.parse_args()
    deployment = json.loads((args.workspace/'deployment.json').read_text())
    artifact = (args.workspace/deployment['active']['file']).resolve()
    if not artifact.is_relative_to(args.workspace.resolve()):
        raise ValueError('Artefato fora do workspace')
    bundle = joblib.load(artifact)  # apenas arquivo local de treinamento próprio
    if bundle.get('train_end') is None:
        raise ValueError('Artefato não contém metadados de corte temporal')
    start = first_holdout_day(bundle['train_end'])
    raw = pd.read_csv(args.input, parse_dates=['day'])
    raw_hash = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    if bundle.get('data_hash') and raw_hash != bundle['data_hash']:
        raise ValueError('Dataset diferente do usado no treino. Faça novo treinamento antes do backtest.')
    if set(raw.product_id.unique()) - set(bundle['products']):
        raise ValueError('Produtos fora do treinamento')
    def predict(product_id, history, day, promo):
        X = next_features(product_id, history, day, promo)
        with threadpool_limits(limits=2):
            return bundle['model'].predict(X)[0]
    detail, report = backtest(raw, predict, start, horizons=args.horizons, stride=args.stride)
    report.update({'model':bundle['name'], 'version':bundle['version'],
                   'synthetic_data':bundle['synthetic'], 'data_sha256':raw_hash})
    args.output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output/'backtest_recursivo_linhas.csv',index=False)
    (args.output/'backtest_recursivo.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'origins':report['origin_count'], 'at_horizon':report['at_horizon']},indent=2,ensure_ascii=False))

if __name__ == '__main__':
    main()
