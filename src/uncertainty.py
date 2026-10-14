"""Incerteza de promoções (cenários) e calibração temporal de erros recursivos.

Faixa entre cenários != intervalo probabilístico. Intervalos por passo são empíricos,
com holdout posterior separado por embargo; não há garantia de cobertura sob drift.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import backtest, rollout
from data import ROOT


def scenario_plans(horizon: int, scenarios=None):
    if not 1 <= horizon <= 14:
        raise ValueError('Horizonte deve estar entre 1 e 14 dias')
    if scenarios is None:
        scenarios = [
            {'name': 'sem_promocao', 'weight': .5, 'promotions': [0]*horizon},
            {'name': 'promocao_primeiro_dia', 'weight': .25, 'promotions': [1]+[0]*(horizon-1)},
            {'name': 'promocao_todos_dias', 'weight': .25, 'promotions': [1]*horizon},
        ]
    if not 1 <= len(scenarios) <= 8:
        raise ValueError('Informe 1 a 8 cenários')
    names = [s['name'] for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError('Cenários devem ter nomes distintos')
    for s in scenarios:
        if (not isinstance(s.get('name'), str) or not s['name']
                or len(s.get('promotions', [])) != horizon
                or any(p not in (0,1) for p in s['promotions'])
                or not math.isfinite(float(s.get('weight', 0)))
                or s['weight'] <= 0):
            raise ValueError('Cenário inválido')
    total = sum(float(s['weight']) for s in scenarios)
    if not np.isclose(total, 1, atol=1e-8):
        raise ValueError('Pesos precisam somar 1')
    return scenarios


def forecast_scenarios(product_id, history, start_day, predictor, scenarios=None, horizon=7):
    planned = scenario_plans(horizon, scenarios)
    results = {s['name']: rollout(product_id, history, start_day, s['promotions'], predictor) for s in planned}
    matrix = np.array([[p['prediction'] for p in results[s['name']]] for s in planned], dtype=float)
    weights = np.array([s['weight'] for s in planned], dtype=float)
    mean = weights @ matrix
    return {'horizon': horizon, 'scenarios': [{'name': s['name'], 'weight': s['weight'],
        'promotions': s['promotions'], 'predictions': results[s['name']]} for s in planned],
        'by_day': [{'day': results[planned[0]['name']][j]['day'],
            'weighted_prediction': float(mean[j]), 'min_scenario': float(matrix[:,j].min()),
            'max_scenario': float(matrix[:,j].max())} for j in range(horizon)],
        'note': 'Pesos ilustrativos, não probabilidades aprendidas. Min/max mede apenas sensibilidade a promoções, não incerteza total.'}


def calibration_report(detail: pd.DataFrame, max_horizon=14, alpha=.10, fraction=.55, embargo_days=14):
    """Purge temporal por ORIGEM para reduzir compartilhamento de alvos entre conjuntos.

    q finito via ceil((n+1)(1-alpha))/n. Cada origem-produto pode ser correlacionada,
    logo cobertura é empírica; sem alegação de conformal distribution-free no tempo.
    """
    if not 0 < alpha < 1 or not .2 <= fraction <= .8 or not 1 <= max_horizon <= 30:
        raise ValueError('Parâmetros de calibração inválidos')
    if embargo_days < max_horizon:
        raise ValueError('Embargo deve ser ao menos o horizonte máximo')
    required = {'origin','horizon_step','actual','prediction'}
    if not required.issubset(detail.columns):
        raise ValueError('Colunas de backtest ausentes')
    detail = detail.copy()
    detail['origin'] = pd.to_datetime(detail.origin)
    origins = pd.DatetimeIndex(sorted(detail.origin.unique()))
    if len(origins) < 6:
        raise ValueError('São necessárias ao menos seis origens')
    cutoff = origins[max(1, int(len(origins)*fraction))-1]
    evaluation_start = pd.Timestamp(cutoff.date() + timedelta(days=embargo_days+1))
    cal = detail[detail.origin <= cutoff]
    test = detail[detail.origin >= evaluation_start]
    steps = {}
    for step in range(1, max_horizon+1):
        prior = cal[cal.horizon_step == step]
        later = test[test.horizon_step == step]
        n = len(prior)
        if n < 5 or later.empty:
            raise ValueError(f'Amostras insuficientes para D+{step}: {n} calibração, {len(later)} avaliação')
        errors = np.sort(np.abs(prior.actual.to_numpy()-prior.prediction.to_numpy()))
        rank = int(math.ceil((n+1)*(1-alpha)))
        width = float(errors[rank-1]) if rank <= n else float('inf')
        if not np.isfinite(width):
            raise ValueError('Amostra insuficiente para quantil finito nesta confiança')
        observed_error = np.abs(later.actual.to_numpy()-later.prediction.to_numpy())
        lower = np.maximum(0, later.prediction.to_numpy()-width)
        upper = later.prediction.to_numpy()+width
        steps[str(step)] = {'calibration_n': n, 'evaluation_n': int(len(later)),
            'absolute_error_quantile': width,
            'empirical_coverage': float(np.mean((later.actual.to_numpy()>=lower)&(later.actual.to_numpy()<=upper))),
            'mean_interval_width': float(np.mean(upper-lower)),
            'evaluation_mae': float(np.mean(observed_error))}
    return {'alpha': alpha, 'nominal_coverage': 1-alpha,
            'calibration_cutoff_origin': cutoff.date().isoformat(),
            'evaluation_start_origin': evaluation_start.date().isoformat(),
            'embargo_days': embargo_days, 'per_step': steps,
            'limitations': ['Avaliação posterior separada por embargo; janelas de rolling origin ainda dependentes.',
                'Quantis residuais temporais não asseguram cobertura futura sob não-estacionariedade.',
                'Cobertura marginal agregada por produto, não garante cobertura condicional.',
                'Não inclui incerteza sobre promoções salvo via cenários separados.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'work/evaluation/backtest_recursivo_linhas.csv')
    p.add_argument('--output',type=Path,default=ROOT/'work/evaluation/intervalos_multihorizonte.json')
    p.add_argument('--max-horizon',type=int,default=14)
    p.add_argument('--embargo',type=int,default=14)
    a=p.parse_args()
    data=pd.read_csv(a.input)
    report=calibration_report(data,max_horizon=a.max_horizon,embargo_days=a.embargo)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'steps':len(report['per_step']), 'D1':report['per_step']['1']},indent=2))

if __name__=='__main__':main()
