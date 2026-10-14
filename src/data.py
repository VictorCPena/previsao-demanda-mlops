"""Demanda sintética diária: calendário e promoções conhecidas na previsão."""
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["product_code", "day_of_week", "month", "trend_day", "promo",
            "lag_1", "lag_7", "lag_14", "lag_28", "mean_7", "mean_28"]
ORIGIN = pd.Timestamp("2024-01-01")
DAY_NS = 86_400_000_000_000  # Um dia exato em nanossegundos, sem conversão genérica do NumPy.


def consecutive_days(values) -> bool:
    """Compara datas normalizadas via inteiros ns; evita warnings do NumPy 2/Pandas 2.

    A ordenação é exigida: períodos duplicados, vazios e lacunas devem ser rejeitados.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(values))
    return bool(len(idx) > 0 and not idx.hasnans and idx.is_monotonic_increasing
                and np.all(np.diff(idx.asi8) == DAY_NS))


def generate(days=730, products=6, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(products):
        for t, date in enumerate(pd.date_range(ORIGIN, periods=days)):
            promo = int((t + p*3) % 21 < 3)
            weekly = [0.85, .90, .95, 1., 1.15, 1.45, 1.2][date.dayofweek]
            yearly = 1 + .16*np.sin(2*np.pi*t/365)
            expected = (15 + p*7 + .025*t)*weekly*yearly*(1 + .65*promo)
            rows.append({"day": date, "product_id": f"P{p:02}",
                         "units": int(rng.poisson(expected)), "promo": promo})
    return pd.DataFrame(rows).sort_values(["product_id", "day"]).reset_index(drop=True)


def validate_daily(frame):
    required = {"day", "product_id", "units", "promo"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Colunas necessárias: {required}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Dados ausentes")
    if frame.duplicated(["day", "product_id"]).any():
        raise ValueError("Mais de uma observação por produto/dia")
    if not np.isfinite(frame.units).all() or (frame.units < 0).any():
        raise ValueError("Demanda inválida")
    if not frame.promo.isin([0, 1]).all():
        raise ValueError("promo precisa ser 0 ou 1")
    dates = pd.to_datetime(frame.day)
    if (dates != dates.dt.normalize()).any():
        raise ValueError("Use datas diárias, sem horários")
    for _, group in frame.assign(day=dates).groupby("product_id"):
        if not consecutive_days(group.day.sort_values()):
            raise ValueError("Série diária tem lacunas; não preencher zeros sem verificar a fonte")


def features(frame):
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame.day)
    validate_daily(frame)
    frame = frame.sort_values(["product_id", "day"])
    frame["product_code"] = frame.product_id.str.removeprefix("P").astype(int)
    frame["day_of_week"] = frame.day.dt.dayofweek
    frame["month"] = frame.day.dt.month
    frame["trend_day"] = (frame.day-ORIGIN).dt.days
    grouped = frame.groupby("product_id")["units"]
    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 28):
        frame[f"mean_{window}"] = grouped.transform(lambda s: s.shift(1).rolling(window).mean())
    frame[FEATURES] = frame[FEATURES].astype('float64')
    return frame.dropna(subset=FEATURES).reset_index(drop=True)


def temporal_folds(frame, folds=3, validation_days=42, test_days=90):
    """Splits por data global: todos os produtos do mesmo dia ficam juntos."""
    days = pd.DatetimeIndex(frame.day.unique()).sort_values()
    if len(days) < test_days + folds*validation_days + 90:
        raise ValueError("Poucos dias para os folds temporais e holdout")
    test_start = days[-test_days]
    development = frame[frame.day < test_start]
    dev_days = pd.DatetimeIndex(development.day.unique()).sort_values()
    splits = []
    for i in range(folds):
        start = dev_days[-(folds-i)*validation_days]
        end = pd.Timestamp(start.date() + timedelta(days=int(validation_days)))
        train = development[development.day < start]
        valid = development[(development.day >= start) & (development.day < end)]
        splits.append((train, valid))
    return splits, development, frame[frame.day >= test_start]


def next_features(product_id, history, day, promo):
    dates = pd.DatetimeIndex(pd.to_datetime([h["day"] for h in history]))
    values = np.array([h["units"] for h in history], dtype=float)
    day = pd.Timestamp(day)
    if len(values) < 28 or len(set(dates)) != len(dates):
        raise ValueError("Envie pelo menos 28 dias históricos únicos")
    if not consecutive_days(dates):
        raise ValueError("Histórico precisa estar ordenado e ser diário, sem lacunas")
    if dates[-1] != pd.Timestamp(day.date() - timedelta(days=1)):
        raise ValueError("Histórico deve terminar no dia anterior à previsão")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demanda histórica inválida")
    if promo not in (0, 1):
        raise ValueError("Promoção inválida")
    row = {"product_code": int(product_id.removeprefix("P")), "day_of_week": day.dayofweek,
           "month": day.month, "trend_day": (day-ORIGIN).days, "promo": promo,
           "lag_1": values[-1], "lag_7": values[-7], "lag_14": values[-14], "lag_28": values[-28],
           "mean_7": values[-7:].mean(), "mean_28": values[-28:].mean()}
    return pd.DataFrame([row], columns=FEATURES, dtype='float64')
