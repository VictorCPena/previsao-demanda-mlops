"""Drift de atributos e erro observado: sinalizam investigação, sem retreino automático."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from data import ROOT


def psi(reference, current):
    reference, current = np.asarray(reference), np.asarray(current)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 2:
        return float(0 if np.array_equal(np.unique(reference), np.unique(current)) else 1)
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0]/len(reference)
    observed = np.histogram(current, edges)[0]/len(current)
    expected, observed = np.clip(expected, 1e-6, 1), np.clip(observed, 1e-6, 1)
    return float(((observed-expected)*np.log(observed/expected)).sum())


def drift(reference, current):
    results = []
    # Calendário/trend são excluídos: sua distribuição muda por construção.
    columns = ["lag_1", "lag_7", "mean_7", "mean_28"]
    for column in columns:
        value = psi(reference[column], current[column])
        statistic = ks_2samp(reference[column], current[column])
        results.append({"feature": column, "psi": value, "ks_statistic": float(statistic.statistic),
                        "ks_p_descriptive": float(statistic.pvalue), "review": value > .2})
    return {"features": results, "review_needed": any(r["review"] for r in results),
            "policy": "PSI > 0.2 é heurística; KS descritivo, séries dependentes. Investigar antes de retreinar."}


def main():
    work = ROOT / "work"
    reference = pd.read_csv(work / "reference_features.csv")
    current = pd.read_csv(work / "current_features.csv")
    result = {"normal_window": drift(reference, current)}
    stressed = current.copy()
    for column in ("lag_1", "lag_7", "mean_7", "mean_28"):
        stressed[column] *= 2.5
    result["simulated_shift"] = drift(reference, stressed)
    predictions = pd.read_csv(ROOT / "resultados/previsoes_holdout.csv", parse_dates=["day"])
    days = sorted(predictions.day.unique())
    middle = days[len(days)//2]
    early, late = predictions[predictions.day < middle], predictions[predictions.day >= middle]
    early_mae = float(np.abs(early.actual-early.prediction).mean())
    late_mae = float(np.abs(late.actual-late.prediction).mean())
    result["observed_error"] = {"reference_mae": early_mae, "current_mae": late_mae,
        "ratio": late_mae/max(early_mae, 1e-9), "review_needed": late_mae > 1.25*early_mae,
        "note": "Erro só pode ser monitorado depois da chegada dos valores reais."}
    (ROOT / "resultados/monitoramento.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
