"""Diagnóstico de holdout por produto; não escolhe nem retreina o modelo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import ROOT

REQUIRED = {"day", "product_id", "actual", "prediction", "baseline"}


def evaluate(frame: pd.DataFrame) -> dict:
    if not REQUIRED.issubset(frame.columns):
        raise ValueError(f"Colunas faltando: {sorted(REQUIRED - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Holdout vazio")
    frame = frame.copy()
    frame["day"] = pd.to_datetime(frame["day"], errors="raise")
    if frame[["day", "product_id", "actual", "prediction", "baseline"]].isna().any().any():
        raise ValueError("Holdout com valores ausentes")
    if frame.duplicated(["day", "product_id"]).any():
        raise ValueError("Há produto/dia repetido no holdout")
    if frame["product_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Identificador de produto vazio")
    for col in ("actual", "prediction", "baseline"):
        values = pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{col} deve ser finito e não negativo")
        frame[col] = values

    def scores(part):
        actual = part.actual.to_numpy(dtype=float)
        predicted = part.prediction.to_numpy(dtype=float)
        baseline = part.baseline.to_numpy(dtype=float)
        error = predicted - actual
        mae = float(np.abs(error).mean())
        base_mae = float(np.abs(baseline-actual).mean())
        denominator = float(actual.sum())
        return {"observations": len(part), "mae": mae,
                "baseline_mae": base_mae, "bias_units": float(error.mean()),
                "wape": float(np.abs(error).sum()/denominator) if denominator else None,
                "improvement_vs_baseline_pct": float(100*(1-mae/base_mae)) if base_mae else None,
                "beaten_by_baseline": mae > base_mae}

    per_product = {str(product): scores(part) for product, part in frame.groupby("product_id", sort=True)}
    total = scores(frame)
    total["products"] = len(per_product)
    return {"protocol": "Holdout rolling de 1 dia; demanda D-1 real já observada.",
            "overall": total, "by_product": per_product,
            "products_worse_than_baseline": sorted([p for p, r in per_product.items() if r["beaten_by_baseline"]]),
            "limits": ["Análise por grupo descritiva, feita depois da seleção do campeão.",
                       "Não constitui validação de previsão recursiva a 7 ou 14 dias.",
                       "O WAPE é indefinido quando a demanda total do grupo é zero."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "work/evaluation/previsoes_holdout.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "work/evaluation/avaliacao_por_produto.json")
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
