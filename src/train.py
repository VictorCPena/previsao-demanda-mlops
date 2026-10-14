"""Treino com seleção em CV temporal, holdout intocado e MLflow local."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ENABLE_TELEMETRY", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

from data import ROOT, FEATURES, generate, features, temporal_folds
from models import candidates


def metrics(y, predictions):
    y, predictions = np.asarray(y), np.maximum(0, predictions)
    return {"mae": float(mean_absolute_error(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "wape": float(np.abs(y-predictions).sum()/max(np.abs(y).sum(), 1))}


def configure_mlflow(work):
    work.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + str((work / "mlflow.db").resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    name = "demanda-diaria-local"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(work / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)
    return uri


def train(input_csv=None, work=None, output=None):
    work = Path(work or ROOT / "work")
    output = Path(output or ROOT / "resultados")
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if input_csv:
        raw = pd.read_csv(input_csv, parse_dates=["day"])
        synthetic = "external_unverified"  # Procedência externa deve ser documentada pelo usuário.
    else:
        raw = generate()
        synthetic = True
        (ROOT / "data").mkdir(exist_ok=True)
        raw.to_csv(ROOT / "data/demanda_sintetica.csv", index=False)
    frame = features(raw)
    folds, development, test = temporal_folds(frame)
    data_hash = hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest()
    cv_rows, oof, trained = [], {}, {}
    with threadpool_limits(limits=2):
        for name, estimator in candidates().items():
            preds = []
            for fold, (train_data, valid) in enumerate(folds, start=1):
                model = clone(estimator).fit(train_data[FEATURES], train_data.units)
                predicted = np.maximum(0, model.predict(valid[FEATURES]))
                preds.append(pd.DataFrame({"day": valid.day, "product_id": valid.product_id,
                                           "actual": valid.units, "prediction": predicted}))
                cv_rows.append({"model": name, "fold": fold,
                    "train_end": str(train_data.day.max().date()),
                    "valid_start": str(valid.day.min().date()), "valid_end": str(valid.day.max().date()),
                    **metrics(valid.units, predicted)})
            oof[name] = pd.concat(preds)
            trained[name] = clone(estimator).fit(development[FEATURES], development.units)
    cv = pd.DataFrame(cv_rows)
    scores = cv.groupby("model").mae.mean().sort_values()
    winner = scores.index[0]  # Escolhido antes de calcular qualquer métrica do holdout.
    champion = trained[winner]
    cv.to_csv(output / "validacao_temporal.csv", index=False)
    # Faixa empírica a partir dos resíduos de validação; cobertura não é garantia.
    residuals = np.abs(oof[winner].actual-oof[winner].prediction)
    q90 = float(np.quantile(residuals, .9, method="higher"))
    uri = configure_mlflow(work)
    registry = MlflowClient()
    model_versions, test_predictions, test_scores = {}, {}, {}
    for name, model in trained.items():
        with threadpool_limits(limits=2):
            prediction = np.maximum(0, model.predict(test[FEATURES]))
        score = metrics(test.units, prediction)
        test_scores[name] = score
        test_predictions[name] = prediction
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"data_sha256": data_hash, "synthetic_data": str(synthetic),
                             "validation": "expanding-window-by-date", "horizon": "rolling-one-day",
                             "selection": "mean_cv_mae_before_holdout"})
            mlflow.log_params({"model": name, "train_rows": len(development), "test_rows": len(test),
                               "features": ",".join(FEATURES), "seed": 42})
            mlflow.log_metrics({"cv_mae": float(scores[name]), **{"test_"+k: v for k, v in score.items()}})
            mlflow.log_artifact(str(output / "validacao_temporal.csv"))
            info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle",
                code_paths=[str(ROOT / "src/models.py")],
                signature=infer_signature(development[FEATURES].head(5), model.predict(development[FEATURES].head(5))),
                input_example=development[FEATURES].head(2),
                pip_requirements=["scikit-learn>=1.5,<2", "pandas>=2.2,<3", "numpy>=1.26,<3"])
            version = mlflow.register_model(info.model_uri, "demanda-diaria")
            artifact_name = f"model_v{version.version}.joblib"
            model_q90 = float(np.quantile(np.abs(oof[name].actual-oof[name].prediction), .9, method="higher"))
            bundle = {"model": model, "name": name, "features": FEATURES, "q90": model_q90,
                      "products": sorted(raw.product_id.unique()), "data_hash": data_hash,
                      "version": str(version.version), "synthetic": synthetic,
                      "train_end": str(development.day.max().date()), "horizon": "rolling-one-day"}
            joblib.dump(bundle, work / artifact_name)
            model_versions[name] = {"version": str(version.version), "file": artifact_name, "run_id": run.info.run_id}
    registry.set_registered_model_alias("demanda-diaria", "champion", model_versions[winner]["version"])
    deployment = {"active": model_versions[winner], "winner": winner, "versions": model_versions,
                  "tracking_uri": uri, "policy": "Champion selecionado pela CV; holdout apenas para avaliação final."}
    temp = work / "deployment.tmp.json"
    temp.write_text(json.dumps(deployment, indent=2))
    temp.replace(work / "deployment.json")
    predictions = test[["day", "product_id", "units"]].copy().rename(columns={"units": "actual"})
    predictions["prediction"] = test_predictions[winner]
    predictions["baseline"] = test_predictions["seasonal_naive"]
    predictions["lower90_empirical"] = np.maximum(0, predictions.prediction-q90)
    predictions["upper90_empirical"] = predictions.prediction+q90
    predictions.to_csv(output / "previsoes_holdout.csv", index=False)
    coverage = ((predictions.actual >= predictions.lower90_empirical) &
                (predictions.actual <= predictions.upper90_empirical)).mean()
    summary = {"synthetic_data": synthetic, "data_sha256": data_hash,
        "winner_selected_by_cv": winner, "cv_mean_mae": scores.to_dict(), "holdout": test_scores,
        "holdout_start": str(test.day.min().date()), "holdout_end": str(test.day.max().date()),
        "train_end": str(development.day.max().date()), "rows": len(frame),
        "improvement_vs_baseline_mae_pct": float(100*(1-test_scores[winner]["mae"]/test_scores["seasonal_naive"]["mae"])),
        "empirical_90_interval_holdout_coverage": float(coverage), "q90": q90,
        "evaluation_protocol": "Rolling 1 dia: demanda real de D-1 fica disponível antes da previsão de D.",
        "multi_day_limit": "Endpoint recursivo não foi validado para 7/14 dias; não reutilizar métricas de 1 dia.",
        "versions": model_versions}
    (output / "treinamento.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    development.loc[development.day >= development.day.max()-pd.Timedelta(days=99), FEATURES].to_csv(work / "reference_features.csv", index=False)
    test.loc[test.day >= test.day.max()-pd.Timedelta(days=29), FEATURES].to_csv(work / "current_features.csv", index=False)
    example_product = raw.product_id.iloc[0]
    history = raw[raw.product_id == example_product].tail(28)
    request = {"product_id": example_product, "day": str((raw.day.max()+pd.Timedelta(days=1)).date()),
               "promo": 0, "history": [{"day": str(r.day.date()), "units": int(r.units)}
                                         for r in history.itertuples()]}
    (output / "exemplo_requisicao.json").write_text(json.dumps(request, indent=2))
    print(json.dumps({"winner": winner, "holdout": test_scores[winner],
                      "improvement_pct": summary["improvement_vs_baseline_mae_pct"]}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    train(args.input)
