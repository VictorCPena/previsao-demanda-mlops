"""Direct multi-horizon training; temporal selection before a sealed holdout."""

import os

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
os.environ["DO_NOT_TRACK"] = "true"
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from forecasting.data import features, FEATURES
from forecasting.models import ColumnBaseline, TorchMLP, SklearnMLP, neural_backend


def metrics(y, p):
    y = np.asarray(y)
    p = np.maximum(0, p)
    return {
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "wape": float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1)),
    }


def candidates():
    backend = neural_backend()
    return {
        "seasonal_naive": ColumnBaseline(),
        "persistence": ColumnBaseline("lag_origin"),
        "moving_average": ColumnBaseline("mean24"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "random_forest": RandomForestRegressor(
            n_estimators=60, min_samples_leaf=8, max_depth=12, n_jobs=2, random_state=42
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, l2_regularization=2, random_state=42
        ),
        f"{backend}_mlp": TorchMLP() if backend == "torch" else SklearnMLP(),
    }


def atomic_json(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def train(input_path: Path, work: Path):
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.telemetry.utils import is_telemetry_disabled

    if not is_telemetry_disabled():
        raise RuntimeError("MLflow telemetry must be disabled for local execution")

    work.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(input_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    local_uri = "sqlite:///" + str((work / "mlflow.db").resolve())
    mlflow.set_tracking_uri(local_uri)
    mlflow.set_registry_uri(local_uri)
    mlflow.set_experiment("tlc-direct-forecast")
    client = MlflowClient()
    results = []
    predictions = []
    bundles = {}
    versions = {}
    end = raw.time.max()
    holdout_start = end - pd.Timedelta(days=14) + pd.Timedelta(hours=1)
    with threadpool_limits(limits=2):
        for h in (1, 6, 24):
            f = features(raw, h)
            dev = f[f.time < holdout_start - pd.Timedelta(hours=h)]
            test = f[f.time >= holdout_start]
            folds = []
            for days in (14, 7):
                start = holdout_start - pd.Timedelta(days=days)
                stop = start + pd.Timedelta(days=7)
                tr = f[f.time < start - pd.Timedelta(hours=h)]
                val = f[(f.time >= start) & (f.time < stop)]
                folds.append((tr, val))
            cv = {}
            q = {}
            for name, base in candidates().items():
                errors = []
                scores = []
                for tr, val in folds:
                    model = clone(base).fit(tr[FEATURES], tr["count"])
                    p = np.maximum(0, model.predict(val[FEATURES]))
                    errors.extend(np.abs(val["count"] - p))
                    scores.append(metrics(val["count"], p)["mae"])
                cv[name] = float(np.mean(scores))
                q[name] = float(np.quantile(errors, 0.9, method="higher"))
            winner = min(cv, key=cv.get)
            for name, base in candidates().items():
                model = clone(base).fit(dev[FEATURES], dev["count"])
                p = np.maximum(0, model.predict(test[FEATURES]))
                score = metrics(test["count"], p)
                with mlflow.start_run(run_name=f"{name}-h{h}") as run:
                    mlflow.log_params(
                        {
                            "horizon_hours": h,
                            "model": name,
                            "data_sha256": digest,
                            "train_rows": len(dev),
                            "seed": 42,
                        }
                    )
                    mlflow.log_metrics({"cv_mae": cv[name], **{"holdout_" + k: v for k, v in score.items()}})
                    results.append(
                        {"horizon": h, "model": name, "cv_mae": cv[name], **score, "run_id": run.info.run_id}
                    )
            champion = clone(candidates()[winner]).fit(dev[FEATURES], dev["count"])
            p = np.maximum(0, champion.predict(test[FEATURES]))
            part = test[["zone", "time", "origin", "count"]].copy()
            part["horizon"] = h
            part["prediction"] = p
            part["lower90"] = np.maximum(0, p - q[winner])
            part["upper90"] = p + q[winner]
            predictions.append(part)
            reference = dev[FEATURES].sample(min(3000, len(dev)), random_state=42)
            bundle = {
                "model": champion,
                "features": FEATURES,
                "horizon": h,
                "zones": sorted(raw.zone.unique().tolist()),
                "q90": q[winner],
                "winner": winner,
                "data_sha256": digest,
                "reference": reference,
                "train_end": str(dev.time.max()),
                "cv_mae": cv[winner],
            }
            version = hashlib.sha256((digest + winner + str(h)).encode()).hexdigest()[:12]
            filename = f"model-h{h}-{version}.joblib"
            joblib.dump(bundle, work / filename)
            reg = f"tlc-demand-h{h}"
            if not client.search_registered_models(filter_string=f"name='{reg}'"):
                client.create_registered_model(reg)
            mv = client.create_model_version(reg, (work / filename).resolve().as_uri())
            versions[str(h)] = {
                "file": filename,
                "sha256": hashlib.sha256((work / filename).read_bytes()).hexdigest(),
                "version": mv.version,
                "cv_mae": cv[winner],
            }
            bundles[str(h)] = {"winner": winner, "cv": cv, "q90": q[winner]}
    allp = pd.concat(predictions)
    allp.to_csv(work / "holdout_predictions.csv", index=False)
    pd.DataFrame(results).to_csv(work / "model_metrics.csv", index=False)
    grouped = []
    for (h, z), g in allp.groupby(["horizon", "zone"]):
        grouped.append(
            {
                "horizon": int(h),
                "zone": int(z),
                **metrics(g["count"], g.prediction),
                "interval90_coverage": float(((g["count"] >= g.lower90) & (g["count"] <= g.upper90)).mean()),
            }
        )
    report = {
        "neural_backend": neural_backend(),
        "source": "Real NYC TLC green taxi Jan–Mar 2025",
        "sha256": digest,
        "holdout_start": str(holdout_start),
        "protocol": "Rolling origin; direct horizons 1,6,24 hours; labels embargoed by horizon; CV precedes final holdout",
        "selection": bundles,
        "per_zone_horizon": grouped,
        "interval": "Empirical absolute CV residuals; no guaranteed coverage under temporal dependence",
        "metrics": results,
    }
    atomic_json(work / "training.json", report)
    # Promotion gate uses validation only, avoids looking at holdout for model selection.
    if any(v["cv_mae"] > bundles[h]["cv"]["seasonal_naive"] * 1.01 for h, v in versions.items()):
        raise ValueError("Promotion gate: CV regression vs seasonal baseline")
    atomic_json(
        work / "deployment.json",
        {"active": versions, "previous": None, "policy": "local validation gate; no production claim"},
    )
    return report
