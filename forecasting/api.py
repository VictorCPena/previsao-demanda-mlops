"""Local forecast API: strict observed-history contract and verified model artifacts."""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from forecasting.data import features, FEATURES


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time: datetime
    count: float = Field(ge=0, le=1_000_000, allow_inf_nan=False)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    zone: int = Field(ge=1, le=263)
    horizon: int = Field(ge=1, le=24)
    history: list[Observation] = Field(min_length=192, max_length=1000)


def create_app(work=None):
    root = Path(work or os.environ.get("FORECAST_WORK", "work/tlc"))
    app = FastAPI(title="TLC production-pattern forecast")
    from forecasting.observability import instrument

    instrument(app, 250000)
    cache = {}

    def bundle(h):
        try:
            deployment = json.loads((root / "deployment.json").read_text())
            info = deployment["active"][str(h)]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(503, "Model unavailable") from exc
        path = root / info["file"]
        if path.parent.resolve() != root.resolve():
            raise HTTPException(503, "Invalid artifact path")
        key = info["sha256"]
        if key not in cache:
            if hashlib.sha256(path.read_bytes()).hexdigest() != key:
                raise HTTPException(503, "Artifact integrity failure")
            cache[key] = joblib.load(
                path
            )  # Only trusted artifacts created by this project; not an upload endpoint.
        return cache[key]

    @app.get("/health")
    def health():
        bundle(1)
        return {"ready": True}

    @app.post("/predict")
    def predict(request: Request):
        if request.horizon not in (1, 6, 24):
            raise HTTPException(422, "Unsupported horizon")
        b = bundle(request.horizon)
        if request.zone not in b["zones"]:
            raise HTTPException(422, "Untrained region")
        try:
            history = pd.DataFrame(
                [{"time": o.time, "count": o.count, "zone": request.zone} for o in request.history]
            )
            history["time"] = pd.to_datetime(history.time, utc=True)
            if history.time.duplicated().any() or not history.time.is_monotonic_increasing:
                raise ValueError("History must be chronological and unique")
            if not (history.time == history.time.dt.floor("h")).all():
                raise ValueError("History must use exact hourly UTC boundaries")
            target = history.time.max() + pd.Timedelta(hours=request.horizon)
            extension = pd.DataFrame(
                {
                    "time": pd.date_range(history.time.max() + pd.Timedelta(hours=1), target, freq="h"),
                    "zone": request.zone,
                    "count": 0.0,
                }
            )
            row = features(pd.concat([history, extension], ignore_index=True), request.horizon).tail(1)
            if row.empty:
                raise ValueError("Insufficient contiguous history")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        value = float(np.maximum(0, b["model"].predict(row[FEATURES])[0]))
        return {
            "target": str(target),
            "horizon": request.horizon,
            "prediction": value,
            "lower90": max(0, value - b["q90"]),
            "upper90": value + b["q90"],
            "model": b["winner"],
            "data_sha256": b["data_sha256"],
        }

    return app


app = create_app()
