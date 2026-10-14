"""API local; carrega o artefato promovido a cada requisição para permitir rollback."""
import json
import os
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from data import ROOT, next_features


class HistoryPoint(BaseModel):
    day: date
    units: float = Field(ge=0, le=1000000, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    product_id: str = Field(pattern=r"^P\d{2}$")
    day: date
    promo: int = Field(ge=0, le=1)
    history: list[HistoryPoint] = Field(min_length=28, max_length=730)


class ForecastRequest(PredictionRequest):
    horizon: int = Field(default=7, ge=1, le=14)
    planned_promotions: list[int] = Field(min_length=1, max_length=14)


def create_app(work=None):
    workspace = Path(work or os.environ.get("DEMAND_WORKSPACE", ROOT / "work"))
    app = FastAPI(title="Demanda diária • laboratório local", version="1.0.0")

    def load_bundle():
        if not (workspace / "deployment.json").exists():
            raise HTTPException(status_code=503, detail="Execute o treinamento primeiro")
        deployment = json.loads((workspace / "deployment.json").read_text())
        path = workspace / deployment["active"]["file"]
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Artefato promovido não encontrado")
        return joblib.load(path)

    def predict(bundle, product_id, history, day, promo):
        if product_id not in bundle["products"]:
            raise HTTPException(status_code=422, detail="Produto não visto no treinamento")
        try:
            X = next_features(product_id, history, day, promo)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with threadpool_limits(limits=2):
            value = float(np.maximum(0, bundle["model"].predict(X)[0]))
        return value

    @app.get("/")
    def root():
        return {"service": "previsao-demanda-mlops", "docs": "/docs",
                "health": "/health", "endpoints": ["/predict", "/forecast"],
                "mode": "laboratorio-local"}

    @app.get("/health")
    def health():
        """Prontidão exige modelo presente, não apenas um JSON de deployment."""
        deployment_path = workspace / "deployment.json"
        ready = False
        if deployment_path.is_file():
            try:
                deployment = json.loads(deployment_path.read_text())
                artifact = workspace / deployment["active"]["file"]
                ready = artifact.is_file() and artifact.resolve().is_relative_to(workspace.resolve())
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
                ready = False
        return {"status": "ok" if ready else "degraded", "model_ready": ready}

    @app.post("/predict")
    def one_day(request: PredictionRequest):
        bundle = load_bundle()
        history = [{"day": h.day.isoformat(), "units": h.units} for h in request.history]
        value = predict(bundle, request.product_id, history, request.day, request.promo)
        return {"product_id": request.product_id, "day": str(request.day), "prediction": value,
                "lower90_empirical": max(0, value-bundle["q90"]), "upper90_empirical": value+bundle["q90"],
                "model": bundle["name"], "version": bundle["version"], "synthetic_training": bundle["synthetic"],
                "interval_note": "Faixa empírica de resíduos de validação; cobertura não garantida."}

    @app.post("/forecast")
    def recursive(request: ForecastRequest):
        if len(request.planned_promotions) != request.horizon or any(p not in (0, 1) for p in request.planned_promotions):
            raise HTTPException(status_code=422, detail="Informe uma promoção 0/1 para cada dia do horizonte")
        if request.promo != request.planned_promotions[0]:
            raise HTTPException(status_code=422, detail="promo deve coincidir com a primeira promoção planejada")
        bundle = load_bundle()
        history = [{"day": h.day.isoformat(), "units": h.units} for h in request.history]
        result = []
        for step, promo in enumerate(request.planned_promotions):
            day = request.day + timedelta(days=step)
            value = predict(bundle, request.product_id, history, day, promo)
            result.append({"day": str(day), "prediction": value})
            history.append({"day": str(day), "units": value})
        return {"predictions": result, "version": bundle["version"], "mode": "recursive",
                "validation_note": "Demonstração funcional; métricas do holdout valem apenas para rolling 1 dia."}

    return app


app = create_app()
