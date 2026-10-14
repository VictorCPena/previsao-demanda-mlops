"""API local; carrega o artefato promovido a cada requisição para permitir rollback."""
import json
import os
from datetime import date, timedelta
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from data import ROOT, next_features
from backtest import rollout
from uncertainty import forecast_scenarios


class HistoryPoint(BaseModel):
    day: date
    units: float = Field(ge=0, le=1000000, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    product_id: str = Field(pattern=r"^P\d{2}$")
    day: date
    promo: int = Field(ge=0, le=1)
    history: list[HistoryPoint] = Field(min_length=28, max_length=730)


class PromotionScenario(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    weight: float = Field(gt=0, le=1, allow_inf_nan=False)
    promotions: list[int] = Field(min_length=1, max_length=14)


class ScenarioRequest(BaseModel):
    product_id: str = Field(pattern=r"^P\d{2}$")
    day: date
    history: list[HistoryPoint] = Field(min_length=28, max_length=730)
    horizon: int = Field(default=7, ge=1, le=14)
    scenarios: list[PromotionScenario] | None = None


class ForecastRequest(PredictionRequest):
    horizon: int = Field(default=7, ge=1, le=14)
    planned_promotions: list[int] = Field(min_length=1, max_length=14)


def create_app(work=None):
    workspace = Path(work or os.environ.get("DEMAND_WORKSPACE", ROOT / "work"))
    app = FastAPI(title="Demanda diária • laboratório local", version="1.0.0")

    @lru_cache(maxsize=8)
    def cached_bundle(filename: str, modified_ns: int, size: int):
        return joblib.load(filename)  # arquivos locais de origem confiável apenas

    def load_bundle():
        if not (workspace / "deployment.json").exists():
            raise HTTPException(status_code=503, detail="Execute o treinamento primeiro")
        deployment = json.loads((workspace / "deployment.json").read_text())
        path = workspace / deployment["active"]["file"]
        if not path.is_file():
            raise HTTPException(status_code=503, detail="Artefato promovido não encontrado")
        resolved = path.resolve()
        if not resolved.is_relative_to(workspace.resolve()):
            raise HTTPException(status_code=503, detail="Caminho do modelo não autorizado")
        stat = resolved.stat()
        return cached_bundle(str(resolved), stat.st_mtime_ns, stat.st_size)

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
                "health": "/health", "endpoints": ["/predict", "/forecast", "/forecast/scenarios"],
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
        result = rollout(request.product_id, history, request.day, request.planned_promotions,
                         lambda pid, hist, day, promo: predict(bundle, pid, hist, day, promo))
        return {"predictions": result, "version": bundle["version"], "mode": "recursive",
                "validation_note": "Para métricas por horizonte, execute make backtest; intervalos de 1 dia não se aplicam automaticamente."}

    @app.get("/live")
    def live():
        return {"status": "alive"}

    @app.post("/forecast/scenarios")
    def uncertain(request: ScenarioRequest):
        bundle = load_bundle()
        history = [{"day": h.day.isoformat(), "units": h.units} for h in request.history]
        supplied = [s.model_dump() for s in request.scenarios] if request.scenarios else None
        try:
            result = forecast_scenarios(request.product_id, history, request.day,
                       lambda pid, hist, day, promo: predict(bundle, pid, hist, day, promo),
                       scenarios=supplied, horizon=request.horizon)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {**result, "model_version": bundle["version"], "synthetic_training": bundle["synthetic"]}

    return app


app = create_app()
